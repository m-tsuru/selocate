from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import dotenv
import numpy as np
import serial

# === パラメータ設定（調整可能な変数） ===
# シリアル通信設定
EXPECTED_ELEMENTS = 8  # パース時の期待要素数
BAUD_RATE = 9600  # シリアル通信のボーレート

# カルマンフィルター設定
KALMAN_PROCESS_VARIANCE = 1e-3  # プロセスノイズ共分散（上げると追従性向上）
KALMAN_MEASUREMENT_VARIANCE = 5e-2  # 観測ノイズ共分散（下げると感度向上）
KALMAN_RESET_COVARIANCE = 0.1  # リセット時の誤差共分散

# 静止判定設定
STATIONARY_THRESHOLD = (
    0.3  # 静止判定の閾値（重力加速度との差）（下げると動き検出しやすい）
)
GRAVITY = 9.80665  # 重力加速度 [m/s²]

# 加速度ノイズ除去設定
ACC_MOTION_THRESHOLD = 0.1  # 移動加速度の最小閾値 [m/s²]（下げると遅い動き検出）
VELOCITY_THRESHOLD = 0.02  # 速度ノイズの閾値 [m/s]（下げると遅い速度も保持）
ACC_THRESHOLD = 0.05  # 加速度の閾値 [m/s²]

# 駆動制御設定
DIRECTION = (1, 1, -1, 1)  # 正転 (1), 逆転 (-1) のタプル
WHEEL_PORT = (False, True, True, False)  # ホイールのポート割り当て
DIRECTION_PORT = (True, False, False, False)  # 方向制御のポート割り当て


# その他の設定
ERROR_RANGE = 0  # 従来のエラー範囲（未使用の可能性あり）


@dataclass
class MotionState:
    """モーション状態を保持するデータクラス"""

    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    t: float = 0.0
    is_stationary: bool = True


@dataclass
class MotorControl:
    """モーター制御の状態"""

    direction_power: int = 0
    wheel_power: int = 0


# カスタム例外クラス
class SerialParseError(ValueError):
    """シリアルデータのパースエラー"""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Serial parse error: {reason}")


class InvalidElementCountError(SerialParseError):
    def __init__(self, count: int) -> None:
        super().__init__(
            f"Invalid number of elements: expected {EXPECTED_ELEMENTS}, got {count}",
        )


class InvalidStartCharacterError(SerialParseError):
    def __init__(self, char: str) -> None:
        super().__init__(f"Invalid start character: expected '/', got '{char}'")


class InvalidValueError(SerialParseError):
    def __init__(self, field: str, value: str) -> None:
        super().__init__(f"Invalid value for {field}: '{value}' is not a decimal")


class KalmanFilter:
    """カルマンフィルターの実装"""

    def __init__(
        self, process_variance: float = 1e-5, measurement_variance: float = 1e-2
    ):
        # 状態変数: [位置, 速度, 加速度]
        self.x = np.zeros(3)  # 状態ベクトル
        self.P = np.eye(3) * 1.0  # 誤差共分散行列

        # プロセスノイズと観測ノイズ
        self.Q = np.eye(3) * process_variance  # プロセスノイズ共分散
        self.R = measurement_variance  # 観測ノイズ共分散

    def predict(self, dt: float):
        """予測ステップ"""
        # 状態遷移行列
        F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]])

        # 状態予測
        self.x = F @ self.x
        # 誤差共分散予測
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement: float):
        """更新ステップ"""
        # 観測行列（加速度を観測）
        H = np.array([[0, 0, 1]])

        # カルマンゲイン
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S

        # 状態更新
        y = measurement - H @ self.x  # 観測残差
        self.x = self.x + K.flatten() * y

        # 誤差共分散更新
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

    def get_state(self) -> tuple[float, float, float]:
        """位置、速度、加速度を返す"""
        return self.x[0], self.x[1], self.x[2]


def _acc(ax: float, ay: float, az: float) -> tuple[float, float, float]:
    def __acc(n: float) -> float:
        return (n / 16384) * 9.80665

    return __acc(ax), __acc(ay), __acc(az)


def _velo(
    v0x: float, v0y: float, v0z: float, ax: float, ay: float, az: float, dt: float
) -> tuple[float, float, float, bool]:
    total_acc = (ax**2 + ay**2 + az**2) ** ERROR_RANGE

    # 総加速度が重力加速度に近い場合は静止状態と判断
    if abs(total_acc - 9.80665) < 3:
        return 0.0, 0.0, 0.0, True

    def __velo(v0: float, a: float, dt: float) -> float:
        if abs(a) < ERROR_RANGE:
            return v0
        return v0 + a * dt

    x, y, z, b = __velo(v0x, ax, dt), __velo(v0y, ay, dt), __velo(v0z, az, dt), False

    return x, y, z, b


def _dist(
    x: float,
    y: float,
    z: float,
    v0x: float,
    v0y: float,
    v0z: float,
    ax: float,
    ay: float,
    az: float,
    dt: float,
) -> tuple[float, float, float]:
    def __dist(x0: float, v0: float, a: float, dt: float) -> float:
        if abs(a) < 0.5:
            return x0 + v0 * dt
        return x0 + v0 * dt + 0.5 * a * dt**2

    return __dist(x, v0x, ax, dt), __dist(y, v0y, ay, dt), __dist(z, v0z, az, dt)


def run_motor(direction_power: int, wheel_power: int) -> str:
    s = ""
    for i in range(4):
        if WHEEL_PORT[i]:
            s += str(wheel_power * DIRECTION[i])
        elif DIRECTION_PORT[i]:
            s += str(direction_power * DIRECTION[i])
        else:
            s += "0"
        if i < 3:
            s += " "
    print(s)
    return s


def parse_serial(
    s: str,
) -> tuple[
    bool,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    try:
        raw: list[str] = s.split()
        if len(raw) != EXPECTED_ELEMENTS:
            raise InvalidElementCountError(len(raw))
        if raw[0] != "/":
            raise InvalidStartCharacterError(raw[0])

        # 各フィールドを浮動小数点数に変換してバリデーション
        try:
            t = float(raw[1])
        except ValueError:
            raise InvalidValueError("t", raw[1])
        try:
            x = float(raw[2])
        except ValueError:
            raise InvalidValueError("x", raw[2])
        try:
            y = float(raw[3])
        except ValueError:
            raise InvalidValueError("y", raw[3])
        try:
            z = float(raw[4])
        except ValueError:
            raise InvalidValueError("z", raw[4])
        try:
            gx = float(raw[5])
        except ValueError:
            raise InvalidValueError("gx", raw[5])
        try:
            gy = float(raw[6])
        except ValueError:
            raise InvalidValueError("gy", raw[6])
        try:
            gz = float(raw[7])
        except ValueError:
            raise InvalidValueError("gz", raw[7])
    except SerialParseError as e:
        print(f"Parsing error: {e}")
        return False, None, None, None, None, None, None, None
    else:
        return True, t, x, y, z, gx, gy, gz


dotenv.load_dotenv()

if "PORT" in dotenv.dotenv_values():
    SERIAL_PORT: str | None = os.getenv("PORT")
else:
    SERIAL_PORT = "/dev/serial0"


class SerialController:
    """シリアル通信を非同期で制御するクラス"""

    def __init__(self, port: str | None = None):
        self.port = port or SERIAL_PORT
        self.serial: serial.Serial | None = None
        self.running = False
        self.state = MotionState()
        self.motor = MotorControl()
        self._last_motor = MotorControl()  # 前回送信したモーター値
        self._motor_send_interval = 0.1  # モーター送信間隔（秒）
        self._last_motor_send = 0.0  # 最後にモーターコマンドを送信した時刻
        self.before_t = 0.0
        self._task: asyncio.Task | None = None
        self._on_state_update: Callable[[MotionState], None] | None = None

        # カルマンフィルター
        self.kf_x = KalmanFilter(
            process_variance=KALMAN_PROCESS_VARIANCE,
            measurement_variance=KALMAN_MEASUREMENT_VARIANCE,
        )
        self.kf_y = KalmanFilter(
            process_variance=KALMAN_PROCESS_VARIANCE,
            measurement_variance=KALMAN_MEASUREMENT_VARIANCE,
        )
        self.kf_z = KalmanFilter(
            process_variance=KALMAN_PROCESS_VARIANCE,
            measurement_variance=KALMAN_MEASUREMENT_VARIANCE,
        )

    def on_state_update(self, callback: Callable[[MotionState], None]):
        """状態更新時のコールバックを設定"""
        self._on_state_update = callback

    def set_motor(self, direction_power: int, wheel_power: int):
        """モーターの制御値を設定"""
        self.motor.direction_power = direction_power
        self.motor.wheel_power = wheel_power

    def get_state(self) -> MotionState:
        """現在のモーション状態を取得"""
        return self.state

    def reset_position(self):
        """位置をリセット"""
        self.state.x = 0.0
        self.state.y = 0.0
        self.state.z = 0.0
        self.kf_x.x[0] = 0.0
        self.kf_y.x[0] = 0.0
        self.kf_z.x[0] = 0.0

    async def start(self):
        """シリアル通信を開始"""
        if self.running:
            return
        try:
            self.serial = serial.Serial(self.port)
            self.serial.baudrate = BAUD_RATE
            self.running = True
            self._task = asyncio.create_task(self._loop())
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            raise

    async def stop(self):
        """シリアル通信を停止"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.serial and self.serial.is_open:
            self.serial.close()

    async def _loop(self):
        """メインループ"""
        import time

        while self.running:
            try:
                # シリアルデータを受信して処理
                if self.serial and self.serial.in_waiting > 0:
                    raw = self.serial.readline().decode("utf-8").strip()
                    await self._process_data(raw)

                # モーター制御：値が変わったとき、または一定間隔で送信
                now = time.time()
                motor_changed = (
                    self.motor.direction_power != self._last_motor.direction_power
                    or self.motor.wheel_power != self._last_motor.wheel_power
                )
                interval_elapsed = (
                    now - self._last_motor_send
                ) >= self._motor_send_interval

                if (
                    self.serial
                    and self.serial.is_open
                    and (motor_changed or interval_elapsed)
                ):
                    props = run_motor(
                        self.motor.direction_power, self.motor.wheel_power
                    )
                    self.serial.write((props + "\n").encode())
                    self._last_motor.direction_power = self.motor.direction_power
                    self._last_motor.wheel_power = self.motor.wheel_power
                    self._last_motor_send = now

                await asyncio.sleep(0.01)  # 10ms待機
            except Exception as e:
                print(f"Loop error: {e}")
                await asyncio.sleep(0.1)

    async def _process_data(self, raw: str):
        """受信データを処理"""
        success, t, ax, ay, az, _, _, _ = parse_serial(raw)
        if not success:
            return

        t, ax, ay, az = t / 1000, ax or 0.0, ay or 0.0, az or 0.0
        dt = t - self.before_t

        if dt > 0:
            self.before_t = t
            ax, ay, az = _acc(ax, ay, az)

            total_acc = (ax**2 + ay**2 + az**2) ** 0.5
            is_stationary = abs(total_acc - GRAVITY) < STATIONARY_THRESHOLD

            if is_stationary:
                ax_motion, ay_motion, az_motion = 0.0, 0.0, 0.0
            else:
                ax_motion = ax if abs(ax) > ACC_MOTION_THRESHOLD else 0.0
                ay_motion = ay if abs(ay) > ACC_MOTION_THRESHOLD else 0.0
                az_motion = (
                    (az - GRAVITY) if abs(az - GRAVITY) > ACC_MOTION_THRESHOLD else 0.0
                )

            # カルマンフィルタ処理
            self.kf_x.predict(dt)
            self.kf_x.update(ax_motion)
            x_filtered, vx, ax_filtered = self.kf_x.get_state()

            self.kf_y.predict(dt)
            self.kf_y.update(ay_motion)
            y_filtered, vy, ay_filtered = self.kf_y.get_state()

            self.kf_z.predict(dt)
            self.kf_z.update(az_motion)
            z_filtered, vz, az_filtered = self.kf_z.get_state()

            # 静止状態処理
            if is_stationary:
                self._reset_velocity()
                vx, vy, vz = 0.0, 0.0, 0.0
                ax_filtered, ay_filtered, az_filtered = 0.0, 0.0, 0.0
            else:
                vx, vy, vz = self._apply_velocity_threshold(vx, vy, vz)

            # 状態を更新
            self.state = MotionState(
                ax=ax_filtered,
                ay=ay_filtered,
                az=az_filtered,
                vx=vx,
                vy=vy,
                vz=vz,
                x=x_filtered,
                y=y_filtered,
                z=z_filtered,
                t=t,
                is_stationary=is_stationary,
            )

            if self._on_state_update:
                self._on_state_update(self.state)
        else:
            self.before_t = t

    def _reset_velocity(self):
        """速度をリセット"""
        for kf in [self.kf_x, self.kf_y, self.kf_z]:
            kf.x[1] = 0.0
            kf.x[2] = 0.0
            kf.P[1, 1] = KALMAN_RESET_COVARIANCE
            kf.P[2, 2] = KALMAN_RESET_COVARIANCE

    def _apply_velocity_threshold(self, vx: float, vy: float, vz: float):
        """速度閾値を適用"""
        if abs(vx) < VELOCITY_THRESHOLD:
            vx = 0.0
            self.kf_x.x[1] = 0.0
            self.kf_x.P[1, 1] = KALMAN_RESET_COVARIANCE
        if abs(vy) < VELOCITY_THRESHOLD:
            vy = 0.0
            self.kf_y.x[1] = 0.0
            self.kf_y.P[1, 1] = KALMAN_RESET_COVARIANCE
        if abs(vz) < VELOCITY_THRESHOLD:
            vz = 0.0
            self.kf_z.x[1] = 0.0
            self.kf_z.P[1, 1] = KALMAN_RESET_COVARIANCE
        return vx, vy, vz


# シングルトンインスタンス
_serial_controller: SerialController | None = None


def get_serial_controller() -> SerialController:
    """シリアルコントローラーのシングルトンを取得"""
    global _serial_controller
    if _serial_controller is None:
        _serial_controller = SerialController()
    return _serial_controller
