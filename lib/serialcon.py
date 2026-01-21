import os

import dotenv
import numpy as np
import serial

# === パラメータ設定（調整可能な変数） ===
# シリアル通信設定
EXPECTED_ELEMENTS = 8  # パース時の期待要素数
BAUD_RATE = 9600  # シリアル通信のボーレート

# カルマンフィルター設定
KALMAN_PROCESS_VARIANCE = 1e-4  # プロセスノイズ共分散
KALMAN_MEASUREMENT_VARIANCE = 1e-1  # 観測ノイズ共分散
KALMAN_RESET_COVARIANCE = 0.1  # リセット時の誤差共分散

# 静止判定設定
STATIONARY_THRESHOLD = 0.5  # 静止判定の閾値（重力加速度との差）
GRAVITY = 9.80665  # 重力加速度 [m/s²]

# 加速度ノイズ除去設定
ACC_MOTION_THRESHOLD = 0.3  # 移動加速度の最小閾値 [m/s²]
VELOCITY_THRESHOLD = 0.05  # 速度ノイズの閾値 [m/s]
ACC_THRESHOLD = 0.1  # 加速度の閾値 [m/s²]

# その他の設定
ERROR_RANGE = 0  # 従来のエラー範囲（未使用の可能性あり）


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
    PORT: str | None = os.getenv("PORT")
else:
    PORT = "/dev/serial0"

try:
    s: serial.Serial = serial.Serial(PORT)
    s.baudrate: int = BAUD_RATE
    vx, vy, vz, x, y, z, before_t = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # 各軸にカルマンフィルターを初期化
    kf_x = KalmanFilter(
        process_variance=KALMAN_PROCESS_VARIANCE,
        measurement_variance=KALMAN_MEASUREMENT_VARIANCE,
    )
    kf_y = KalmanFilter(
        process_variance=KALMAN_PROCESS_VARIANCE,
        measurement_variance=KALMAN_MEASUREMENT_VARIANCE,
    )
    kf_z = KalmanFilter(
        process_variance=KALMAN_PROCESS_VARIANCE,
        measurement_variance=KALMAN_MEASUREMENT_VARIANCE,
    )

    while True:
        raw: str = s.readline().decode("utf-8").strip()
        success, t, ax, ay, az, _, _, _ = parse_serial(raw)
        if not success:
            continue
        t, ax, ay, az = t / 1000, ax or 0.0, ay or 0.0, az or 0.0  # ty:ignore[unsupported-operator]
        dt = t - before_t
        if dt > 0:  # 最初のイテレーションをスキップ
            before_t = t
            ax, ay, az = _acc(ax, ay, az)

            # 重力加速度の大きさをチェック
            total_acc = (ax**2 + ay**2 + az**2) ** 0.5

            # 総加速度が重力加速度に近い場合、実際の移動加速度はゼロとみなす
            # （センサーが静止しているか等速直線運動中）
            is_stationary = abs(total_acc - GRAVITY) < STATIONARY_THRESHOLD

            if is_stationary:
                # 重力成分のみで移動加速度なし
                ax_motion, ay_motion, az_motion = 0.0, 0.0, 0.0
            else:
                # 重力を考慮した移動加速度（簡易的な処理）
                ax_motion = ax if abs(ax) > ACC_MOTION_THRESHOLD else 0.0
                ay_motion = ay if abs(ay) > ACC_MOTION_THRESHOLD else 0.0
                az_motion = (
                    (az - GRAVITY) if abs(az - GRAVITY) > ACC_MOTION_THRESHOLD else 0.0
                )

            # カルマンフィルタで加速度をフィルタリング
            kf_x.predict(dt)
            kf_x.update(ax_motion)
            x_filtered, vx, ax_filtered = kf_x.get_state()

            kf_y.predict(dt)
            kf_y.update(ay_motion)
            y_filtered, vy, ay_filtered = kf_y.get_state()

            kf_z.predict(dt)
            kf_z.update(az_motion)
            z_filtered, vz, az_filtered = kf_z.get_state()

            # 静止状態の判定と状態のリセット
            if is_stationary:
                # 速度と加速度を完全にリセット
                kf_x.x[1] = 0.0  # 速度
                kf_x.x[2] = 0.0  # 加速度
                kf_x.P[1, 1] = KALMAN_RESET_COVARIANCE  # 速度の誤差共分散をリセット
                kf_x.P[2, 2] = KALMAN_RESET_COVARIANCE  # 加速度の誤差共分散をリセット
                vx = 0.0
                ax_filtered = 0.0

                kf_y.x[1] = 0.0
                kf_y.x[2] = 0.0
                kf_y.P[1, 1] = KALMAN_RESET_COVARIANCE
                kf_y.P[2, 2] = KALMAN_RESET_COVARIANCE
                vy = 0.0
                ay_filtered = 0.0

                kf_z.x[1] = 0.0
                kf_z.x[2] = 0.0
                kf_z.P[1, 1] = KALMAN_RESET_COVARIANCE
                kf_z.P[2, 2] = KALMAN_RESET_COVARIANCE
                vz = 0.0
                az_filtered = 0.0
            else:
                # 動いている時でも、非常に小さい速度はノイズとして除去
                if abs(vx) < VELOCITY_THRESHOLD:
                    vx = 0.0
                    kf_x.x[1] = 0.0
                    kf_x.P[1, 1] = KALMAN_RESET_COVARIANCE
                if abs(vy) < VELOCITY_THRESHOLD:
                    vy = 0.0
                    kf_y.x[1] = 0.0
                    kf_y.P[1, 1] = KALMAN_RESET_COVARIANCE
                if abs(vz) < VELOCITY_THRESHOLD:
                    vz = 0.0
                    kf_z.x[1] = 0.0
                    kf_z.P[1, 1] = KALMAN_RESET_COVARIANCE

            # 位置は積算（累積）するため、リセットしない
            x, y, z = x_filtered, y_filtered, z_filtered

            print(f"time: {t:.2f}")
            print(
                f"acc: {ax_filtered:.3f}, {ay_filtered:.3f}, {az_filtered:.3f}, "
                f"vel: {vx:.3f}, {vy:.3f}, {vz:.3f}, "
                f"pos: {x:.3f}, {y:.3f}, {z:.3f}"
            )
        else:
            before_t = t
except serial.SerialException as e:
    print(f"Serial error: {e}")
except Exception as e:
    print(f"Error: {e}")
    s.close()
else:
    s.close()
