import os

import dotenv
import serial

EXPECTED_ELEMENTS = 5
ERROR_RANGE = 0.4


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


def _acc(ax: float, ay: float, az: float) -> tuple[float, float, float]:
    def __acc(n: float) -> float:
        return (n / 16384) * 9.80665

    return __acc(ax), __acc(ay), __acc(az)


def _velo(
    v0x: float, v0y: float, v0z: float, ax: float, ay: float, az: float, dt: float
) -> tuple[float, float, float, bool]:
    total_acc = (ax**2 + ay**2 + az**2) ** ERROR_RANGE

    # 総加速度が重力加速度に近い場合は静止状態と判断
    if abs(total_acc - 9.80665) < 2:
        return 0.0, 0.0, 0.0, True

    def __velo(v0: float, a: float, dt: float) -> float:
        if abs(a) < ERROR_RANGE:
            return v0
        return v0 + a * dt

    return __velo(v0x, ax, dt), __velo(v0y, ay, dt), __velo(v0z, az, dt), False


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
) -> tuple[bool, float | None, float | None, float | None, float | None]:
    try:
        raw: list[str] = s.split()
        if len(raw) != EXPECTED_ELEMENTS:
            raise InvalidElementCountError(len(raw))
        if raw[0] != "/":
            raise InvalidStartCharacterError(raw[0])
        if not raw[1].lstrip("-").isnumeric():
            raise InvalidValueError("t", raw[1])
        if not raw[2].lstrip("-").isnumeric():
            raise InvalidValueError("x", raw[2])
        if not raw[3].lstrip("-").isnumeric():
            raise InvalidValueError("y", raw[3])
        if not raw[4].lstrip("-").isnumeric():
            raise InvalidValueError("z", raw[4])
        t = float(raw[1])
        x = float(raw[2])
        y = float(raw[3])
        z = float(raw[4])
    except SerialParseError as e:
        print(f"Parsing error: {e}")
        return False, None, None, None, None
    else:
        return True, t, x, y, z


dotenv.load_dotenv()

if "PORT" in dotenv.dotenv_values():
    PORT: str | None = os.getenv("PORT")
else:
    PORT = "/dev/serial0"
BAUD_RATE = 9600

try:
    s: serial.Serial = serial.Serial(PORT)
    s.baudrate: int = BAUD_RATE
    vx, vy, vz = 0.0, 0.0, 0.0
    x, y, z = 0.0, 0.0, 0.0
    before_t = 0.0
    while True:
        raw: str = s.readline().decode("utf-8").strip()
        success, t, ax, ay, az = parse_serial(raw)
        if not success:
            continue
        t, ax, ay, az = t / 1000.0, ax or 0.0, ay or 0.0, az or 0.0
        dt = t - before_t
        before_t = t
        ax, ay, az = _acc(ax, ay, az)
        vx, vy, vz, is_stationary = _velo(vx, vy, vz, ax, ay, az, dt)
        if not is_stationary:
            x, y, z = _dist(x, y, z, vx, vy, vz, ax, ay, az, dt)
        print(f"time: {t}, {x:.3f}, {y:.3f}, {z:.3f}")
except serial.SerialException as e:
    print(f"Serial error: {e}")
except Exception as e:
    print(f"Error: {e}")
    s.close()
else:
    s.close()
