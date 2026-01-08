import dataclasses
import json
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.scan import scan

SCAN_INTERVAL = 2  # seconds


@dataclasses.dataclass
class Trace:
    t: int = -1
    x: float = 0
    y: float = 0
    bssid: str = ""
    ssid: str = ""
    rssi: float = 0
    freq: int = 0


try:
    while True:
        networks = scan()
        res = [
            Trace(
                t=int(time.time()),
                x=0,
                y=0,
                bssid=n["bssid"],
                ssid=n["ssid"],
                rssi=n["signal"],
                freq=n["freq"],
            )
            for n in networks
        ]
        for n in networks:
            trace = Trace(
                t=int(time.time()),
                bssid=n["bssid"],
                ssid=n["ssid"],
                rssi=n["signal"],
                freq=n["freq"],
            )
            print(
                json.dumps([dataclasses.asdict(trace) for trace in res]),
            )
        time.sleep(SCAN_INTERVAL)
except KeyboardInterrupt:
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
