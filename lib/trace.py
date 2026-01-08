import dataclasses
import json
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.scan import scan

SCAN_INTERVAL = 2  # seconds
TRACE_FILE_PREFIX = "trace"


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
    save_date = int(time.time())
    while True:
        networks = scan()

        # Read existing data
        if (
            Path(f"{TRACE_FILE_PREFIX}-{save_date}.json").exists()
            and Path(f"{TRACE_FILE_PREFIX}-{save_date}.json").stat().st_size > 0
        ):
            with Path(f"{TRACE_FILE_PREFIX}-{save_date}.json").open("r") as f:
                data = json.load(f)
        else:
            data = []

        # Add new traces
        for n in networks:
            trace = Trace(
                t=int(time.time()),
                x=0,
                y=0,
                bssid=n["bssid"],
                ssid=n["ssid"],
                rssi=n["signal"],
                freq=n["freq"],
            )
            data.append(dataclasses.asdict(trace))

        # Write back
        with Path(TRACE_FILE).open("w") as f:
            json.dump(data, f, indent=2)

        time.sleep(SCAN_INTERVAL)
except KeyboardInterrupt:
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
