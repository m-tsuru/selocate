from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Callable

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
    z: float = 0
    bssid: str = ""
    ssid: str = ""
    rssi: float = 0
    freq: int = 0


@dataclasses.dataclass
class TraceState:
    """トレース状態を保持するデータクラス"""

    running: bool = False
    save_date: int = 0
    ap_count: int = 0
    total_traces: int = 0
    last_scan_time: float = 0
    scan_interval: float = 2.0


class TraceController:
    """WiFiトレースを非同期で制御するクラス"""

    def __init__(
        self, output_dir: str | None = None, interface_name: str | None = None
    ):
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.interface_name = interface_name
        self.scan_interval = SCAN_INTERVAL
        self.state = TraceState(scan_interval=self.scan_interval)
        self._task: asyncio.Task | None = None
        self._position_callback: Callable[[], tuple[float, float, float]] | None = None
        self._on_state_update: Callable[[TraceState, list[dict]], None] | None = None
        self._data: list[dict] = []

    def set_interface(self, interface_name: str | None):
        """使用するWi-Fiインターフェースを設定"""
        self.interface_name = interface_name

    def set_scan_interval(self, interval: float):
        """スキャン間隔を設定（秒）"""
        self.scan_interval = max(1.0, interval)  # 最小1秒
        self.state.scan_interval = self.scan_interval

    def set_position_callback(self, callback: Callable[[], tuple[float, float, float]]):
        """位置情報を取得するコールバックを設定"""
        self._position_callback = callback

    def on_state_update(self, callback: Callable[[TraceState, list[dict]], None]):
        """状態更新時のコールバックを設定"""
        self._on_state_update = callback

    def get_state(self) -> TraceState:
        """現在のトレース状態を取得"""
        return self.state

    def get_data(self) -> list[dict]:
        """収集したデータを取得"""
        return self._data

    async def start(self):
        """トレースを開始"""
        if self.state.running:
            return
        self.state.running = True
        self.state.save_date = int(time.time())
        self._data = []
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        """トレースを停止"""
        self.state.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def reset(self):
        """データをリセット"""
        self._data = []
        self.state.ap_count = 0
        self.state.total_traces = 0
        self.state.save_date = int(time.time())

    def save_data(self) -> str:
        """データをファイルに保存"""
        filename = self.output_dir / f"{TRACE_FILE_PREFIX}-{self.state.save_date}.json"
        with filename.open("w") as f:
            json.dump(self._data, f, indent=2)
        return str(filename)

    def load_data(self, filename: str) -> list[dict]:
        """データをファイルから読み込む"""
        with Path(filename).open("r") as f:
            self._data = json.load(f)
        self.state.total_traces = len(self._data)
        return self._data

    async def _loop(self):
        """メインループ"""
        while self.state.running:
            try:
                networks = await asyncio.to_thread(scan, self.interface_name)

                # 位置情報を取得
                x, y, z = 0.0, 0.0, 0.0
                if self._position_callback:
                    x, y, z = self._position_callback()

                # トレースデータを追加
                current_time = int(time.time())
                for n in networks:
                    trace = Trace(
                        t=current_time,
                        x=x,
                        y=y,
                        z=z,
                        bssid=n.get("bssid", ""),
                        ssid=n.get("ssid", ""),
                        rssi=n.get("signal", 0),
                        freq=n.get("freq", 0),
                    )
                    self._data.append(dataclasses.asdict(trace))

                self.state.ap_count = len(networks)
                self.state.total_traces = len(self._data)
                self.state.last_scan_time = time.time()

                if self._on_state_update:
                    self._on_state_update(self.state, networks)

                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                print(f"Trace loop error: {e}")
                await asyncio.sleep(self.scan_interval)


# シングルトンインスタンス
_trace_controller: TraceController | None = None


def get_trace_controller(interface_name: str | None = None) -> TraceController:
    """トレースコントローラーのシングルトンを取得

    Args:
        interface_name: Wi-Fiインターフェース名（初回呼び出し時のみ有効）
    """
    global _trace_controller
    if _trace_controller is None:
        _trace_controller = TraceController(interface_name=interface_name)
    return _trace_controller
