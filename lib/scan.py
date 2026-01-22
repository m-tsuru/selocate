from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pywifi

if TYPE_CHECKING:
    from pywifi import iface as pywifi_iface

SCAN_TIME = 2


def get_interface(
    name: str | None = None,
    index: int = 0,
) -> pywifi_iface.Interface | None:
    """Wi-Fiインターフェースを取得する

    Args:
        name: インターフェース名（指定した場合は名前で検索）
        index: インターフェースのインデックス（nameが指定されていない場合に使用）

    Returns:
        Wi-Fiインターフェース

    Raises:
        RuntimeError: インターフェースが見つからない場合
        ValueError: 指定された名前のインターフェースが見つからない場合
    """
    w = pywifi.PyWiFi()
    interfaces = w.interfaces()

    if not interfaces:
        raise RuntimeError("No Wi-Fi interface found")

    if name:
        for iface in interfaces:
            if iface.name() == name:
                return iface
        available = [i.name() for i in interfaces]
        raise ValueError(f"Interface '{name}' not found. Available: {available}")

    if index >= len(interfaces):
        raise IndexError(
            f"Interface index {index} out of range (0-{len(interfaces) - 1})"
        )

    return interfaces[index]


def list_interfaces() -> list[str]:
    """利用可能なWi-Fiインターフェース名の一覧を取得"""
    w = pywifi.PyWiFi()
    return [iface.name() for iface in w.interfaces()]


# スキャン間隔を管理するためのグローバル変数
_last_scan_time: float = 0
_scan_lock = False


def scan(
    interface_name: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> list[dict]:
    """Wi-Fiネットワークをスキャンする

    Args:
        interface_name: 使用するインターフェース名（省略時は最初のインターフェース）
        max_retries: スキャン失敗時のリトライ回数
        retry_delay: リトライ間の待機時間（秒）

    Returns:
        スキャン結果のリスト
    """
    global _last_scan_time, _scan_lock

    # スキャン中の場合は前回の結果を返す
    if _scan_lock:
        iface = get_interface(name=interface_name)
        if iface:
            return _format_scan_results(iface.scan_results())
        return []

    # 前回のスキャンから十分な時間が経っていない場合は待機
    elapsed = time.time() - _last_scan_time
    if elapsed < SCAN_TIME:
        time.sleep(SCAN_TIME - elapsed)

    iface = get_interface(name=interface_name)
    if not iface:
        return []

    _scan_lock = True
    try:
        for attempt in range(max_retries):
            try:
                iface.scan()
                time.sleep(SCAN_TIME)
                _last_scan_time = time.time()
                return _format_scan_results(iface.scan_results())
            except Exception as e:
                error_msg = str(e)
                if "FAIL-BUSY" in error_msg or attempt < max_retries - 1:
                    print(f"Scan attempt {attempt + 1} failed, retrying...")
                    time.sleep(retry_delay)
                else:
                    print(f"Scan failed: {e}")
                    # 失敗しても前回の結果を返す
                    return _format_scan_results(iface.scan_results())
    finally:
        _scan_lock = False

    return []


def _format_scan_results(results) -> list[dict]:
    """スキャン結果をフォーマットする"""
    return [
        {
            "ssid": r.ssid,
            "bssid": r.bssid,
            "signal": r.signal,
            "freq": r.freq,
            "auth": r.auth,
            "cipher": r.cipher,
            "akm": r.akm,
        }
        for r in results
    ]


if __name__ == "__main__":
    import sys

    # コマンドライン引数でインターフェース名を指定可能
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            print("Available interfaces:", list_interfaces())
        else:
            ns = scan(interface_name=sys.argv[1])
            print(ns)
    else:
        print("Available interfaces:", list_interfaces())
        ns = scan()
        print(ns)
