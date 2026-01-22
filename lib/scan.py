import time

import pywifi

SCAN_TIME = 2


def get_interface(
    name: str | None = None, index: int = 0
) -> pywifi.interfaces.Interface:
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


def scan(interface_name: str | None = None) -> list[dict]:
    """Wi-Fiネットワークをスキャンする

    Args:
        interface_name: 使用するインターフェース名（省略時は最初のインターフェース）

    Returns:
        スキャン結果のリスト
    """
    iface = get_interface(name=interface_name)
    iface.scan()
    time.sleep(SCAN_TIME)
    res = iface.scan_results()
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
        for r in res
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
