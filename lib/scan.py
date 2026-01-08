import time

import pywifi

SCAN_TIME = 2

w = pywifi.PyWiFi()
iface = w.interfaces()[0]


def scan():
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
    ns = scan()
    print(ns)
