# Wi‑Fi スキャン

依存関係をインストールして、スクリプトを実行します。

インストール:

```bash
python3 -m pip install -r requirements.txt
```

実行:

```bash
# インターフェース番号は省略可能（デフォルト 0）
python3 lib/scan.py [INTERFACE_INDEX]
```

スクリプトは周囲のアクセスポイントを検出して、SSID / BSSID / 信号強度を表示します。
