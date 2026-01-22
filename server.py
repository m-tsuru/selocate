import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lib.scan import list_interfaces
from lib.serialcon import get_serial_controller
from lib.trace import get_trace_controller

# 環境変数からWi-Fiインターフェース名を取得
WIFI_INTERFACE = os.getenv("WIFI_INTERFACE", None)


class MotorCommand(BaseModel):
    direction_power: int = 0
    wheel_power: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    # 起動時の処理
    serial_ctrl = get_serial_controller()
    trace_ctrl = get_trace_controller(interface_name=WIFI_INTERFACE)

    if WIFI_INTERFACE:
        print(f"Using Wi-Fi interface: {WIFI_INTERFACE}")
    else:
        print("Using default Wi-Fi interface")

    # トレースコントローラーに位置情報コールバックを設定
    trace_ctrl.set_position_callback(
        lambda: (
            serial_ctrl.get_state().x,
            serial_ctrl.get_state().y,
            serial_ctrl.get_state().z,
        )
    )

    yield

    # 終了時の処理
    await serial_ctrl.stop()
    await trace_ctrl.stop()


app = FastAPI(lifespan=lifespan)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter()


# === シリアル通信 API ===
@api.post("/serial/start")
async def serial_start():
    """シリアル通信を開始"""
    try:
        ctrl = get_serial_controller()
        await ctrl.start()
        return {"status": "ok", "message": "Serial started"}
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


@api.post("/serial/stop")
async def serial_stop():
    """シリアル通信を停止"""
    ctrl = get_serial_controller()
    await ctrl.stop()
    return {"status": "ok", "message": "Serial stopped"}


@api.get("/serial/state")
async def serial_state():
    """シリアル通信の状態を取得"""
    ctrl = get_serial_controller()
    state = ctrl.get_state()
    return {
        "running": ctrl.running,
        "state": asdict(state),
    }


@api.post("/serial/motor")
async def serial_motor(cmd: MotorCommand):
    """モーター制御"""
    ctrl = get_serial_controller()
    ctrl.set_motor(cmd.direction_power, cmd.wheel_power)
    return {
        "status": "ok",
        "direction_power": cmd.direction_power,
        "wheel_power": cmd.wheel_power,
    }


@api.post("/serial/reset")
async def serial_reset():
    """位置をリセット"""
    ctrl = get_serial_controller()
    ctrl.reset_position()
    return {"status": "ok", "message": "Position reset"}


# === Wi-Fi インターフェース API ===
@api.get("/wifi/interfaces")
async def wifi_interfaces():
    """利用可能なWi-Fiインターフェース一覧を取得"""
    try:
        interfaces = list_interfaces()
        return {
            "interfaces": interfaces,
            "current": WIFI_INTERFACE,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


@api.post("/wifi/interface")
async def wifi_set_interface(interface_name: str | None = None):
    """使用するWi-Fiインターフェースを変更"""
    ctrl = get_trace_controller()
    ctrl.set_interface(interface_name)
    return {"status": "ok", "interface": interface_name}


# === トレース API ===
@api.post("/trace/start")
async def trace_start():
    """トレースを開始"""
    try:
        ctrl = get_trace_controller()
        await ctrl.start()
        return {"status": "ok", "message": "Trace started"}
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


@api.post("/trace/stop")
async def trace_stop():
    """トレースを停止"""
    ctrl = get_trace_controller()
    await ctrl.stop()
    return {"status": "ok", "message": "Trace stopped"}


@api.get("/trace/state")
async def trace_state():
    """トレースの状態を取得"""
    ctrl = get_trace_controller()
    state = ctrl.get_state()
    return {
        "running": state.running,
        "ap_count": state.ap_count,
        "total_traces": state.total_traces,
        "save_date": state.save_date,
        "scan_interval": state.scan_interval,
    }


@api.post("/trace/interval")
async def trace_set_interval(interval: float):
    """スキャン間隔を設定（秒）"""
    ctrl = get_trace_controller()
    ctrl.set_scan_interval(interval)
    return {"status": "ok", "scan_interval": ctrl.scan_interval}


@api.get("/trace/data")
async def trace_data():
    """収集したデータを取得"""
    ctrl = get_trace_controller()
    return {"data": ctrl.get_data()}


@api.post("/trace/reset")
async def trace_reset():
    """トレースデータをリセット"""
    ctrl = get_trace_controller()
    ctrl.reset()
    return {"status": "ok", "message": "Trace data reset"}


@api.post("/trace/save")
async def trace_save():
    """トレースデータを保存"""
    ctrl = get_trace_controller()
    filename = ctrl.save_data()
    return {"status": "ok", "filename": filename}


@api.get("/trace/download")
async def trace_download():
    """トレースデータをダウンロード"""
    ctrl = get_trace_controller()
    filename = ctrl.save_data()
    return FileResponse(
        filename, media_type="application/json", filename=Path(filename).name
    )


@api.post("/trace/upload")
async def trace_upload(file: UploadFile):
    """チャレンジデータをアップロード"""
    try:
        content = await file.read()
        # 一時ファイルに保存
        temp_path = Path(f"uploads/{file.filename}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(content)
        return {"status": "ok", "filename": str(temp_path)}
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


# APIルーターを登録
app.include_router(api, prefix="/api")

# 静的ファイル配信
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse("static/index.html")
