from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .camera_manager import CameraManager
from .schemas import StartRequest, StartResponse, StatusResponse


app = FastAPI(title="Target Hit MVP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

camera_manager = CameraManager(Path(__file__).resolve().parent)
camera_manager.session_manager.cleanup_all_sessions()


@app.post("/start", response_model=StartResponse)
async def start_system(payload: StartRequest) -> StartResponse:
    ok, message = camera_manager.start(payload.camera_source, payload.target_mode)
    status = "ok" if ok else "error"
    return StartResponse(status=status, message=message)


@app.post("/stop", response_model=StartResponse)
async def stop_system() -> StartResponse:
    ok, message = camera_manager.stop()
    status = "ok" if ok else "error"
    return StartResponse(status=status, message=message)


@app.post("/check_now", response_model=StartResponse)
async def check_now() -> StartResponse:
    ok, message = camera_manager.request_check()
    status = "ok" if ok else "error"
    return StartResponse(status=status, message=message)


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    return StatusResponse(**camera_manager.get_status())


@app.get("/cameras")
async def cameras() -> dict:
    return {"cameras": camera_manager.list_available_cameras()}


@app.websocket("/ws")
async def ws_feed(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = camera_manager.get_latest_payload()
            if payload is None:
                payload = {
                    "frame": "",
                    "target_type": camera_manager.target_type,
                    "status": camera_manager.status,
                    "hit_detected": False,
                    "bbox": None,
                    "last_score": camera_manager.score_engine.state.last_score,
                    "total_score": camera_manager.score_engine.state.total_score,
                    "tries": camera_manager.score_engine.state.tries,
                }
            await websocket.send_json(payload)
            await asyncio.sleep(0.12)
    except WebSocketDisconnect:
        return
