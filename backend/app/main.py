from __future__ import annotations

import asyncio
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .camera_manager import CameraManager
from .schemas import ExposureRequest, RoiRequest, StartRequest, StartResponse, StatusResponse

_SESSION_DIR_RE = re.compile(r"^session_\d{8}_\d{6}$")
_GALLERY_FILE_RE = re.compile(r"^[\w.-]+$")


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


@app.post("/exposure", response_model=StartResponse)
async def set_exposure(payload: ExposureRequest) -> StartResponse:
    ok, message = camera_manager.set_manual_exposure(payload.bias)
    status = "ok" if ok else "error"
    return StartResponse(status=status, message=message)


@app.post("/capture_reference", response_model=StartResponse)
async def capture_reference() -> StartResponse:
    ok, message = camera_manager.request_reference_snap()
    status = "ok" if ok else "error"
    return StartResponse(status=status, message=message)


@app.post("/set_roi", response_model=StartResponse)
async def set_roi(payload: RoiRequest) -> StartResponse:
    ok, message = camera_manager.set_manual_roi(payload.x, payload.y, payload.w, payload.h)
    status = "ok" if ok else "error"
    return StartResponse(status=status, message=message)


@app.post("/clear_roi", response_model=StartResponse)
async def clear_roi() -> StartResponse:
    ok, message = camera_manager.clear_manual_roi()
    status = "ok" if ok else "error"
    return StartResponse(status=status, message=message)


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    return StatusResponse(**camera_manager.get_status())


@app.get("/cameras")
async def cameras() -> dict:
    return {"cameras": camera_manager.list_available_cameras()}


@app.get("/session/gallery")
async def session_gallery() -> dict:
    sessions = camera_manager.session_manager.list_gallery_sessions()
    return {"sessions": sessions}


@app.get("/session/{session_id}/file/{filename}")
async def session_file(session_id: str, filename: str) -> FileResponse:
    if not _SESSION_DIR_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    if not _GALLERY_FILE_RE.match(filename) or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    root = camera_manager.session_manager.root_dir.resolve()
    session_dir = (root / session_id).resolve()
    if not session_dir.is_dir() or session_dir.parent != root:
        raise HTTPException(status_code=404, detail="Session not found")
    full_path = (session_dir / filename).resolve()
    if full_path.parent != session_dir or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full_path, media_type="image/jpeg")


@app.websocket("/ws")
async def ws_feed(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = camera_manager.get_latest_payload()
            if payload is None:
                state = camera_manager.score_engine.state
                payload = {
                    "frame": "",
                    "target_type": camera_manager.target_type,
                    "status": camera_manager.status,
                    "hit_detected": False,
                    "bbox": None,
                    "last_score": state.last_score,
                    "total_score": state.total_score,
                    "tries": state.tries,
                    "hits": state.hits,
                    "misses": state.misses,
                    "hit_center": state.last_hit_center,
                    "target_center": state.last_target_center,
                    "offset_from_center": state.last_offset_px,
                    "exposure_bias": camera_manager.get_exposure_bias(),
                }
            await websocket.send_json(payload)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return
