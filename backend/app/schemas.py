from pydantic import BaseModel, Field
from typing import Literal


TargetMode = Literal["auto", "figure_1", "figure_2"]


class StartRequest(BaseModel):
    camera_source: str = Field(
        default="0",
        description="USB camera index (e.g. 0) or network stream URL (rtsp://, rtsps://, http://, https://)",
    )
    target_mode: TargetMode = "auto"


class StartResponse(BaseModel):
    status: str
    message: str


class StatusResponse(BaseModel):
    running: bool
    status: str
    camera_source: str | None = None
    target_mode: TargetMode = "auto"
    target_type: str = "unknown"
    tries: int = 0
    last_score: int = 0
    total_score: int = 0
    hit_detected: bool = False
    actual_resolution: str = ""
    motion: float = 0.0
    stable: bool = False
