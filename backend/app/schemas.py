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
    exposure_bias: float = 0.0


class ExposureRequest(BaseModel):
    """Manual exposure trim: -1 darker, 0 neutral, +1 brighter (software + USB)."""

    bias: float = Field(default=0.0, ge=-1.0, le=1.0)


class RoiRequest(BaseModel):
    """Normalised (0-1) bounding box drawn by the user on the live feed.

    x, y are the top-left corner; w, h are width and height — all as
    fractions of the displayed frame dimensions so they work regardless
    of stream resolution.
    """

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)
