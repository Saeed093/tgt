from __future__ import annotations

import base64
import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .hit_detector import HitDetector
from .score_engine import ScoreEngine
from .session_manager import SessionManager
from .target_detector import TargetDetector

log = logging.getLogger("camera")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

# Resolutions we request — highest first.
RES_CANDIDATES = [
    (1920, 1080),
    (1280, 720),
    (640, 480),
]

MOTION_STILL_THRESHOLD = 2.5
STABLE_DWELL_SEC = 1.5
HIT_COOLDOWN_SEC = 1.0

# Live WebSocket feed: downscale large frames before JPEG to reduce macroblocking.
STREAM_MAX_EDGE = 1280
STREAM_JPEG_QUALITY = 84
DISK_JPEG_QUALITY = 93

# When the global mean luma is above this, scale the frame down slightly
# so IP streams / auto-exposure USB frames are not blown out.
OVEREXPOSE_MEAN_LUMA_GATE = 120.0
OVEREXPOSE_TARGET_MEAN = 105.0

# USB (DirectShow / MSMF): best-effort manual-ish exposure — values are
# driver-specific; we log what stuck.
USB_EXPOSURE_RELATIVE = -10.0
USB_GAIN = 8.0
USB_BRIGHTNESS = 118.0

# Network-stream reader tuning.
NETWORK_READ_TIMEOUT_SEC = 4.0          # block in main loop waiting for a fresh frame
NETWORK_RECONNECT_AFTER_SEC = 3.0       # if reader sees no frame for this long → reconnect
NETWORK_RECONNECT_BACKOFF_SEC = 1.5


class _NetworkFrameReader:
    """Continuously drains an RTSP / HTTP VideoCapture in a background thread.

    For network streams, OpenCV/FFmpeg buffers frames internally.  If the
    consumer (our processing loop) is slower than the camera's frame rate
    even slightly, that internal buffer fills up and ``cap.read()`` starts
    returning stale frames in bursts — visible as freezing followed by
    fast-forward stutters, and as macroblock 'pixelation' when partially
    decoded packets are flushed.

    This reader keeps the buffer drained by reading flat-out in a thread
    and exposing **only the most recent decoded frame** via a 1-slot
    queue.  The processing loop is then always operating on near-realtime
    frames and never blocks the camera.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._running = True
        self._stat_lock = threading.Lock()
        self._consecutive_failures = 0
        self._last_frame_ts = time.time()
        self._frames_total = 0
        self._thread = threading.Thread(
            target=self._loop, name="ip-cam-reader", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                ok, frame = self._cap.read()
            except Exception:
                ok, frame = False, None

            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                with self._stat_lock:
                    self._consecutive_failures += 1
                time.sleep(0.02)
                continue

            with self._stat_lock:
                self._consecutive_failures = 0
                self._last_frame_ts = time.time()
                self._frames_total += 1

            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass

    def read(self, timeout: float = NETWORK_READ_TIMEOUT_SEC):
        try:
            frame = self._q.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            return False, None

    def seconds_since_last_frame(self) -> float:
        with self._stat_lock:
            return time.time() - self._last_frame_ts

    def consecutive_failures(self) -> int:
        with self._stat_lock:
            return self._consecutive_failures

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class CameraManager:
    def __init__(self, app_root: Path) -> None:
        self.app_root = app_root
        self.target_detector = TargetDetector(app_root / "templates")
        self.hit_detector = HitDetector()
        self.score_engine = ScoreEngine()
        self.session_manager = SessionManager(app_root.parent / "temp")

        self.capture = None
        self._stream_reader: _NetworkFrameReader | None = None
        self.running = False
        self.status = "idle"
        self.camera_source: str | None = None
        self.target_mode = "auto"
        self.target_type = "unknown"
        self.reference_frame = None
        self.reference_saved = False
        self.latest_payload = None
        self.last_hit_ts = 0.0
        self._sticky_boxes: list[list[int]] | None = None
        self._last_hit_detail: dict | None = None
        self._all_hits_detail: list[dict] = []

        self.actual_resolution: str = ""
        self.motion: float = 0.0
        self.stable: bool = False

        self._prev_gray_small: np.ndarray | None = None
        self._last_motion_ts: float = 0.0
        self._manual_check_pending: bool = False

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        # -1 = darker, +1 = brighter (software gain on every frame; USB also nudges CAP_PROP_EXPOSURE).
        self._manual_exposure_bias = 0.0
        self._snap_reference_requested = False
        # Normalised (0-1) ROI drawn by the user on the live feed.
        # None means auto-segmentation; set via /set_roi.
        self._manual_roi: tuple[float, float, float, float] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_available_cameras(self, max_index: int = 4) -> list[dict]:
        return [{"index": idx, "label": f"Camera {idx}"} for idx in range(max_index)]

    def start(self, camera_source: str, target_mode: str) -> tuple[bool, str]:
        if self.running:
            return False, "Camera already running. Stop it first."

        self.score_engine.reset()
        self.target_mode = target_mode
        self.camera_source = camera_source
        self.target_type = "unknown"
        self.reference_frame = None
        self.reference_saved = False
        self.latest_payload = None
        self.last_hit_ts = 0.0
        self._sticky_boxes = None
        self._last_hit_detail = None
        self._all_hits_detail = []
        self.hit_detector.clear_mask_cache()
        self.actual_resolution = ""
        self.motion = 0.0
        self.stable = False
        self._prev_gray_small = None
        self._last_motion_ts = time.time()
        self._manual_check_pending = False
        self._manual_exposure_bias = 0.0
        self._snap_reference_requested = False
        self._manual_roi = None
        self.status = "starting"
        self.running = True

        self.session_manager.start_new_session()

        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        return True, "Camera starting"

    def stop(self) -> tuple[bool, str]:
        self.running = False

        if self._stream_reader is not None:
            try:
                self._stream_reader.stop()
            except Exception:
                pass
            self._stream_reader = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8)

        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None

        self.status = "idle"
        self.reference_frame = None
        self.reference_saved = False
        self.target_type = "unknown"
        self._sticky_boxes = None
        self._last_hit_detail = None
        self._all_hits_detail = []
        self.actual_resolution = ""
        self.motion = 0.0
        self.stable = False
        self._prev_gray_small = None
        self._manual_check_pending = False
        self._manual_exposure_bias = 0.0
        self._snap_reference_requested = False
        self._manual_roi = None
        self.hit_detector.clear_mask_cache()
        self._publish_idle_feed()
        log.info("Camera stopped")
        return True, "Camera stopped"

    def request_check(self) -> tuple[bool, str]:
        if not self.running:
            return False, "System not running"
        if self.reference_frame is None:
            return False, "Reference not captured yet"
        self._manual_check_pending = True
        log.info("Manual check queued")
        return True, "Check queued"

    def get_status(self) -> dict:
        state = self.score_engine.state
        return {
            "running": self.running,
            "status": self.status,
            "camera_source": self.camera_source,
            "target_mode": self.target_mode,
            "target_type": self.target_type,
            "tries": state.tries,
            "hits": state.hits,
            "misses": state.misses,
            "last_score": state.last_score,
            "total_score": state.total_score,
            "hit_detected": (state.hit_status == "hit"),
            "actual_resolution": self.actual_resolution,
            "motion": self.motion,
            "stable": self.stable,
            "exposure_bias": round(float(self._manual_exposure_bias), 4),
        }

    def get_exposure_bias(self) -> float:
        return float(self._manual_exposure_bias)

    def set_manual_exposure(self, bias: float) -> tuple[bool, str]:
        b = max(-1.0, min(1.0, float(bias)))
        with self._lock:
            self._manual_exposure_bias = b
        cap = self.capture
        src = self.camera_source
        if cap is not None and src is not None:
            parsed = self._parse_source(str(src))
            if not self._is_network_stream(parsed):
                ex = getattr(cv2, "CAP_PROP_EXPOSURE", None)
                if ex is not None:
                    mapped = float(USB_EXPOSURE_RELATIVE) + b * 7.0
                    try:
                        cap.set(ex, mapped)
                    except Exception:
                        pass
        return True, f"Exposure bias {b:+.2f}"

    def set_manual_roi(
        self, nx: float, ny: float, nw: float, nh: float
    ) -> tuple[bool, str]:
        """Set a normalised (0-1) ROI; resets the mask cache so it is rebuilt
        from the stored reference at the next check."""
        nx = max(0.0, min(1.0, float(nx)))
        ny = max(0.0, min(1.0, float(ny)))
        nw = max(0.01, min(1.0 - nx, float(nw)))
        nh = max(0.01, min(1.0 - ny, float(nh)))
        self._manual_roi = (nx, ny, nw, nh)
        self.hit_detector.clear_mask_cache()
        log.info("Manual ROI set: x=%.3f y=%.3f w=%.3f h=%.3f", nx, ny, nw, nh)
        return True, f"ROI set ({nx:.2f},{ny:.2f}) {nw:.2f}×{nh:.2f}"

    def clear_manual_roi(self) -> tuple[bool, str]:
        self._manual_roi = None
        self.hit_detector.clear_mask_cache()
        log.info("Manual ROI cleared — reverting to auto-segmentation")
        return True, "ROI cleared"

    def get_manual_roi(self) -> tuple[float, float, float, float] | None:
        return self._manual_roi

    def request_reference_snap(self) -> tuple[bool, str]:
        if not self.running:
            return False, "System not running"
        if self.status != "monitoring":
            return False, "Wait until monitoring is active"
        self._snap_reference_requested = True
        log.info("Manual reference snap requested")
        return True, "New reference will be taken from the next frame"

    def get_latest_payload(self) -> dict | None:
        """Return latest payload, encoding the BGR frame on demand.

        Encoding here (instead of inside the capture loop) means the
        capture thread is never blocked behind cv2.imencode / base64.
        It also caps real JPEG work to the WebSocket send rate.
        """
        with self._lock:
            if self.latest_payload is None:
                return None
            payload = dict(self.latest_payload)
            frame = payload.pop("_frame_bgr", None)
        payload["frame"] = self._encode_frame(frame) if frame is not None else ""
        return payload

    # ------------------------------------------------------------------
    # Camera open helpers
    # ------------------------------------------------------------------

    def _parse_source(self, source: str):
        source = str(source).strip()
        return int(source) if source.isdigit() else source

    @staticmethod
    def _is_network_stream(source: str | int) -> bool:
        if isinstance(source, int):
            return False
        lower = source.strip().lower()
        return lower.startswith(
            ("rtsp://", "rtsps://", "http://", "https://", "tcp://", "udp://")
        )

    def _safe_read(self, cap: cv2.VideoCapture):
        try:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return True, frame
        except cv2.error:
            pass
        except Exception:
            pass
        return False, None

    def _compensate_overexposure(self, frame: np.ndarray | None) -> np.ndarray | None:
        """Auto tone-down very bright frames, then apply manual exposure bias."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return frame
        with self._lock:
            bias = float(self._manual_exposure_bias)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m = float(np.mean(gray))
        out = frame.astype(np.float32)
        if m > OVEREXPOSE_MEAN_LUMA_GATE:
            scale = min(0.98, OVEREXPOSE_TARGET_MEAN / m)
            out = out * scale
        if abs(bias) > 1e-5:
            # bias +1 → brighter, -1 → darker
            out = out * (1.0 + 0.55 * bias)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _configure_usb_exposure(self, cap: cv2.VideoCapture) -> None:
        """Best-effort lower exposure / gain on USB webcams (driver-specific)."""
        for _ in range(8):
            self._safe_read(cap)

        notes: list[str] = []
        ae = getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", None)
        if ae is not None:
            for val in (0.25, 1.0, 0.0):
                try:
                    if cap.set(ae, float(val)):
                        notes.append(f"auto_exposure={val}")
                        break
                except Exception:
                    pass

        ex = getattr(cv2, "CAP_PROP_EXPOSURE", None)
        if ex is not None:
            try:
                cap.set(ex, float(USB_EXPOSURE_RELATIVE))
                notes.append(f"exposure={USB_EXPOSURE_RELATIVE}")
            except Exception:
                pass

        gn = getattr(cv2, "CAP_PROP_GAIN", None)
        if gn is not None:
            try:
                cap.set(gn, float(USB_GAIN))
                notes.append(f"gain={USB_GAIN}")
            except Exception:
                pass

        br = getattr(cv2, "CAP_PROP_BRIGHTNESS", None)
        if br is not None:
            try:
                cap.set(br, float(USB_BRIGHTNESS))
                notes.append(f"brightness={USB_BRIGHTNESS}")
            except Exception:
                pass

        for _ in range(4):
            self._safe_read(cap)

        got_ex: float | None = None
        if ex is not None:
            try:
                got_ex = float(cap.get(ex))
            except Exception:
                got_ex = None
        log.info(
            "USB exposure tune (%s) | CAP_PROP_EXPOSURE readback=%s",
            ", ".join(notes) if notes else "no props accepted",
            got_ex,
        )

    def _open_camera(self, source: str) -> cv2.VideoCapture | None:
        """Open camera with DSHOW on Windows (fastest for USB) and negotiate max resolution."""
        parsed = self._parse_source(source)

        if isinstance(parsed, int):
            cap = self._open_usb(parsed)
        else:
            cap = self._open_network_stream(str(parsed))

        if cap is None:
            log.error("Could not open camera source: %s", source)
            return None

        self._negotiate_resolution(cap, is_stream=self._is_network_stream(parsed))
        return cap

    def _configure_ffmpeg_for(self, url: str) -> None:
        """Apply the right FFmpeg capture options for the given URL scheme.

        IMPORTANT: we **assign** the env var (not setdefault), because
        OpenCV reads it at ``VideoCapture()`` construction time and we
        may be opening a different camera type than last time.
        """
        lower = url.strip().lower()
        if lower.startswith(("rtsp://", "rtsps://")):
            opts = (
                # TCP transport survives NAT / firewalls / wifi jitter far
                # better than UDP and is the main fix for "pixelation".
                "rtsp_transport;tcp"
                # Detect a dead socket within 5s instead of blocking forever.
                "|stimeout;5000000"
                # ~1s jitter buffer — smooths bursty arrivals into clean playback.
                "|max_delay;1000000"
                # Don't reorder, we want frames as soon as they arrive.
                "|reorder_queue_size;0"
                # Larger UDP socket buffer for the rare UDP fallback.
                "|buffer_size;1048576"
                # Probe size & analyzeduration kept small to start fast.
                "|analyzeduration;1000000"
                "|probesize;500000"
            )
        elif lower.startswith(("http://", "https://")):
            # MJPEG / HLS / generic HTTP: auto-reconnect on transient drops.
            opts = (
                "reconnect;1"
                "|reconnect_streamed;1"
                "|reconnect_delay_max;5"
                "|stimeout;5000000"
            )
        else:
            opts = "stimeout;5000000"

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
        log.info("FFmpeg capture options set: %s", opts)

    def _open_network_stream(self, url: str) -> cv2.VideoCapture | None:
        """Open RTSP / HTTP(S) / other URL streams (IP cameras)."""
        log.info("Opening stream URL: %s", url)
        self._configure_ffmpeg_for(url)

        cap: cv2.VideoCapture | None = None
        try:
            if hasattr(cv2, "CAP_FFMPEG"):
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(url)
        except Exception:
            cap = None

        if not (cap and cap.isOpened()):
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            try:
                cap = cv2.VideoCapture(url)
            except Exception:
                return None

        if not (cap and cap.isOpened()):
            if cap:
                cap.release()
            return None

        # Keep the OpenCV-side queue as short as possible so even if the
        # FFmpeg backend buffers, OpenCV itself doesn't.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Best-effort open/read timeouts (supported on newer OpenCV builds).
        for prop_name, value_ms in (
            ("CAP_PROP_OPEN_TIMEOUT_MSEC", 5000),
            ("CAP_PROP_READ_TIMEOUT_MSEC", 5000),
        ):
            prop = getattr(cv2, prop_name, None)
            if prop is not None:
                try:
                    cap.set(prop, value_ms)
                except Exception:
                    pass

        return cap

    def _open_usb(self, index: int) -> cv2.VideoCapture | None:
        """Open a USB camera. On Windows use DSHOW first — it starts faster and
        gives higher resolutions more reliably than MSMF."""
        is_win = sys.platform == "win32"
        backends: list[int | None] = []

        if is_win and hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        if is_win and hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)
        backends.append(None)  # default as last resort

        for be in backends:
            label = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF"}.get(be, "DEFAULT") if be is not None else "DEFAULT"
            log.info("Trying camera %d with backend %s ...", index, label)
            try:
                cap = cv2.VideoCapture(index, be) if be is not None else cv2.VideoCapture(index)
            except Exception as exc:
                log.warning("  backend %s raised %s", label, exc)
                continue

            if not cap or not cap.isOpened():
                log.warning("  backend %s did not open", label)
                if cap:
                    cap.release()
                continue

            ok, _ = self._safe_read(cap)
            if not ok:
                log.warning("  backend %s opened but first read failed", label)
                cap.release()
                continue

            log.info("  backend %s opened successfully", label)
            return cap

        return None

    def _negotiate_resolution(
        self, cap: cv2.VideoCapture, *, is_stream: bool = False
    ) -> None:
        """Set the highest resolution for USB; for IP streams, read frames and detect size."""
        if is_stream:
            self.actual_resolution = ""
            for _ in range(20):
                ok, frame = self._safe_read(cap)
                if ok and frame is not None and frame.size > 0:
                    fh, fw = frame.shape[:2]
                    self.actual_resolution = f"{fw}x{fh}"
                    break
                time.sleep(0.05)
            if not self.actual_resolution:
                ok, frame = self._safe_read(cap)
                if ok and frame is not None and frame.size > 0:
                    fh, fw = frame.shape[:2]
                    self.actual_resolution = f"{fw}x{fh}"
            log.info("Stream resolution: %s", self.actual_resolution or "unknown")
            return

        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            log.info("Set MJPG fourcc")
        except Exception:
            log.info("MJPG fourcc not accepted — using default codec")

        for w, h in RES_CANDIDATES:
            log.info("Requesting %dx%d ...", w, h)
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            except Exception:
                continue

            # Flush a few frames so the driver actually applies the new size.
            actual_w, actual_h = 0, 0
            for _ in range(5):
                ok, frame = self._safe_read(cap)
                if ok and frame is not None:
                    actual_h, actual_w = frame.shape[:2]
            if actual_w == 0:
                continue

            log.info("  -> camera delivering %dx%d", actual_w, actual_h)

            if actual_w >= w * 0.9 and actual_h >= h * 0.9:
                self.actual_resolution = f"{actual_w}x{actual_h}"
                log.info("Resolution locked: %s", self.actual_resolution)
                self._configure_usb_exposure(cap)
                return

        # Fallback: accept whatever the camera is currently outputting.
        ok, frame = self._safe_read(cap)
        if ok and frame is not None:
            ah, aw = frame.shape[:2]
            self.actual_resolution = f"{aw}x{ah}"
            log.info("Fallback resolution: %s", self.actual_resolution)

        self._configure_usb_exposure(cap)

    # ------------------------------------------------------------------
    # Encoding / persistence helpers
    # ------------------------------------------------------------------

    def _encode_frame(self, frame) -> str:
        h, w = frame.shape[:2]
        m = max(h, w)
        to_encode = frame
        if m > STREAM_MAX_EDGE:
            scale = STREAM_MAX_EDGE / float(m)
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            to_encode = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(
            ".jpg",
            to_encode,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                STREAM_JPEG_QUALITY,
                int(cv2.IMWRITE_JPEG_OPTIMIZE),
                1,
            ],
        )
        if not ok:
            return ""
        return base64.b64encode(jpeg.tobytes()).decode("utf-8")

    def _save_reference(self, frame) -> None:
        path = self.session_manager.path_for("reference.jpg")
        if path:
            cv2.imwrite(str(path), frame)

    def _save_debug_crops(self, prefix: str, debug: dict | None) -> None:
        """Write before/after greyscale crops and diff/mask images.

        Files written (all single-channel JPEG, same folder as the session):
          <prefix>_before_gray.jpg  – reference crop
          <prefix>_after_gray.jpg   – current-frame crop
          <prefix>_diff.jpg         – absolute-difference image
          <prefix>_change_mask.jpg  – thresholded change mask
        """
        if not debug:
            return
        jpg_opts = [int(cv2.IMWRITE_JPEG_QUALITY), DISK_JPEG_QUALITY]
        for key, suffix in (
            ("before_gray", "before_gray"),
            ("after_gray", "after_gray"),
            ("diff", "diff"),
            ("change_mask", "change_mask"),
        ):
            img = debug.get(key)
            if img is None:
                continue
            path = self.session_manager.path_for(f"{prefix}_{suffix}.jpg")
            if path:
                cv2.imwrite(str(path), img, jpg_opts)

    def _save_hit_images(self, frame, annotated, tries: int) -> None:
        hit_path = self.session_manager.path_for(f"hit_{tries:03d}.jpg")
        annotated_path = self.session_manager.path_for(f"hit_{tries:03d}_annotated.jpg")
        latest_path = self.session_manager.path_for("annotated_latest.jpg")
        latest_check = self.session_manager.path_for("latest_check.jpg")
        jpg_opts = [int(cv2.IMWRITE_JPEG_QUALITY), DISK_JPEG_QUALITY]
        if hit_path:
            cv2.imwrite(str(hit_path), frame, jpg_opts)
        if annotated_path:
            cv2.imwrite(str(annotated_path), annotated, jpg_opts)
        if latest_path:
            cv2.imwrite(str(latest_path), annotated, jpg_opts)
        if latest_check:
            cv2.imwrite(str(latest_check), annotated, jpg_opts)
        self._save_debug_crops(f"hit_{tries:03d}", self.hit_detector.last_debug)

    def _save_miss_image(self, frame, annotated, tries: int) -> None:
        miss_path = self.session_manager.path_for(f"miss_{tries:03d}.jpg")
        miss_annotated = self.session_manager.path_for(
            f"miss_{tries:03d}_annotated.jpg"
        )
        latest_check = self.session_manager.path_for("latest_check.jpg")
        jpg_opts = [int(cv2.IMWRITE_JPEG_QUALITY), DISK_JPEG_QUALITY]
        if miss_path:
            cv2.imwrite(str(miss_path), frame, jpg_opts)
        if miss_annotated:
            cv2.imwrite(str(miss_annotated), annotated, jpg_opts)
        if latest_check:
            cv2.imwrite(str(latest_check), annotated, jpg_opts)
        self._save_debug_crops(
            f"miss_{self.score_engine.state.tries:03d}", self.hit_detector.last_debug
        )

    def _save_hit_data(self, detection: dict, hit_result: dict, tries: int) -> None:
        record = {
            "try": tries,
            "bbox": detection["bbox"],
            "center_px": hit_result["hit_center_px"],
            "target_center_px": hit_result["target_center_px"],
            "offset_from_center_px": hit_result["offset_px"],
            "offset_from_center_norm": hit_result["offset_norm"],
            "score": hit_result["score"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        per_hit = self.session_manager.path_for(f"hit_{tries:03d}.json")
        if per_hit:
            per_hit.write_text(json.dumps(record, indent=2), encoding="utf-8")

        summary = self.session_manager.path_for("session_hits.json")
        if summary:
            existing: list = []
            if summary.exists():
                try:
                    existing = json.loads(summary.read_text(encoding="utf-8"))
                except Exception:
                    existing = []
            existing.append(record)
            summary.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def _save_hit_data_batch(self, all_hits: list[dict], tries: int) -> None:
        """Persist JSON for all hits detected in a single Check Now press."""
        record = {
            "try": tries,
            "hit_count": len(all_hits),
            "hits": [
                {
                    "index": h["index"],
                    "bbox": h["bbox"],
                    "center_px": h["center"],
                    "score": h["score"],
                    "offset_from_center_px": h["offset_px"],
                }
                for h in all_hits
            ],
            "total_score_this_check": sum(h["score"] for h in all_hits),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        per_hit = self.session_manager.path_for(f"hit_{tries:03d}.json")
        if per_hit:
            per_hit.write_text(json.dumps(record, indent=2), encoding="utf-8")

        summary = self.session_manager.path_for("session_hits.json")
        if summary:
            existing: list = []
            if summary.exists():
                try:
                    existing = json.loads(summary.read_text(encoding="utf-8"))
                except Exception:
                    existing = []
            existing.append(record)
            summary.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Payload publishing
    # ------------------------------------------------------------------

    def _push_payload(self, frame=None, hit_detected=False, bbox=None) -> None:
        state = self.score_engine.state
        payload = {
            "_frame_bgr": frame,
            "target_type": self.target_type,
            "status": self.status,
            "hit_detected": hit_detected,
            "bbox": bbox,
            "all_hits": list(self._all_hits_detail),
            "last_score": state.last_score,
            "total_score": state.total_score,
            "tries": state.tries,
            "hits": state.hits,
            "misses": state.misses,
            "actual_resolution": self.actual_resolution,
            "motion": round(float(self.motion), 3),
            "stable": bool(self.stable),
            "hit_center": state.last_hit_center,
            "target_center": state.last_target_center,
            "offset_from_center": state.last_offset_px,
            "exposure_bias": round(float(self._manual_exposure_bias), 4),
            "manual_roi": list(self._manual_roi) if self._manual_roi else None,
        }
        with self._lock:
            self.latest_payload = payload

    def _publish_idle_feed(self) -> None:
        state = self.score_engine.state
        with self._lock:
            self.latest_payload = {
                "_frame_bgr": None,
                "target_type": "unknown",
                "status": "idle",
                "hit_detected": False,
                "bbox": None,
                "all_hits": [],
                "last_score": state.last_score,
                "total_score": state.total_score,
                "tries": state.tries,
                "hits": state.hits,
                "misses": state.misses,
                "actual_resolution": "",
                "motion": 0.0,
                "stable": False,
                "hit_center": None,
                "target_center": None,
                "offset_from_center": None,
                "exposure_bias": round(float(self._manual_exposure_bias), 4),
                "manual_roi": list(self._manual_roi) if self._manual_roi else None,
            }

    # ------------------------------------------------------------------
    # Motion analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _to_motion_gray(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = 240.0 / max(w, 1)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _update_motion(self, frame: np.ndarray) -> float:
        gray_small = self._to_motion_gray(frame)
        mask_full = self.hit_detector.last_target_mask
        if (
            mask_full is None
            or self.reference_frame is None
            or mask_full.shape[:2] != frame.shape[:2]
        ):
            if self._prev_gray_small is None or self._prev_gray_small.shape != gray_small.shape:
                self._prev_gray_small = gray_small
                return 0.0
            diff = cv2.absdiff(gray_small, self._prev_gray_small)
            score = float(np.mean(diff))
            self._prev_gray_small = gray_small
            return score

        mask_small = cv2.resize(
            mask_full,
            (gray_small.shape[1], gray_small.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        if self._prev_gray_small is None or self._prev_gray_small.shape != gray_small.shape:
            self._prev_gray_small = gray_small
            return 0.0
        diff = cv2.absdiff(gray_small, self._prev_gray_small)
        m = mask_small > 127
        n = int(np.count_nonzero(m))
        if n < 80:
            score = float(np.mean(diff))
        else:
            score = float(np.sum(diff.astype(np.float32) * m.astype(np.float32)) / n)
        self._prev_gray_small = gray_small
        return score

    # ------------------------------------------------------------------
    # Processing thread
    # ------------------------------------------------------------------

    def _pipeline_read(self):
        """Read the next frame, using the threaded reader for network streams."""
        reader = self._stream_reader  # local ref — safe under stop() races
        if reader is not None:
            return reader.read(timeout=NETWORK_READ_TIMEOUT_SEC)
        return self._safe_read(self.capture)

    def _capture_stable_reference(self, max_wait_sec: float = 8.0) -> np.ndarray | None:
        self.status = "capturing reference"
        log.info("Capturing stable reference (hold still) ...")
        accum: np.ndarray | None = None
        kept = 0
        target_frames = 10
        deadline = time.time() + max_wait_sec
        prev_small: np.ndarray | None = None

        while self.running and kept < target_frames and time.time() < deadline:
            ok, frame = self._pipeline_read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue

            frame = self._compensate_overexposure(frame)

            small = self._to_motion_gray(frame)
            if prev_small is None:
                prev_small = small
                self._push_payload(frame)
                time.sleep(0.05)
                continue

            motion = float(np.mean(cv2.absdiff(small, prev_small)))
            prev_small = small
            self.motion = motion
            self.stable = motion <= MOTION_STILL_THRESHOLD

            if self.stable:
                if accum is None:
                    accum = frame.astype(np.float64)
                else:
                    accum += frame.astype(np.float64)
                kept += 1
            else:
                accum = None
                kept = 0

            self._push_payload(frame)
            time.sleep(0.05)

        if accum is None or kept == 0:
            log.warning("Reference capture failed — scene not stable within %.1fs", max_wait_sec)
            return None
        log.info("Reference captured (%d averaged frames)", kept)
        return (accum / kept).astype(np.uint8)

    def _start_stream_reader_if_network(self, cap: cv2.VideoCapture) -> None:
        """If the current source is a network stream, wrap cap in a threaded reader.

        This is the main fix for IP-camera 'cut/stutter/pixelation': the
        reader keeps the FFmpeg backlog drained so we never process stale
        frames, and the consumer (this thread) never has to wait on I/O.
        """
        if self._stream_reader is not None:
            try:
                self._stream_reader.stop()
            except Exception:
                pass
            self._stream_reader = None

        parsed = self._parse_source(self.camera_source or "")
        if self._is_network_stream(parsed):
            log.info("Starting threaded reader for network stream")
            self._stream_reader = _NetworkFrameReader(cap)

    def _process_loop(self) -> None:
        cap: cv2.VideoCapture | None = None
        try:
            log.info("Opening camera source='%s' ...", self.camera_source)
            cap = self._open_camera(self.camera_source or "0")
            if cap is None:
                self.status = "error: camera not found"
                self.running = False
                self._push_payload()
                return

            self.capture = cap
            self._start_stream_reader_if_network(cap)
            self.status = "warming up"
            log.info("Camera opened  |  resolution: %s  |  warming up ...", self.actual_resolution)

            warmup_start = time.time()
            while self.running and (time.time() - warmup_start) < 2.0:
                ok, frame = self._pipeline_read()
                if ok and frame is not None:
                    frame = self._compensate_overexposure(frame)
                    self.motion = self._update_motion(frame)
                    self._push_payload(frame)
                time.sleep(0.04)

            if not self.running:
                return

            ref = self._capture_stable_reference()
            if ref is None:
                self.status = "error: could not capture stable reference"
                self.running = False
                self._push_payload()
                return

            self.reference_frame = ref
            self.reference_saved = True
            self._save_reference(self.reference_frame)
            self.hit_detector._get_target_mask(self.reference_frame)

            if self.target_mode in ("figure_1", "figure_2"):
                self.target_type = self.target_mode
            else:
                t, _ = self.target_detector.detect(self.reference_frame, self.target_mode)
                self.target_type = t if t != "unknown" else "figure_1"

            self.status = "monitoring"
            self._last_motion_ts = time.time()
            log.info("Monitoring started  |  target=%s  |  res=%s", self.target_type, self.actual_resolution)

            frame_counter = 0
            _consecutive_failures = 0
            _MAX_FAILURES_BEFORE_RECONNECT = 15

            while self.running:
                ok, frame = self._pipeline_read()

                # For network streams, the reader may still be alive but
                # not producing frames (camera unplugged, network blip).
                # Trip a reconnect if we've gone too long without one.
                stream_stale = (
                    self._stream_reader is not None
                    and self._stream_reader.seconds_since_last_frame()
                    > NETWORK_RECONNECT_AFTER_SEC
                )

                if not ok or frame is None or stream_stale:
                    _consecutive_failures += 1
                    self.status = "camera_read_error"
                    self._push_payload()

                    needs_reconnect = (
                        stream_stale
                        or _consecutive_failures >= _MAX_FAILURES_BEFORE_RECONNECT
                    )
                    if needs_reconnect:
                        log.warning(
                            "Camera read stalled (failures=%d, stale=%s) — reconnecting ...",
                            _consecutive_failures, stream_stale,
                        )
                        if self._stream_reader is not None:
                            try:
                                self._stream_reader.stop()
                            except Exception:
                                pass
                            self._stream_reader = None
                        try:
                            cap.release()
                        except Exception:
                            pass
                        time.sleep(NETWORK_RECONNECT_BACKOFF_SEC)
                        new_cap = self._open_camera(self.camera_source or "0")
                        if new_cap is not None:
                            cap = new_cap
                            self.capture = cap
                            self._start_stream_reader_if_network(cap)
                            _consecutive_failures = 0
                            self.status = "monitoring"
                            log.info("Camera reconnected successfully.")
                        else:
                            log.error("Reconnect failed — will retry.")
                            _consecutive_failures = 0
                    else:
                        time.sleep(0.2)
                    continue

                _consecutive_failures = 0

                frame = self._compensate_overexposure(frame)

                if self._snap_reference_requested:
                    self._snap_reference_requested = False
                    self.reference_frame = frame.copy()
                    self._save_reference(self.reference_frame)
                    self.hit_detector.clear_mask_cache()
                    self.hit_detector._get_target_mask(self.reference_frame)
                    self._sticky_boxes = None
                    self._all_hits_detail = []
                    self._prev_gray_small = None
                    log.info("Reference image updated (manual snap)")

                now = time.time()
                frame_counter += 1
                if frame_counter % 3 == 0:
                    self.motion = self._update_motion(frame)
                    if self.motion > MOTION_STILL_THRESHOLD:
                        self._last_motion_ts = now
                    still_for = now - self._last_motion_ts
                    self.stable = still_for >= STABLE_DWELL_SEC

                annotated = frame.copy()
                hit_detected = False
                manual_miss_captured: dict | None = None

                if self.reference_frame is not None:
                    if self.hit_detector.last_target_mask is None:
                        self.hit_detector._get_target_mask(self.reference_frame)
                    outline = self.hit_detector.last_target_outline
                    if outline is not None:
                        cv2.drawContours(annotated, [outline], -1, (0, 255, 0), 2)

                if self._manual_check_pending and self.reference_frame is not None:
                    self._manual_check_pending = False
                    hits = self.hit_detector.detect_hits(
                        self.reference_frame, frame, manual_roi=self._manual_roi
                    )
                    if hits:
                        hit_detected = True
                        centers = [tuple(h["center"]) for h in hits]
                        hit_results = self.score_engine.score_hits_batch(
                            centers, frame.shape, self.target_type
                        )
                        self._sticky_boxes = [list(map(int, h["bbox"])) for h in hits]
                        self._all_hits_detail = [
                            {
                                "index": i + 1,
                                "bbox": list(map(int, h["bbox"])),
                                "center": list(map(int, h["center"])),
                                "score": r["score"],
                                "offset_px": r["offset_px"],
                            }
                            for i, (h, r) in enumerate(zip(hits, hit_results))
                        ]
                        self.last_hit_ts = now
                        self._last_hit_detail = hit_results[0]
                        log.info(
                            "HITS %d  total_score=%d",
                            len(hits),
                            self.score_engine.state.total_score,
                        )
                    else:
                        self.score_engine.record_miss()
                        manual_miss_captured = {
                            "frame": frame.copy(),
                            "annotated": annotated.copy(),
                        }
                        log.info(
                            "MISS  tries=%d  misses=%d",
                            self.score_engine.state.tries,
                            self.score_engine.state.misses,
                        )

                if self._sticky_boxes:
                    for i, box in enumerate(self._sticky_boxes):
                        x, y, w, h = box
                        cx_box, cy_box = x + w // 2, y + h // 2
                        thick = 3 if i == len(self._sticky_boxes) - 1 else 2
                        color = (0, 0, 255)
                        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thick)
                        arm = max(w, h) // 2 + 6
                        cv2.line(annotated, (cx_box - arm, cy_box), (cx_box + arm, cy_box), color, 1)
                        cv2.line(annotated, (cx_box, cy_box - arm), (cx_box, cy_box + arm), color, 1)

                if hit_detected:
                    self._save_hit_images(
                        frame, annotated, self.score_engine.state.tries
                    )
                    self._save_hit_data_batch(
                        self._all_hits_detail, self.score_engine.state.tries
                    )
                elif manual_miss_captured:
                    self._save_miss_image(
                        manual_miss_captured["frame"],
                        manual_miss_captured["annotated"],
                        self.score_engine.state.tries,
                    )

                sticky_primary = self._sticky_boxes[0] if self._sticky_boxes else None
                self._push_payload(annotated, hit_detected, sticky_primary)

                # USB cameras need a small sleep so we don't busy-loop the CPU.
                # Network streams are already paced by the threaded reader
                # (we block on its queue), so adding a sleep there would
                # cause the FFmpeg buffer to fill — exactly the bug that
                # produced 'cuts and pixelation'. Don't sleep there.
                if self._stream_reader is None:
                    time.sleep(0.03)
        except Exception:
            log.exception("Unexpected error in _process_loop")
            self.status = "error: internal"
            self._push_payload()
        finally:
            if self._stream_reader is not None:
                try:
                    self._stream_reader.stop()
                except Exception:
                    pass
                self._stream_reader = None
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.capture = None
            log.info("Camera thread exited")
