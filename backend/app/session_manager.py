from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

_SESSION_DIR_RE = re.compile(r"^session_\d{8}_\d{6}$")


class SessionManager:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.current_session_dir: Path | None = None

    def cleanup_all_sessions(self) -> None:
        if self.root_dir.exists():
            shutil.rmtree(self.root_dir, ignore_errors=True)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def start_new_session(self) -> Path:
        self.cleanup_current_session()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_dir = self.root_dir / f"session_{timestamp}"
        self.current_session_dir.mkdir(parents=True, exist_ok=True)
        return self.current_session_dir

    def cleanup_current_session(self) -> None:
        if self.current_session_dir and self.current_session_dir.exists():
            shutil.rmtree(self.current_session_dir, ignore_errors=True)
        self.current_session_dir = None

    def path_for(self, filename: str) -> Path | None:
        if not self.current_session_dir:
            return None
        return self.current_session_dir / filename

    def list_gallery_sessions(self) -> list[dict]:
        """List session folders under root_dir with image files (newest first)."""
        if not self.root_dir.exists():
            return []
        out: list[dict] = []
        for p in sorted(self.root_dir.iterdir(), key=lambda x: x.name, reverse=True):
            if not p.is_dir() or not _SESSION_DIR_RE.match(p.name):
                continue
            imgs = [
                f.name
                for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg")
            ]
            if not imgs:
                continue
            imgs.sort(key=lambda n: (0 if "_annotated" in n else 1, n))
            out.append({"id": p.name, "files": imgs})
        return out
