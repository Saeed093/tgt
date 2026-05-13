from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


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
