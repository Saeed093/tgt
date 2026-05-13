from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .target_pipeline import TargetIdentificationPipeline


class TargetDetector:
    def __init__(self, templates_dir: Path, confidence_threshold: float = 0.62) -> None:
        self.templates_dir = templates_dir
        self.confidence_threshold = confidence_threshold
        self.templates = self._load_templates()
        self.pipeline = TargetIdentificationPipeline()

    def _load_templates(self) -> dict[str, np.ndarray]:
        templates: dict[str, np.ndarray] = {}
        for key in ("figure_1", "figure_2"):
            path = self.templates_dir / f"{key}.jpg"
            if not path.exists():
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates[key] = img
        return templates

    def reload_templates(self) -> None:
        self.templates = self._load_templates()

    def detect(self, frame: np.ndarray, target_mode: str) -> tuple[str, float]:
        if target_mode in ("figure_1", "figure_2"):
            return target_mode, 1.0

        pipe = self.pipeline.identify(frame)

        if pipe.target_type != "unknown" and pipe.confidence >= 0.35:
            return pipe.target_type, float(pipe.confidence)

        if self.templates:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            best_name = "unknown"
            best_tm = 0.0
            for name, template in self.templates.items():
                resized = cv2.resize(gray, (template.shape[1], template.shape[0]))
                res = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
                score = float(np.max(res))
                if score > best_tm:
                    best_tm = score
                    best_name = name
            if best_tm >= self.confidence_threshold:
                return best_name, best_tm

        if pipe.target_type != "unknown":
            return pipe.target_type, float(pipe.confidence)

        return "unknown", float(max(pipe.confidence, 0.0))
