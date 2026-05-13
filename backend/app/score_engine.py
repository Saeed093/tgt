from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass
class ScoreState:
    tries: int = 0
    hits: int = 0
    misses: int = 0
    last_score: int = 0
    total_score: int = 0
    hit_status: str = "miss"
    last_hit_center: list[int] | None = None
    last_target_center: list[int] | None = None
    last_offset_px: list[int] | None = None


class ScoreEngine:
    def __init__(self) -> None:
        self.state = ScoreState()
        self.zones = {
            "figure_1": [
                {"name": "head_center", "center": (0.5, 0.2), "radius": 0.08, "score": 10},
                {"name": "torso_center", "center": (0.5, 0.45), "radius": 0.12, "score": 10},
                {"name": "inner_ring", "center": (0.5, 0.45), "radius": 0.22, "score": 8},
                {"name": "outer_ring", "center": (0.5, 0.45), "radius": 0.32, "score": 5},
            ],
            "figure_2": [
                {"name": "head_center", "center": (0.5, 0.2), "radius": 0.08, "score": 10},
                {"name": "torso_center", "center": (0.5, 0.5), "radius": 0.1, "score": 10},
                {"name": "inner_ring", "center": (0.5, 0.5), "radius": 0.22, "score": 8},
                {"name": "outer_ring", "center": (0.5, 0.5), "radius": 0.34, "score": 5},
            ],
        }

    def reset(self) -> None:
        self.state = ScoreState()

    def get_target_center(self, target_type: str) -> tuple[float, float]:
        """Return the primary normalised centre for a target type (torso_center zone)."""
        zones = self.zones.get(target_type, [])
        for z in zones:
            if z["name"] == "torso_center":
                return z["center"]
        if zones:
            return zones[0]["center"]
        return (0.5, 0.5)

    def record_miss(self) -> None:
        """Record a Check-Now attempt where the detector found nothing new."""
        self.state.tries += 1
        self.state.misses += 1
        self.state.hit_status = "miss"

    def score_hit(
        self,
        center: tuple[int, int] | None,
        frame_shape: tuple[int, int, int],
        target_type: str,
    ) -> dict:
        """Score a hit and return detailed result including offset from target centre.

        Returns a dict with keys: score, hit_center_px, target_center_px,
        offset_px, offset_norm.
        """
        if center is None:
            return {
                "score": 0,
                "hit_center_px": None,
                "target_center_px": None,
                "offset_px": None,
                "offset_norm": None,
            }

        h, w = frame_shape[:2]
        nx, ny = center[0] / max(w, 1), center[1] / max(h, 1)

        score = 0
        for zone in self.zones.get(target_type, []):
            zx, zy = zone["center"]
            distance = hypot(nx - zx, ny - zy)
            if distance <= zone["radius"]:
                score = max(score, int(zone["score"]))

        tc_norm = self.get_target_center(target_type)
        tc_px = (int(tc_norm[0] * w), int(tc_norm[1] * h))
        offset_px = (center[0] - tc_px[0], center[1] - tc_px[1])
        offset_norm = (round(nx - tc_norm[0], 4), round(ny - tc_norm[1], 4))

        self.state.tries += 1
        self.state.hits += 1
        self.state.last_score = score
        self.state.total_score += score
        self.state.hit_status = "hit"
        self.state.last_hit_center = list(center)
        self.state.last_target_center = list(tc_px)
        self.state.last_offset_px = list(offset_px)

        return {
            "score": score,
            "hit_center_px": list(center),
            "target_center_px": list(tc_px),
            "offset_px": list(offset_px),
            "offset_norm": list(offset_norm),
        }
