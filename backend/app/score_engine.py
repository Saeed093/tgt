from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass
class ScoreState:
    tries: int = 0
    last_score: int = 0
    total_score: int = 0
    hit_status: str = "miss"


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

    def score_hit(
        self,
        center: tuple[int, int] | None,
        frame_shape: tuple[int, int, int],
        target_type: str,
    ) -> int:
        # Only count this as a try when we actually saw a mark on the target.
        if center is None:
            return 0

        h, w = frame_shape[:2]
        nx, ny = center[0] / max(w, 1), center[1] / max(h, 1)

        score = 0
        for zone in self.zones.get(target_type, []):
            zx, zy = zone["center"]
            distance = hypot(nx - zx, ny - zy)
            if distance <= zone["radius"]:
                score = max(score, int(zone["score"]))

        self.state.tries += 1
        self.state.last_score = score
        self.state.total_score += score
        # A confirmed mark is always a "hit" for UI purposes; score may still be
        # 0 if it landed outside scoring zones.
        self.state.hit_status = "hit"
        return score
