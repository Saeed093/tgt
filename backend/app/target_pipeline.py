"""
Geometry-based target identification for two paper target families:

- figure_2: classic circular scoring target (many concentric circles, one center).
- figure_1: human silhouette with separate head (circular rings) and torso
  (stadium / crosshair); detected via tall outer silhouette + head ring cluster,
  and weak single-center full-frame circle stack (unlike figure_2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any

import cv2
import numpy as np


@dataclass
class ConcentricCluster:
    """Circles sharing approximately the same center (pixel coords)."""

    cx: float
    cy: float
    radii: list[float]
    members: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass
class TargetPipelineResult:
    target_type: str  # "figure_1" | "figure_2" | "unknown"
    confidence: float
    figure_2_cluster: ConcentricCluster | None = None
    debug: dict[str, Any] = field(default_factory=dict)


def _dedupe_circles(
    circles: list[tuple[float, float, float]],
    d_pos: float = 6.0,
    d_r: float = 6.0,
) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for cx, cy, r in circles:
        dup = False
        for ox, oy, or_ in out:
            if hypot(cx - ox, cy - oy) < d_pos and abs(r - or_) < d_r:
                dup = True
                break
        if not dup:
            out.append((cx, cy, r))
    return out


def _downscale_gray(gray: np.ndarray, max_side: int = 520) -> tuple[np.ndarray, float]:
    """Return possibly resized gray and uniform scale to map coords back to full resolution."""
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return gray, 1.0
    s = max_side / float(m)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    return small, w / float(nw)


def _circles_from_contours(gray: np.ndarray) -> list[tuple[float, float, float]]:
    """Circle-like rings from threshold contours (fast vs HoughCircles on CPU)."""
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    acc: list[tuple[float, float, float]] = []
    max_area = 0.22 * float(h * w)
    for cnt in contours:
        a = float(cv2.contourArea(cnt))
        if a < 25.0 or a > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-3:
            continue
        circ = 4.0 * np.pi * a / (peri * peri)
        if circ < 0.45:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if r < 2.5 or r > 0.49 * min(h, w):
            continue
        acc.append((float(cx), float(cy), float(r)))
    return _dedupe_circles(acc, d_pos=5.0, d_r=5.0)


def _hough_circles_fallback(gray: np.ndarray) -> list[tuple[float, float, float]]:
    """Single Hough pass — strict param2 to avoid hundreds of spurious circles."""
    h, w = gray.shape[:2]
    diag = float(hypot(w, h))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    circles = cv2.HoughCircles(
        inv,
        cv2.HOUGH_GRADIENT,
        dp=2.0,
        minDist=max(10, int(0.025 * diag)),
        param1=80,
        param2=36,
        minRadius=4,
        maxRadius=max(12, min(h, w) // 2 - 2),
    )
    if circles is None:
        return []
    raw = [(float(cx), float(cy), float(r)) for cx, cy, r in circles[0]]
    raw = _dedupe_circles(raw, d_pos=7.0, d_r=7.0)
    if len(raw) > 64:
        raw = sorted(raw, key=lambda t: -t[2])[:64]
    return raw


def _collect_circles(gray_small: np.ndarray) -> list[tuple[float, float, float]]:
    found = _circles_from_contours(gray_small)
    hs, ws = gray_small.shape[:2]
    # Loose merge so ring stroke contours with slightly different centroids still cluster.
    merge_tol = max(12.0, 0.028 * float(hypot(ws, hs)))
    clusters = _cluster_by_center(found, center_tol=merge_tol)
    best_n = max((len(c) for c in clusters), default=0)
    if best_n < 5:
        extra = _hough_circles_fallback(gray_small)
        found = _dedupe_circles(found + extra, d_pos=6.0, d_r=6.0)
    return found


def _cluster_by_center(
    circles: list[tuple[float, float, float]],
    center_tol: float,
) -> list[list[tuple[float, float, float]]]:
    clusters: list[list[tuple[float, float, float]]] = []
    for c in circles:
        cx, cy, r = c
        placed = False
        for cl in clusters:
            mx = float(np.mean([p[0] for p in cl]))
            my = float(np.mean([p[1] for p in cl]))
            if hypot(cx - mx, cy - my) <= center_tol:
                cl.append(c)
                placed = True
                break
        if not placed:
            clusters.append([c])
    return clusters


def _to_concentric_cluster(members: list[tuple[float, float, float]]) -> ConcentricCluster:
    cx = float(np.median([m[0] for m in members]))
    cy = float(np.median([m[1] for m in members]))
    radii_sorted = sorted(m[2] for m in members)
    # Merge radii that are almost duplicates (noise)
    merged: list[float] = []
    for r in sorted(radii_sorted):
        if not merged or abs(r - merged[-1]) > max(3.0, 0.02 * r):
            merged.append(r)
    return ConcentricCluster(cx=cx, cy=cy, radii=merged, members=members)


def _ring_spacing_quality(radii: list[float]) -> float:
    """1.0 = perfectly even spacing; lower if irregular or collinear noise."""
    if len(radii) < 3:
        return 0.0
    diffs = [radii[i + 1] - radii[i] for i in range(len(radii) - 1)]
    if not diffs or min(diffs) <= 0:
        return 0.0
    m = float(np.mean(diffs))
    if m < 1e-6:
        return 0.0
    cv = float(np.std(diffs) / m)
    # tight CV -> high score
    return max(0.0, 1.0 - min(cv, 1.0))


def _figure2_score(cluster: ConcentricCluster, center_tol: float) -> float:
    members = cluster.members
    if len(members) < 6:
        return 0.0
    # Centers should agree
    spread = max(hypot(m[0] - cluster.cx, m[1] - cluster.cy) for m in members)
    if spread > center_tol * 1.5:
        return 0.0
    radii = sorted(cluster.radii)
    if len(radii) < 6:
        return 0.0
    n_score = min(1.0, (len(radii) - 5) / 5.0)  # 6->0.2, 10->1.0
    q = _ring_spacing_quality(radii)
    # Radii should span a reasonable band (not all tiny noise)
    span = radii[-1] - radii[0]
    if span < 15:
        return 0.0
    return 0.45 * n_score + 0.45 * q + 0.1 * min(1.0, span / 200.0)


def _largest_foreground_contour(gray: np.ndarray) -> tuple[np.ndarray | None, float, tuple[int, int, int, int]]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bi = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, (0, 0, 0, 0)
    best = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(best))
    x, y, w, h = cv2.boundingRect(best)
    return best, area, (int(x), int(y), int(w), int(h))


def _head_region_circles(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    frac_top: float = 0.45,
) -> list[tuple[float, float, float]]:
    H, W = gray.shape[:2]
    x, y, w, h = bbox
    y2 = min(H, y + int(h * frac_top))
    x2 = min(W, x + w)
    if y2 <= y + 4 or x2 <= x + 4:
        return []
    roi = gray[y:y2, x:x2]
    roi_s, sc = _downscale_gray(roi, max_side=400)
    found = _collect_circles(roi_s)
    return [(cx * sc, cy * sc, r * sc) for cx, cy, r in found]


def _score_head_cluster_in_upper_body(
    circles_roi: list[tuple[float, float, float]],
    bbox: tuple[int, int, int, int],
    center_tol: float,
) -> tuple[float, int]:
    """Map ROI circles to full-image coords and score best concentric stack."""
    x, y, _, _ = bbox
    global_circles = [(cx + x, cy + y, r) for cx, cy, r in circles_roi]
    clusters = _cluster_by_center(global_circles, center_tol=center_tol)
    best_n = 0
    best_spread = 1e9
    for cl in clusters:
        if len(cl) < 2:
            continue
        cc = _to_concentric_cluster(cl)
        spread = max(hypot(m[0] - cc.cx, m[1] - cc.cy) for m in cl)
        if len(cl) > best_n or (len(cl) == best_n and spread < best_spread):
            best_n = len(cl)
            best_spread = spread
    if best_n < 3:
        return 0.0, best_n
    conf = min(1.0, (best_n - 2) / 4.0) * (1.0 if best_spread < center_tol * 2 else 0.6)
    return conf, best_n


def _torso_crosshair_hint(gray: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    H, W = gray.shape[:2]
    y1 = min(H - 1, y + int(0.28 * h))
    y2 = min(H, y + int(0.82 * h))
    x1 = max(0, x)
    x2 = min(W, x + w)
    if y2 - y1 < 20 or x2 - x1 < 20:
        return 0.0
    roi = gray[y1:y2, x1:x2]
    edges = cv2.Canny(cv2.GaussianBlur(roi, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, int(0.08 * min(roi.shape))),
        minLineLength=int(0.15 * min(roi.shape)),
        maxLineGap=12,
    )
    if lines is None or len(lines) < 4:
        return 0.0
    rh, rw = roi.shape[:2]
    horiz = []
    vert = []
    for ln in lines[:, 0]:
        x_a, y_a, x_b, y_b = ln
        dx, dy = x_b - x_a, y_b - y_a
        ang = abs(np.degrees(np.arctan2(dy, dx)))
        if ang < 25 or ang > 155:
            horiz.append(ln)
        elif 65 < ang < 115:
            vert.append(ln)
    if len(horiz) < 1 or len(vert) < 1:
        return 0.0
    # Loose hint: enough orthogonal structure
    return min(1.0, 0.25 + 0.1 * min(len(horiz), 8) + 0.1 * min(len(vert), 8))


def _silhouette_score(
    gray: np.ndarray,
    cnt: np.ndarray,
    area: float,
    bbox: tuple[int, int, int, int],
) -> float:
    H, W = gray.shape[:2]
    frac = area / float(H * W)
    if frac < 0.06 or frac > 0.93:
        return 0.0
    _, _, w, h = bbox
    ar = h / max(w, 1)
    s = 0.0
    if ar >= 1.08:
        s += 0.35
    elif ar >= 0.95:
        s += 0.15
    if 0.12 <= frac <= 0.55:
        s += 0.25
    peri = cv2.arcLength(cnt, True)
    if peri > 1e-6:
        circ = 4.0 * np.pi * area / (peri * peri)
        # Silhouette is not a circle; moderate complexity
        if circ < 0.35:
            s += 0.25
    return min(1.0, s)


class TargetIdentificationPipeline:
    """
    Identify figure_1 vs figure_2 from a BGR frame.

    Auto mode uses geometry only (no OCR). Optional template matching can be
    layered in TargetDetector.
    """

    def __init__(
        self,
        min_figure2_rings: int = 6,
        figure2_confidence_floor: float = 0.42,
        figure1_confidence_floor: float = 0.36,
    ) -> None:
        self.min_figure2_rings = min_figure2_rings
        self.figure2_confidence_floor = figure2_confidence_floor
        self.figure1_confidence_floor = figure1_confidence_floor

    def identify(self, frame_bgr: np.ndarray) -> TargetPipelineResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return TargetPipelineResult("unknown", 0.0, debug={"reason": "empty_frame"})

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]
        diag = float(hypot(W, H))
        center_tol = max(10.0, 0.018 * diag)

        small_gray, sc = _downscale_gray(gray, max_side=520)
        circles = [(cx * sc, cy * sc, r * sc) for cx, cy, r in _collect_circles(small_gray)]
        clusters_raw = _cluster_by_center(circles, center_tol=center_tol)
        clusters = [_to_concentric_cluster(cl) for cl in clusters_raw if len(cl) >= 3]

        best_f2: ConcentricCluster | None = None
        best_f2_score = 0.0
        for cc in clusters:
            sc = _figure2_score(cc, center_tol=center_tol)
            if len(cc.radii) >= self.min_figure2_rings and sc > best_f2_score:
                best_f2_score = sc
                best_f2 = cc

        cnt, area, bbox = _largest_foreground_contour(gray)
        sil_score = _silhouette_score(gray, cnt, area, bbox) if cnt is not None else 0.0

        head_circles = _head_region_circles(gray, bbox) if cnt is not None else []
        head_conf, head_n = (
            _score_head_cluster_in_upper_body(head_circles, bbox, center_tol=center_tol)
            if cnt is not None
            else (0.0, 0)
        )
        cross = _torso_crosshair_hint(gray, bbox) if cnt is not None else 0.0

        # figure_1: silhouette + head rings; crosshair adds confidence
        f1_score = sil_score * 0.45 + head_conf * 0.42 + cross * 0.13
        if head_n >= 4:
            f1_score = min(1.0, f1_score + 0.08)

        debug: dict[str, Any] = {
            "n_circle_candidates": len(circles),
            "silhouette_score": sil_score,
            "head_ring_score": head_conf,
            "head_ring_count": head_n,
            "crosshair_hint": cross,
            "figure2_geometry_score": best_f2_score,
            "figure2_ring_count": len(best_f2.radii) if best_f2 else 0,
        }

        # Prefer figure_2 when a strong full-frame concentric stack wins
        if best_f2 is not None and best_f2_score >= self.figure2_confidence_floor:
            conf = min(1.0, best_f2_score)
            return TargetPipelineResult(
                "figure_2",
                conf,
                figure_2_cluster=best_f2,
                debug=debug,
            )

        # Avoid calling a small head stack figure_2 when body silhouette is present
        if (
            best_f2 is not None
            and len(best_f2.radii) >= self.min_figure2_rings
            and sil_score >= 0.35
            and head_n >= 3
            and best_f2_score < self.figure2_confidence_floor + 0.12
        ):
            if f1_score >= self.figure1_confidence_floor:
                return TargetPipelineResult("figure_1", min(1.0, f1_score), debug=debug)

        if f1_score >= self.figure1_confidence_floor and sil_score >= 0.22:
            return TargetPipelineResult("figure_1", min(1.0, f1_score), debug=debug)

        if best_f2 is not None and len(best_f2.radii) >= self.min_figure2_rings:
            return TargetPipelineResult(
                "figure_2",
                min(1.0, best_f2_score),
                figure_2_cluster=best_f2,
                debug=debug,
            )

        return TargetPipelineResult("unknown", max(best_f2_score, f1_score), debug=debug)


def annotate_debug(frame_bgr: np.ndarray, result: TargetPipelineResult) -> np.ndarray:
    """Draw pipeline hints for demos / tuning."""
    out = frame_bgr.copy()
    cc = result.figure_2_cluster
    if cc is not None:
        for r in cc.radii:
            cv2.circle(out, (int(cc.cx), int(cc.cy)), int(r), (0, 165, 255), 1)
        cv2.circle(out, (int(cc.cx), int(cc.cy)), 4, (0, 0, 255), -1)
    label = f"{result.target_type} ({result.confidence:.2f})"
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 220, 40), 2)
    return out
