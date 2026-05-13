from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("hit_detector")


class HitDetector:
    """Detect new pencil/marker marks on paper by comparing to reference.

    Key design decisions:
      - Low diff threshold (12) to catch faint pencil marks.
      - Scoring rewards *intensity* of darkening over raw area, so a tiny
        sharp pencil dot beats a large soft shadow every time.
      - Skin mask (HSV + YCrCb) excludes the user's hand.
      - Otsu paper mask excludes pre-printed silhouette ink.
    """

    def __init__(
        self,
        diff_threshold: int = 12,
        min_area: int = 12,
        max_area: int = 6000,
        max_area_frac: float = 0.02,
        border_margin_px: int = 18,
        skin_overlap_max: float = 0.20,
    ) -> None:
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.max_area_frac = max_area_frac
        self.border_margin_px = border_margin_px
        self.skin_overlap_max = skin_overlap_max

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _skin_mask(frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)

        hsv_lo1 = np.array([0, 30, 60], dtype=np.uint8)
        hsv_hi1 = np.array([20, 170, 255], dtype=np.uint8)
        hsv_lo2 = np.array([170, 30, 60], dtype=np.uint8)
        hsv_hi2 = np.array([180, 170, 255], dtype=np.uint8)
        hsv_mask = cv2.bitwise_or(
            cv2.inRange(hsv, hsv_lo1, hsv_hi1),
            cv2.inRange(hsv, hsv_lo2, hsv_hi2),
        )

        ycrcb_lo = np.array([0, 133, 77], dtype=np.uint8)
        ycrcb_hi = np.array([255, 173, 127], dtype=np.uint8)
        ycrcb_mask = cv2.inRange(ycrcb, ycrcb_lo, ycrcb_hi)

        skin = cv2.bitwise_and(hsv_mask, ycrcb_mask)
        skin = cv2.medianBlur(skin, 5)
        skin = cv2.dilate(skin, np.ones((7, 7), np.uint8), iterations=2)
        return skin

    @staticmethod
    def _paper_mask(ref_blur: np.ndarray) -> np.ndarray:
        _, otsu = cv2.threshold(
            ref_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return otsu

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect_hits(
        self, reference_frame: np.ndarray, current_frame: np.ndarray
    ) -> list[dict]:
        ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        ref_b = cv2.GaussianBlur(ref_gray, (5, 5), 0)
        cur_b = cv2.GaussianBlur(cur_gray, (5, 5), 0)

        # Signed diff — positive means current is darker than reference.
        diff_raw = ref_b.astype(np.int16) - cur_b.astype(np.int16)
        diff = np.clip(diff_raw, 0, 255).astype(np.uint8)

        paper = self._paper_mask(ref_b)
        skin = self._skin_mask(current_frame)
        not_skin = cv2.bitwise_not(skin)

        # Threshold at a low value to catch faint pencil marks.
        strong = (diff >= self.diff_threshold).astype(np.uint8) * 255
        mask = cv2.bitwise_and(strong, paper)
        mask = cv2.bitwise_and(mask, not_skin)

        k3 = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)
        mask = cv2.dilate(mask, k3, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h_img, w_img = ref_gray.shape[:2]
        max_area_abs = min(self.max_area, int(self.max_area_frac * h_img * w_img))
        margin = self.border_margin_px

        candidates = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > max_area_abs:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < 3 or h < 3:
                continue

            # Reject blobs touching image borders.
            if (
                x <= margin
                or y <= margin
                or x + w >= w_img - margin
                or y + h >= h_img - margin
            ):
                continue

            aspect = w / max(h, 1)
            if aspect > 6.0 or aspect < 0.16:
                continue

            perim = cv2.arcLength(contour, True)
            circ = 4 * np.pi * area / max(perim * perim, 1e-6)
            if circ < 0.02:
                continue

            # Reject blobs overlapping skin.
            roi_skin = skin[y : y + h, x : x + w]
            if roi_skin.size > 0:
                skin_overlap = float(np.mean(roi_skin > 0))
                if skin_overlap > self.skin_overlap_max:
                    continue

            # Must be on paper in the reference.
            roi_paper = paper[y : y + h, x : x + w]
            if roi_paper.size == 0:
                continue
            paper_frac = float(np.mean(roi_paper > 0))
            if paper_frac < 0.4:
                continue

            # Reject hits whose center is in the outer 5% border of the frame.
            cx = x + w // 2
            cy = y + h // 2
            ix0, ix1 = int(0.05 * w_img), int(0.95 * w_img)
            iy0, iy1 = int(0.05 * h_img), int(0.95 * h_img)
            if not (ix0 <= cx <= ix1 and iy0 <= cy <= iy1):
                continue

            # --- Mean darkening intensity inside the contour ---
            # This is the key metric. A real pencil mark has high mean intensity
            # in the diff because the pixels are uniformly dark. A soft shadow
            # has low mean intensity because only a few pixels barely cross the
            # threshold.
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(
                contour_mask, [contour], -1, 255, -1, offset=(-x, -y)
            )
            roi_diff = diff[y : y + h, x : x + w]
            if roi_diff.size == 0:
                continue

            # Mean diff value only within the contour shape.
            contour_pixels = roi_diff[contour_mask > 0]
            if contour_pixels.size == 0:
                continue
            mean_intensity = float(np.mean(contour_pixels))

            # --- Score: intensity-first, area as tiebreaker ---
            # mean_intensity dominates. sqrt(area) prevents large diffuse blobs
            # from outscoring small sharp marks. Circularity gives a small bonus
            # to round dots (pencil tips).
            score = mean_intensity * (1.0 + circ) * np.sqrt(area) * paper_frac

            candidates.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "center": [int(cx), int(cy)],
                "area": area,
                "mean_intensity": mean_intensity,
                "score": score,
            })

        if not candidates:
            return []

        # Pick the candidate with highest score.
        best = max(candidates, key=lambda c: c["score"])

        log.info(
            "Candidates: %d  |  best: bbox=%s  intensity=%.1f  area=%.0f  score=%.1f",
            len(candidates), best["bbox"], best["mean_intensity"],
            best["area"], best["score"],
        )

        return [best]
