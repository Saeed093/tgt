from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("hit_detector")


class HitDetector:
    """Detect bullet holes on a target by comparing to a reference frame.

    Tuned for IP-camera range footage where bullet holes produce strong,
    localised intensity changes.  Uses bilateral filtering to suppress
    camera noise while preserving the sharp edges of a hole, and
    aggressive morphological closing to merge fragments of a single
    impact crater.
    """

    def __init__(
        self,
        diff_threshold: int = 40,
        min_area: int = 30,
        max_area: int = 12000,
        max_area_frac: float = 0.03,
        border_margin_px: int = 30,
    ) -> None:
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.max_area_frac = max_area_frac
        self.border_margin_px = border_margin_px

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

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

        ref_b = cv2.bilateralFilter(ref_gray, 9, 75, 75)
        cur_b = cv2.bilateralFilter(cur_gray, 9, 75, 75)

        # Absolute diff — bullet holes can be lighter *or* darker than ref.
        diff = cv2.absdiff(ref_b, cur_b)

        paper = self._paper_mask(ref_b)

        strong = (diff >= self.diff_threshold).astype(np.uint8) * 255
        mask = cv2.bitwise_and(strong, paper)

        k5 = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.dilate(mask, k5, iterations=1)

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

            roi_paper = paper[y : y + h, x : x + w]
            if roi_paper.size == 0:
                continue
            paper_frac = float(np.mean(roi_paper > 0))
            if paper_frac < 0.3:
                continue

            cx = x + w // 2
            cy = y + h // 2
            ix0, ix1 = int(0.05 * w_img), int(0.95 * w_img)
            iy0, iy1 = int(0.05 * h_img), int(0.95 * h_img)
            if not (ix0 <= cx <= ix1 and iy0 <= cy <= iy1):
                continue

            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(
                contour_mask, [contour], -1, 255, -1, offset=(-x, -y)
            )
            roi_diff = diff[y : y + h, x : x + w]
            if roi_diff.size == 0:
                continue

            contour_pixels = roi_diff[contour_mask > 0]
            if contour_pixels.size == 0:
                continue
            mean_intensity = float(np.mean(contour_pixels))

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

        best = max(candidates, key=lambda c: c["score"])

        log.info(
            "Candidates: %d  |  best: bbox=%s  intensity=%.1f  area=%.0f  score=%.1f",
            len(candidates), best["bbox"], best["mean_intensity"],
            best["area"], best["score"],
        )

        return [best]
