from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("hit_detector")


class HitDetector:
    """Detect bullet holes on a target by comparing to a reference frame.

    The detector first segments **just the target** (paper / silhouette) from
    the reference image and ignores everything outside that region.  This is
    what makes the system robust to outdoor backgrounds — trees, sky,
    ground, or anything else moving behind the target cannot generate
    candidate hits because the difference image is masked to the target
    before thresholding.

    Tuned for IP-camera range footage where bullet holes produce strong,
    localised intensity changes.  Uses bilateral filtering to suppress
    camera noise while preserving the sharp edges of a hole, and
    aggressive morphological closing to merge fragments of a single
    impact crater.
    """

    def __init__(
        self,
        diff_threshold: int = 25,
        min_area: int = 18,
        max_area: int = 12000,
        max_area_frac: float = 0.03,
        border_margin_px: int = 8,
        dark_rel_factor: float = 0.30,
        dark_abs_floor: int = 8,
    ) -> None:
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.max_area_frac = max_area_frac
        self.border_margin_px = border_margin_px
        # Bullet holes are almost always darker than the surrounding
        # surface.  These two parameters drive an adaptive "got darker"
        # detector so a hit on the *black* part of the target — where
        # the absolute intensity drop is tiny — still registers.
        self.dark_rel_factor = dark_rel_factor
        self.dark_abs_floor = dark_abs_floor

        self._mask_cache_key: int | None = None
        self._cached_target_mask: np.ndarray | None = None
        self._cached_target_outline: np.ndarray | None = None
        # (x0, y0, x1, y1) in full-frame coords — the rectangle we crop
        # before running any change-detection work.
        self._cached_target_bbox: tuple[int, int, int, int] | None = None

        # Debug images from the last detect_hits call.  Each is a
        # cropped grayscale/uint8 image saved as a numpy array.
        self._last_debug: dict[str, np.ndarray] | None = None

    def clear_mask_cache(self) -> None:
        self._mask_cache_key = None
        self._cached_target_mask = None
        self._cached_target_outline = None
        self._cached_target_bbox = None
        self._last_debug = None

    @property
    def last_debug(self) -> dict[str, np.ndarray] | None:
        """Cropped debug images from the most recent detect_hits call.

        Keys: ``before_gray``, ``after_gray``, ``diff``, ``change_mask``.
        All are single-channel uint8 numpy arrays of the same shape.
        """
        return self._last_debug

    # ------------------------------------------------------------------
    # Target segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _texture_map(gray: np.ndarray, sigma: float = 12.0) -> np.ndarray:
        """Local standard-deviation map (uint8).

        The target paper has high local contrast (scoring rings, text, dark
        bullseye) regardless of its average brightness.  The featureless sky
        has near-zero local variance.  This separates them even when their
        mean intensities are identical — the failure mode of Otsu thresholding
        on an overcast day.
        """
        f = gray.astype(np.float32)
        ksize = max(3, int(sigma * 3) | 1)
        mean = cv2.GaussianBlur(f, (ksize, ksize), sigma)
        mean_sq = cv2.GaussianBlur(f * f, (ksize, ksize), sigma)
        variance = np.maximum(mean_sq - mean * mean, 0.0)
        std = np.sqrt(variance)
        std_norm = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX)
        return std_norm.astype(np.uint8)

    @staticmethod
    def _score_blob(
        area: float,
        x: int,
        y: int,
        bw_w: int,
        bw_h: int,
        cx: float,
        cy: float,
        img_w: int,
        img_h: int,
        solidity: float,
    ) -> float:
        # Hard-reject blobs that touch the image border (sky, ground,
        # side walls all run to the edge; the target never does).
        border = 3
        if x <= border or y <= border:
            return -1.0
        if x + bw_w >= img_w - border or y + bw_h >= img_h - border:
            return -1.0
        frac = area / float(img_w * img_h)
        if frac < 0.01 or frac > 0.80:
            return -1.0
        # Reject extreme aspect ratios (tall thin trees / horizontal banners)
        ar = bw_h / max(bw_w, 1)
        aw = bw_w / max(bw_h, 1)
        if max(ar, aw) > 4.5:
            return -1.0
        if solidity < 0.30:
            return -1.0
        # Heavy centrality bias — target is almost always near the centre.
        diag = float(np.hypot(img_w, img_h))
        cdist = float(np.hypot(cx - img_w / 2.0, cy - img_h / 2.0))
        centrality = max(0.0, 1.0 - cdist / (0.38 * diag + 1e-6))
        return area * (centrality ** 2) * (0.25 + 0.75 * min(1.0, solidity))

    def _best_filled_outer_mask(self, ref_gray: np.ndarray) -> np.ndarray | None:
        """Segment the target using local texture (variance), with an
        intensity-threshold fallback.

        Texture-based segmentation: a shooting target has high local contrast
        (rings, text, bullseye) even when its mean brightness matches the sky.
        A plain sky patch has near-zero local variance.  We build a variance
        map, threshold it, morphologically close the result to fill the paper
        rectangle, then pick the blob with the best centrality / solidity.
        """
        h, w = ref_gray.shape[:2]

        def _pick_best_blob(binary: np.ndarray) -> np.ndarray | None:
            """Return a filled mask for the best-scoring component."""
            bw = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=4
            )
            bw = cv2.morphologyEx(
                bw, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=2
            )
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(
                bw, connectivity=8
            )
            best_mask: np.ndarray | None = None
            best_score = -1.0
            for i in range(1, n):
                area = float(stats[i, cv2.CC_STAT_AREA])
                bx = int(stats[i, cv2.CC_STAT_LEFT])
                by = int(stats[i, cv2.CC_STAT_TOP])
                bw_w = int(stats[i, cv2.CC_STAT_WIDTH])
                bw_h = int(stats[i, cv2.CC_STAT_HEIGHT])
                cx, cy = centroids[i]
                comp = (labels == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    continue
                cnt0 = max(contours, key=cv2.contourArea)
                hull_area = float(cv2.contourArea(cv2.convexHull(cnt0)))
                solidity = area / max(hull_area, 1.0)
                score = self._score_blob(area, bx, by, bw_w, bw_h, cx, cy, w, h, solidity)
                if score <= best_score:
                    continue
                filled = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
                best_score = score
                best_mask = filled
            return best_mask

        # ── Primary: texture / variance map ──────────────────────────────
        tex = self._texture_map(ref_gray, sigma=10.0)
        # Use a permissive threshold so all ring detail is captured.
        tex_thr = max(8, int(np.percentile(tex[tex > 0], 30)) if np.any(tex > 0) else 8)
        _, tex_bin = cv2.threshold(tex, tex_thr, 255, cv2.THRESH_BINARY)
        result = _pick_best_blob(tex_bin)
        if result is not None:
            log.info("Target segmented via texture map")
            return result

        # ── Fallback: both Otsu polarities ───────────────────────────────
        blur = cv2.GaussianBlur(ref_gray, (7, 7), 0)
        for invert in (False, True):
            flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, bw = cv2.threshold(blur, 0, 255, flag + cv2.THRESH_OTSU)
            result = _pick_best_blob(bw)
            if result is not None:
                log.info("Target segmented via Otsu (invert=%s)", invert)
                return result

        return None

    def _get_target_mask(self, reference_frame: np.ndarray) -> np.ndarray:
        key = id(reference_frame)
        if self._mask_cache_key == key and self._cached_target_mask is not None:
            return self._cached_target_mask

        ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        h, w = ref_gray.shape[:2]
        outer = self._best_filled_outer_mask(ref_gray)
        if outer is None:
            log.info("Target segmentation failed — falling back to full frame")
            full = np.full(ref_gray.shape, 255, dtype=np.uint8)
            self._mask_cache_key = key
            self._cached_target_mask = full
            self._cached_target_outline = None
            self._cached_target_bbox = (0, 0, w, h)
            return full

        contours, _ = cv2.findContours(
            outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outline = max(contours, key=cv2.contourArea) if contours else None

        # Inner ROI: strip a band at the paper edge so treeline / sky
        # never contributes to diff even when they touch the outer bbox.
        inner = cv2.erode(outer, np.ones((9, 9), np.uint8), iterations=2)
        if int(np.count_nonzero(inner)) < 500:
            inner = cv2.erode(outer, np.ones((5, 5), np.uint8), iterations=1)

        # Tight crop rectangle around the inner mask. A few pixels of
        # padding gives morphology room to work without re-introducing
        # background pixels.
        ys, xs = np.where(inner > 0)
        if ys.size and xs.size:
            x0 = max(0, int(xs.min()) - 4)
            y0 = max(0, int(ys.min()) - 4)
            x1 = min(w, int(xs.max()) + 5)
            y1 = min(h, int(ys.max()) + 5)
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        bbox = (x0, y0, x1, y1)

        self._mask_cache_key = key
        self._cached_target_mask = inner
        self._cached_target_outline = outline
        self._cached_target_bbox = bbox
        return inner

    @property
    def last_target_mask(self) -> np.ndarray | None:
        return self._cached_target_mask

    @property
    def last_target_outline(self) -> np.ndarray | None:
        return self._cached_target_outline

    @property
    def last_target_bbox(self) -> tuple[int, int, int, int] | None:
        return self._cached_target_bbox

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def detect_hits(
        self,
        reference_frame: np.ndarray,
        current_frame: np.ndarray,
        manual_roi: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        """Compare reference_frame to current_frame and return detected hits.

        Pipeline (all work inside the crop rectangle):
          1. Convert both frames to grayscale.
          2. Apply mask so only the target interior is analysed.
          3. Gaussian denoise.
          4. Absolute difference.
          5. Dual threshold: global absolute + adaptive per-pixel (dark spots).
          6. Morphological close/open/dilate to merge hole fragments.
          7. findContours → filter by size, shape, position.
          8. Return the single best candidate.

        If ``manual_roi`` is provided (normalised 0-1 coords from the UI), it
        is used directly as the crop rectangle and auto-segmentation is
        completely bypassed.
        """
        h_full, w_full = reference_frame.shape[:2]

        # ── Determine crop rectangle ──────────────────────────────────────
        if manual_roi is not None:
            nx, ny, nw, nh = manual_roi
            x0 = int(round(nx * w_full))
            y0 = int(round(ny * h_full))
            x1 = min(w_full, int(round((nx + nw) * w_full)))
            y1 = min(h_full, int(round((ny + nh) * h_full)))
            # Within the manual box use an all-255 mask (every pixel counts).
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            bw_crop, bh_crop = x1 - x0, y1 - y0
            mask_crop = np.full((bh_crop, bw_crop), 255, dtype=np.uint8)
        else:
            target_mask_full = self._get_target_mask(reference_frame)
            bbox = self._cached_target_bbox
            if bbox is None:
                bbox = (0, 0, w_full, h_full)
            x0, y0, x1, y1 = bbox
            mask_crop = target_mask_full[y0:y1, x0:x1]

        if x1 <= x0 or y1 <= y0:
            return []

        ref_gray_full = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        cur_gray_full = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        ref_crop = ref_gray_full[y0:y1, x0:x1]
        cur_crop = cur_gray_full[y0:y1, x0:x1]
        if ref_crop.size == 0 or cur_crop.size == 0:
            return []

        h_crop, w_crop = ref_crop.shape[:2]

        # ── Step 1-3: mask → denoise ──────────────────────────────────────
        ref_in = cv2.bitwise_and(ref_crop, ref_crop, mask=mask_crop)
        cur_in = cv2.bitwise_and(cur_crop, cur_crop, mask=mask_crop)

        # Save clean (pre-blur) crops for debug output.
        _debug_before = ref_crop.copy()
        _debug_after = cur_crop.copy()

        ref_b = cv2.GaussianBlur(ref_in, (3, 3), 0)
        cur_b = cv2.GaussianBlur(cur_in, (3, 3), 0)

        ref_f = ref_b.astype(np.float32)
        cur_f = cur_b.astype(np.float32)

        # ── Step 4: absolute difference ───────────────────────────────────
        diff = cv2.absdiff(ref_b, cur_b)
        diff = cv2.bitwise_and(diff, diff, mask=mask_crop)

        # ── Step 5: dual threshold ────────────────────────────────────────
        # Channel A: global threshold — catches large brightness changes.
        abs_change = (diff >= self.diff_threshold).astype(np.uint8) * 255

        # Channel B: adaptive "got darker" — catches small holes on dark
        # surfaces (black rings, silhouette body) where the absolute drop
        # is tiny but meaningful relative to the local brightness.
        got_darker = np.clip(ref_f - cur_f, 0, 255)
        adaptive_thr = np.maximum(
            float(self.dark_abs_floor),
            ref_f * float(self.dark_rel_factor),
        )
        dark_change = (got_darker >= adaptive_thr).astype(np.uint8) * 255

        combined = cv2.bitwise_or(abs_change, dark_change)
        change_mask = cv2.bitwise_and(combined, mask_crop)
        darker_mag = got_darker.astype(np.uint8)

        # ── Step 6: morphology — merge fragments, remove speckle ─────────
        k5 = np.ones((5, 5), np.uint8)
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, k5, iterations=2)
        change_mask = cv2.morphologyEx(
            change_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )
        change_mask = cv2.dilate(change_mask, k5, iterations=1)
        change_mask = cv2.bitwise_and(change_mask, mask_crop)

        # ── Step 7: contour detection and filtering ───────────────────────
        contours, _ = cv2.findContours(
            change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        max_area_abs = min(self.max_area, int(self.max_area_frac * h_full * w_full))
        margin = self.border_margin_px

        candidates = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > max_area_abs:
                continue

            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < 3 or bh < 3:
                continue

            # Back to full-frame coordinates.
            gx = bx + x0
            gy = by + y0

            # Reject contours right at the image border (never a real hole).
            if (
                gx <= margin
                or gy <= margin
                or gx + bw >= w_full - margin
                or gy + bh >= h_full - margin
            ):
                continue

            aspect = bw / max(bh, 1)
            if aspect > 6.0 or aspect < 0.16:
                continue

            perim = cv2.arcLength(contour, True)
            circ = 4 * np.pi * area / max(perim * perim, 1e-6)
            if circ < 0.02:
                continue

            cx_local = bx + bw // 2
            cy_local = by + bh // 2
            if not (0 <= cx_local < w_crop and 0 <= cy_local < h_crop):
                continue
            if mask_crop[cy_local, cx_local] == 0:
                continue

            roi_mask = mask_crop[by : by + bh, bx : bx + bw]
            if roi_mask.size == 0:
                continue
            target_frac = float(np.mean(roi_mask > 0))
            if target_frac < 0.60:
                continue

            contour_px_mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(
                contour_px_mask, [contour], -1, 255, -1, offset=(-bx, -by)
            )
            roi_diff = diff[by : by + bh, bx : bx + bw]
            roi_darker = darker_mag[by : by + bh, bx : bx + bw]
            if roi_diff.size == 0:
                continue

            px = roi_diff[contour_px_mask > 0]
            dk = roi_darker[contour_px_mask > 0]
            if px.size == 0:
                continue

            mean_intensity = float(np.mean(px))
            mean_darker = float(np.mean(dk)) if dk.size else 0.0
            signal = max(mean_intensity, mean_darker)
            score = signal * (1.0 + circ) * np.sqrt(area) * target_frac

            candidates.append({
                "bbox": [int(gx), int(gy), int(bw), int(bh)],
                "center": [int(gx + bw // 2), int(gy + bh // 2)],
                "area": area,
                "mean_intensity": mean_intensity,
                "score": score,
            })

        # Always persist debug images so callers can save them regardless
        # of whether a hit was found.
        self._last_debug = {
            "before_gray": _debug_before,
            "after_gray": _debug_after,
            "diff": diff.copy(),
            "change_mask": change_mask.copy(),
        }

        if not candidates:
            return []

        candidates.sort(key=lambda c: c["score"], reverse=True)

        log.info(
            "Candidates: %d  |  best: bbox=%s  intensity=%.1f  area=%.0f  score=%.1f",
            len(candidates), candidates[0]["bbox"], candidates[0]["mean_intensity"],
            candidates[0]["area"], candidates[0]["score"],
        )

        return candidates
