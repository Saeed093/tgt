import { useCallback, useEffect, useRef, useState } from "react";

const statusMessages = {
  idle: "STANDBY  //  PRESS START",
  starting: "OPENING CAMERA",
  "warming up": "AUTO-EXPOSURE WARMUP",
  "capturing reference": "LOCKING REFERENCE FRAME",
  monitoring: "ACTIVE  //  AWAITING TARGET",
  camera_read_error: "CAMERA READ ERROR",
  "error: camera not found": "CAMERA NOT FOUND",
  "error: could not capture stable reference":
    "REFERENCE FAILED  //  HOLD STILL & RESTART",
};

function targetLabel(t) {
  if (t === "figure_1") return "FIGURE 1";
  if (t === "figure_2") return "FIGURE 2";
  return "UNKNOWN";
}

function formatOffset(val) {
  if (val == null) return "—";
  return (val >= 0 ? "+" : "") + val;
}

/**
 * Given an <img> element with object-fit:contain, return the pixel rectangle
 * (relative to the element's top-left) where the actual image content renders.
 * This accounts for letterbox / pillarbox bars.
 */
function getImageContentRect(imgEl) {
  if (!imgEl) return null;
  const elW = imgEl.clientWidth;
  const elH = imgEl.clientHeight;
  const nw = imgEl.naturalWidth;
  const nh = imgEl.naturalHeight;
  if (!nw || !nh) {
    // Natural size not known yet — assume the element box = image content.
    return { left: 0, top: 0, width: elW, height: elH };
  }
  const scale = Math.min(elW / nw, elH / nh);
  const rw = nw * scale;
  const rh = nh * scale;
  const ox = (elW - rw) / 2;
  const oy = (elH - rh) / 2;
  return { left: ox, top: oy, width: rw, height: rh };
}

export default function CameraPanel({
  frame,
  status,
  actualResolution,
  motion,
  stable,
  hitDetected,
  targetType,
  offsetFromCenter,
  allHits,
  drawingMode,
  savedRoi,
  onRoiDrawn,
}) {
  const message = statusMessages[status] || status?.toUpperCase() || "";
  const motionPct = Math.min(100, Math.round((motion || 0) * 12));
  const [hitFlash, setHitFlash] = useState(false);
  const lastHitRef = useRef(false);

  // Drawing state
  const [dragStart, setDragStart] = useState(null);
  const [dragCurrent, setDragCurrent] = useState(null);

  // Tracks the pixel rect of the rendered image inside the <img> element
  // (relative to the <img> element's own top-left corner).
  const [contentRect, setContentRect] = useState(null);

  const imgRef = useRef(null);
  const stageRef = useRef(null);

  // ── Update content rect whenever frame loads or stage resizes ─────
  const updateContentRect = useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    setContentRect(getImageContentRect(img));
  }, []);

  useEffect(() => {
    updateContentRect();
  }, [frame, updateContentRect]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const ro = new ResizeObserver(updateContentRect);
    ro.observe(stage);
    return () => ro.disconnect();
  }, [updateContentRect]);

  // ── Hit flash ─────────────────────────────────────────────────────
  useEffect(() => {
    if (hitDetected && !lastHitRef.current) {
      setHitFlash(true);
      const t = setTimeout(() => setHitFlash(false), 900);
      return () => clearTimeout(t);
    }
    lastHitRef.current = hitDetected;
  }, [hitDetected]);

  // Cancel drag when drawing mode is turned off
  useEffect(() => {
    if (!drawingMode) {
      setDragStart(null);
      setDragCurrent(null);
    }
  }, [drawingMode]);

  // ── Coordinate mapping ────────────────────────────────────────────
  /**
   * Convert a mouse event's clientX/Y to normalised (0-1) coords
   * relative to the actual rendered image content, not the element box.
   */
  function clientToNorm(e) {
    const img = imgRef.current;
    if (!img) return null;
    const cr = getImageContentRect(img); // freshest values
    if (!cr || cr.width < 1 || cr.height < 1) return null;
    const elRect = img.getBoundingClientRect();
    // Mouse relative to element top-left
    const mx = e.clientX - elRect.left;
    const my = e.clientY - elRect.top;
    // Mouse relative to content top-left, normalised
    const nx = (mx - cr.left) / cr.width;
    const ny = (my - cr.top) / cr.height;
    return {
      x: Math.max(0, Math.min(1, nx)),
      y: Math.max(0, Math.min(1, ny)),
    };
  }

  function onMouseDown(e) {
    if (!drawingMode || !frame) return;
    e.preventDefault();
    const pt = clientToNorm(e);
    if (!pt) return;
    setDragStart(pt);
    setDragCurrent(pt);
  }

  function onMouseMove(e) {
    if (!drawingMode || !dragStart) return;
    e.preventDefault();
    setDragCurrent(clientToNorm(e));
  }

  function onMouseUp(e) {
    if (!drawingMode || !dragStart) return;
    e.preventDefault();
    const end = clientToNorm(e);
    if (end) {
      const x = Math.min(dragStart.x, end.x);
      const y = Math.min(dragStart.y, end.y);
      const w = Math.abs(end.x - dragStart.x);
      const h = Math.abs(end.y - dragStart.y);
      if (w > 0.02 && h > 0.02) {
        onRoiDrawn?.({ x, y, w, h });
      }
    }
    setDragStart(null);
    setDragCurrent(null);
  }

  // ── SVG overlay — positioned to exactly cover the image content ───
  let svgEl = null;
  const showSaved = savedRoi;
  const showDrag = drawingMode && dragStart && dragCurrent;
  const hasHitLabels =
    Array.isArray(allHits) && allHits.length > 0 && contentRect && imgRef.current;

  if ((showSaved || showDrag || hasHitLabels) && contentRect) {
    const img = imgRef.current;
    const nw = img?.naturalWidth || 1;
    const nh = img?.naturalHeight || 1;
    const children = [];

    if (showSaved) {
      children.push(
        <rect
          key="saved"
          x={`${(savedRoi.x * 100).toFixed(2)}%`}
          y={`${(savedRoi.y * 100).toFixed(2)}%`}
          width={`${(savedRoi.w * 100).toFixed(2)}%`}
          height={`${(savedRoi.h * 100).toFixed(2)}%`}
          className="roi-saved"
        />
      );
    }

    if (showDrag) {
      const rx = Math.min(dragStart.x, dragCurrent.x);
      const ry = Math.min(dragStart.y, dragCurrent.y);
      const rw = Math.abs(dragCurrent.x - dragStart.x);
      const rh = Math.abs(dragCurrent.y - dragStart.y);
      children.push(
        <rect
          key="drawing"
          x={`${(rx * 100).toFixed(2)}%`}
          y={`${(ry * 100).toFixed(2)}%`}
          width={`${(rw * 100).toFixed(2)}%`}
          height={`${(rh * 100).toFixed(2)}%`}
          className="roi-drawing"
        />
      );
    }

    if (hasHitLabels) {
      allHits.forEach((h) => {
        if (!Array.isArray(h.bbox) || h.bbox.length < 4) return;
        const [bx, by, bw, bh] = h.bbox;
        // Normalised coords of box top-left corner in the SVG (0-1 space = content rect)
        const nx = bx / nw;
        const ny = by / nh;
        const lx = `${(nx * 100).toFixed(3)}%`;
        const ly = `${(ny * 100).toFixed(3)}%`;
        children.push(
          <text
            key={`hit-label-${h.index}`}
            x={lx}
            y={ly}
            className="hit-label-text"
            dominantBaseline="text-before-edge"
          >
            #{h.index}
          </text>
        );
      });
    }

    svgEl = (
      <svg
        className="roi-overlay"
        style={{
          position: "absolute",
          left: contentRect.left,
          top: contentRect.top,
          width: contentRect.width,
          height: contentRect.height,
          pointerEvents: "none",
          zIndex: 2,
        }}
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
      >
        {children}
      </svg>
    );
  }

  const stateChip =
    status === "monitoring"
      ? { text: "READY  //  PRESS CHECK", cls: "chip chip-ok" }
      : { text: message, cls: "chip chip-info" };

  const hasOffset =
    Array.isArray(offsetFromCenter) && offsetFromCenter.length >= 2;

  return (
    <section className={`panel camera-panel${hitFlash ? " hit-pulse" : ""}`}>
      <header className="panel-header">
        <span className="panel-title">LIVE FEED</span>
        {drawingMode && (
          <span className="chip chip-draw">
            DRAW TARGET ZONE — CLICK &amp; DRAG
          </span>
        )}
        <span className="panel-tag">CAM-01</span>
      </header>

      <div
        ref={stageRef}
        className={`camera-stage${drawingMode ? " drawing-mode" : ""}`}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        {frame ? (
          <img
            ref={imgRef}
            src={`data:image/jpeg;base64,${frame}`}
            alt="Camera stream"
            className="camera-frame"
            draggable={false}
            onLoad={updateContentRect}
          />
        ) : (
          <div className="camera-placeholder">
            <div className="reticle">
              <span /> <span /> <span /> <span />
              <div className="reticle-dot" />
            </div>
            <div className="placeholder-text">{message}</div>
          </div>
        )}

        {svgEl}

        <div className="hud-corner tl" />
        <div className="hud-corner tr" />
        <div className="hud-corner bl" />
        <div className="hud-corner br" />

        <div className="hud-top">
          <span className={stateChip.cls}>{stateChip.text}</span>
          <span className="chip chip-info">TGT: {targetLabel(targetType)}</span>
        </div>

        <div className="hud-top-right">
          <span className="chip chip-info">RES: {actualResolution || "—"}</span>
          <div className="motion-meter" title="Scene motion">
            <span className="motion-label">MOTION</span>
            <div className="motion-bar">
              <div
                className={`motion-fill${stable ? " ok" : " hot"}`}
                style={{ width: `${motionPct}%` }}
              />
            </div>
          </div>
        </div>

        {hitFlash && <div className="hit-overlay">HIT REGISTERED</div>}

        {hasOffset && (
          <div className="hud-bottom-left">
            <span className="chip chip-offset">
              OFFSET: {formatOffset(offsetFromCenter[0])}px ,{" "}
              {formatOffset(offsetFromCenter[1])}px
            </span>
          </div>
        )}
      </div>
    </section>
  );
}
