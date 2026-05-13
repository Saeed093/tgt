import { useEffect, useRef, useState } from "react";

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

export default function CameraPanel({
  frame,
  status,
  actualResolution,
  motion,
  stable,
  hitDetected,
  targetType,
  offsetFromCenter,
}) {
  const message = statusMessages[status] || status?.toUpperCase() || "";
  const motionPct = Math.min(100, Math.round((motion || 0) * 12));
  const [hitFlash, setHitFlash] = useState(false);
  const lastHitRef = useRef(false);

  useEffect(() => {
    if (hitDetected && !lastHitRef.current) {
      setHitFlash(true);
      const t = setTimeout(() => setHitFlash(false), 900);
      return () => clearTimeout(t);
    }
    lastHitRef.current = hitDetected;
  }, [hitDetected]);

  const stateChip = status === "monitoring"
    ? { text: "READY  //  PRESS CHECK", cls: "chip chip-ok" }
    : { text: message, cls: "chip chip-info" };

  const hasOffset = Array.isArray(offsetFromCenter) && offsetFromCenter.length >= 2;

  return (
    <section className={`panel camera-panel${hitFlash ? " hit-pulse" : ""}`}>
      <header className="panel-header">
        <span className="panel-title">LIVE FEED</span>
        <span className="panel-tag">CAM-01</span>
      </header>

      <div className="camera-stage">
        {frame ? (
          <img
            src={`data:image/jpeg;base64,${frame}`}
            alt="Camera stream"
            className="camera-frame"
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

        <div className="hud-corner tl" />
        <div className="hud-corner tr" />
        <div className="hud-corner bl" />
        <div className="hud-corner br" />

        <div className="hud-top">
          <span className={stateChip.cls}>{stateChip.text}</span>
          <span className="chip chip-info">
            TGT: {targetLabel(targetType)}
          </span>
        </div>

        <div className="hud-top-right">
          <span className="chip chip-info">
            RES: {actualResolution || "—"}
          </span>
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
              OFFSET: {formatOffset(offsetFromCenter[0])}px , {formatOffset(offsetFromCenter[1])}px
            </span>
          </div>
        )}
      </div>
    </section>
  );
}
