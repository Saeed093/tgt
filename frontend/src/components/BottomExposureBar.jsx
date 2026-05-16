import { useEffect, useRef, useState } from "react";

export default function BottomExposureBar({
  apiBase,
  systemRunning,
  canCaptureReference,
  exposureBiasFromFeed,
  drawingMode,
  savedRoi,
  onSetZone,
  onClearZone,
}) {
  const [sliderPct, setSliderPct] = useState(0);
  const [pending, setPending] = useState(false);
  const [refBusy, setRefBusy] = useState(false);
  const sliderRef = useRef(null);
  const debounceRef = useRef(null);

  useEffect(() => {
    if (sliderRef.current === document.activeElement) return;
    const b = Number(exposureBiasFromFeed);
    if (!Number.isFinite(b)) return;
    setSliderPct(Math.round(b * 100));
  }, [exposureBiasFromFeed]);

  const postExposure = (bias) => {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      setPending(true);
      try {
        const res = await fetch(`${apiBase}/exposure`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ bias }),
        });
        const data = await res.json().catch(() => ({}));
        if (data.status !== "ok") {
          console.warn(data.message || "exposure update failed");
        }
      } catch (e) {
        console.warn(e);
      } finally {
        setPending(false);
      }
    }, 140);
  };

  const onSliderChange = (e) => {
    const v = Number(e.target.value);
    setSliderPct(v);
    postExposure(v / 100);
  };

  const captureReference = async () => {
    setRefBusy(true);
    try {
      const res = await fetch(`${apiBase}/capture_reference`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      const data = await res.json().catch(() => ({}));
      if (data.status !== "ok") {
        alert(data.message || "Could not capture reference");
      }
    } catch (err) {
      alert(err.message || "Network error");
    } finally {
      setRefBusy(false);
    }
  };

  const label = sliderPct === 0 ? "0" : sliderPct > 0 ? `+${sliderPct}` : String(sliderPct);

  return (
    <footer className="bottom-exposure-bar" aria-label="Live camera tuning">
      <div className="bottom-exposure-group">
        <span className="bottom-exposure-title">EXPOSURE</span>
        <span className="bottom-exposure-hint">← darker · brighter →</span>
        <input
          ref={sliderRef}
          type="range"
          className="bottom-exposure-slider"
          min={-100}
          max={100}
          step={1}
          value={sliderPct}
          onChange={onSliderChange}
          disabled={!systemRunning}
          aria-valuemin={-100}
          aria-valuemax={100}
          aria-valuenow={sliderPct}
        />
        <span className={`bottom-exposure-value ${pending ? "is-pending" : ""}`}>
          {label}
        </span>
      </div>
      <div className="bottom-zone-group">
        <button
          type="button"
          className={`btn bottom-ref-btn${drawingMode ? " is-active" : ""}`}
          onClick={onSetZone}
          title="Click then drag a rectangle on the live feed to set the target zone"
        >
          {drawingMode ? "CANCEL DRAW" : savedRoi ? "REDRAW ZONE" : "SET ZONE"}
        </button>
        {savedRoi && !drawingMode && (
          <button
            type="button"
            className="btn bottom-ref-btn danger-btn"
            onClick={onClearZone}
            title="Remove the manual zone and revert to auto-detection"
          >
            CLEAR ZONE
          </button>
        )}
      </div>

      <button
        type="button"
        className="btn bottom-ref-btn"
        onClick={captureReference}
        disabled={!canCaptureReference || refBusy}
        title="Replace the clean reference with the current frame (for lighting or target moves)"
      >
        NEW REFERENCE
      </button>
    </footer>
  );
}
