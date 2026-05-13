import { useState } from "react";

export default function ControlPanel({
  source,
  webcams,
  savedIpCameras,
  onAddIpCamera,
  onRemoveIpCamera,
  targetMode,
  onSourceChange,
  onTargetModeChange,
  onRefreshWebcams,
  onStart,
  onStop,
  onCheckNow,
  busy,
  systemRunning,
}) {
  const [ipName, setIpName] = useState("");
  const [ipUrl, setIpUrl] = useState("");

  const handleAddIp = () => {
    if (onAddIpCamera(ipName, ipUrl)) {
      setIpName("");
      setIpUrl("");
    }
  };

  const displaySource = String(source).trim();
  const inUsb = webcams.some((w) => String(w.index) === displaySource);
  const inSaved = savedIpCameras.some((c) => c.url === displaySource);
  const inList = inUsb || inSaved;
  const selectValue =
    !inList && displaySource === ""
      ? String(webcams[0]?.index ?? "0")
      : displaySource;

  return (
    <section className="panel control-panel">
      <header className="panel-header">
        <span className="panel-title">CONTROL</span>
        <span className="panel-tag">OPS</span>
      </header>

      <div className="control-body">
        <label className="field">
          <span className="field-label">DEVICE</span>
          <div className="camera-select-row">
            <select
              value={selectValue}
              onChange={(e) => onSourceChange(e.target.value)}
              disabled={systemRunning}
            >
              {!inList && displaySource !== "" && (
                <option value={displaySource}>
                  Current URL (not in list)
                </option>
              )}
              <optgroup label="Local USB / index">
                {webcams.map((cam) => (
                  <option key={`usb-${cam.index}`} value={String(cam.index)}>
                    {cam.label}
                  </option>
                ))}
              </optgroup>
              {savedIpCameras.length > 0 && (
                <optgroup label="Saved IP cameras">
                  {savedIpCameras.map((cam) => (
                    <option key={cam.id} value={cam.url}>
                      {cam.name ? `${cam.name}` : cam.url}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
            <button
              type="button"
              className="ghost"
              onClick={onRefreshWebcams}
              disabled={busy || systemRunning}
            >
              SCAN
            </button>
          </div>
        </label>

        <div className="field ip-camera-block">
          <span className="field-label">IP CAMERA LIBRARY</span>
          <p className="field-hint">
            Save RTSP or HTTP(S) stream URLs (e.g. from your range DVR or IP cam) for one-click selection.
          </p>
          <div className="ip-add-row">
            <input
              className="ip-name-input"
              value={ipName}
              onChange={(e) => setIpName(e.target.value)}
              placeholder="Label (optional)"
              disabled={systemRunning}
              maxLength={64}
            />
            <input
              className="ip-url-input"
              value={ipUrl}
              onChange={(e) => setIpUrl(e.target.value)}
              placeholder="rtsp://user:pass@192.168.1.10:554/stream"
              disabled={systemRunning}
              spellCheck={false}
            />
            <button
              type="button"
              className="ghost ip-add-btn"
              onClick={handleAddIp}
              disabled={busy || systemRunning || !ipUrl.trim()}
            >
              ADD
            </button>
          </div>
          {savedIpCameras.length > 0 && (
            <ul className="ip-camera-list">
              {savedIpCameras.map((cam) => (
                <li key={cam.id} className="ip-camera-item">
                  <div className="ip-camera-meta">
                    <span className="ip-camera-name">
                      {cam.name || "Unnamed"}
                    </span>
                    <span className="ip-camera-url" title={cam.url}>
                      {cam.url}
                    </span>
                  </div>
                  <div className="ip-camera-actions">
                    <button
                      type="button"
                      className="ghost tiny"
                      onClick={() => onSourceChange(cam.url)}
                      disabled={systemRunning}
                    >
                      USE
                    </button>
                    <button
                      type="button"
                      className="ghost tiny danger"
                      onClick={() => onRemoveIpCamera(cam.id)}
                      disabled={systemRunning}
                    >
                      DEL
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        <label className="field">
          <span className="field-label">SOURCE / URL</span>
          <p className="field-hint">
            Last USB index or stream URL is remembered in this browser.
          </p>
          <input
            value={source}
            onChange={(e) => {
              const v = e.target.value;
              onSourceChange(v.trim() === "" ? "0" : v);
            }}
            placeholder="0  or  rtsp://…  or  http://…"
            disabled={systemRunning}
            spellCheck={false}
          />
        </label>

        <label className="field">
          <span className="field-label">TARGET PROFILE</span>
          <select
            value={targetMode}
            onChange={(e) => onTargetModeChange(e.target.value)}
            disabled={systemRunning}
          >
            <option value="auto">Auto Detect</option>
            <option value="figure_1">Figure 1</option>
            <option value="figure_2">Figure 2</option>
          </select>
        </label>
      </div>

      <div className="button-row">
        {!systemRunning ? (
          <button
            className="btn btn-start"
            onClick={onStart}
            disabled={busy}
          >
            <span className="btn-glyph" />
            START
          </button>
        ) : (
          <>
            <button
              className="btn btn-check"
              onClick={onCheckNow}
              disabled={busy}
            >
              <span className="btn-glyph" />
              CHECK NOW
            </button>
            <button
              className="btn btn-stop"
              onClick={onStop}
              disabled={busy}
            >
              <span className="btn-glyph" />
              STOP
            </button>
          </>
        )}
      </div>
    </section>
  );
}
