import { useEffect, useState } from "react";

function prettyTarget(value) {
  if (value === "figure_1") return "FIG-1";
  if (value === "figure_2") return "FIG-2";
  return "—";
}

function prettyStatus(s) {
  if (!s) return "OFFLINE";
  return s.toUpperCase();
}

export default function StatusBar({
  status,
  targetType,
  wsConnected,
  stable,
  motion,
  systemRunning,
}) {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const time = now.toLocaleTimeString([], { hour12: false });
  const motionDot = !systemRunning
    ? "dot off"
    : stable
    ? "dot ok"
    : "dot warn";

  return (
    <section className="status-bar">
      <span className="sb-item">
        <span className="sb-key">SYS</span>
        <span className="sb-val">{prettyStatus(status)}</span>
      </span>
      <span className="sb-item">
        <span className="sb-key">TGT</span>
        <span className="sb-val">{prettyTarget(targetType)}</span>
      </span>
      <span className="sb-item">
        <span className="sb-key">SCENE</span>
        <span className={motionDot} />
        <span className="sb-val">
          {!systemRunning ? "—" : stable ? "STABLE" : "MOTION"}
        </span>
      </span>
      <span className="sb-item">
        <span className="sb-key">MOT</span>
        <span className="sb-val mono">
          {(motion ?? 0).toFixed(2)}
        </span>
      </span>
      <span className={`sb-item ${wsConnected ? "ws-on" : "ws-off"}`}>
        <span className={wsConnected ? "dot ok" : "dot off"} />
        {wsConnected ? "LINK" : "NOLINK"}
      </span>
      <span className="sb-item sb-clock mono">{time}</span>
    </section>
  );
}
