export default function ScorePanel({
  tries,
  lastScore,
  totalScore,
  hitDetected,
  stable,
  systemRunning,
}) {
  const statusText = !systemRunning
    ? "OFFLINE"
    : hitDetected
    ? "HIT"
    : stable
    ? "READY"
    : "MOTION";

  const statusCls = !systemRunning
    ? "stat-off"
    : hitDetected
    ? "stat-hit"
    : stable
    ? "stat-ready"
    : "stat-warn";

  return (
    <section className="panel score-panel">
      <div className="score-cell">
        <span className="score-label">TRIES</span>
        <span className="score-value">{String(tries).padStart(3, "0")}</span>
      </div>
      <div className="score-cell">
        <span className="score-label">LAST</span>
        <span className="score-value accent">
          {String(lastScore).padStart(2, "0")}
        </span>
      </div>
      <div className="score-cell">
        <span className="score-label">TOTAL</span>
        <span className="score-value big">
          {String(totalScore).padStart(3, "0")}
        </span>
      </div>
      <div className="score-cell">
        <span className="score-label">STATUS</span>
        <span className={`score-status ${statusCls}`}>{statusText}</span>
      </div>
    </section>
  );
}
