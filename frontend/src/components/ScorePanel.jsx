function formatOffset(val) {
  if (val == null) return "—";
  return (val >= 0 ? "+" : "") + val;
}

export default function ScorePanel({
  tries,
  hits,
  misses,
  hitDetected,
  stable,
  systemRunning,
  offsetFromCenter,
  lastScore,
  totalScore,
}) {
  const statusText = !systemRunning
    ? "OFFLINE"
    : hitDetected
    ? "HIT"
    : "READY";

  const statusCls = !systemRunning
    ? "stat-off"
    : hitDetected
    ? "stat-hit"
    : "stat-ready";

  const hasOffset = Array.isArray(offsetFromCenter) && offsetFromCenter.length >= 2;

  return (
    <section className="panel score-panel">
      <div className="score-cell">
        <span className="score-label">TRIES</span>
        <span className="score-value">{String(tries).padStart(3, "0")}</span>
      </div>
      <div className="score-cell">
        <span className="score-label">HITS</span>
        <span className="score-value accent">
          {String(hits).padStart(3, "0")}
        </span>
      </div>
      <div className="score-cell">
        <span className="score-label">MISSES</span>
        <span className="score-value big">
          {String(misses).padStart(3, "0")}
        </span>
      </div>
      <div className="score-cell">
        <span className="score-label">LAST CHK</span>
        <span className="score-value accent">
          {lastScore != null ? String(lastScore).padStart(3, "0") : "—"}
        </span>
      </div>
      <div className="score-cell">
        <span className="score-label">TOTAL</span>
        <span className="score-value accent">
          {totalScore != null ? String(totalScore).padStart(4, "0") : "—"}
        </span>
      </div>
      <div className="score-cell">
        <span className="score-label">OFFSET</span>
        <span className="score-value offset-value">
          {hasOffset
            ? `${formatOffset(offsetFromCenter[0])}, ${formatOffset(offsetFromCenter[1])}`
            : "—"}
        </span>
      </div>
      <div className="score-cell">
        <span className="score-label">STATUS</span>
        <span className={`score-status ${statusCls}`}>{statusText}</span>
      </div>
    </section>
  );
}
