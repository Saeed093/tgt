function formatOffset(val) {
  if (val == null) return "—";
  return (val >= 0 ? "+" : "") + val;
}

export default function HitsListPanel({ allHits }) {
  if (!Array.isArray(allHits) || allHits.length === 0) return null;

  const checkTotal = allHits.reduce((s, h) => s + (h.score ?? 0), 0);

  return (
    <section className="panel hits-list-panel">
      <header className="panel-header">
        <span className="panel-title">DETECTED HITS</span>
        <span className="chip chip-ok">{allHits.length} HIT{allHits.length !== 1 ? "S" : ""}</span>
        <span className="hits-check-total">CHECK SCORE: {checkTotal} PTS</span>
      </header>

      <div className="hits-list-scroll">
        {allHits.map((h) => {
          const cx = Array.isArray(h.center) ? h.center[0] : "—";
          const cy = Array.isArray(h.center) ? h.center[1] : "—";
          const ox = Array.isArray(h.offset_px) ? h.offset_px[0] : null;
          const oy = Array.isArray(h.offset_px) ? h.offset_px[1] : null;
          return (
            <div key={h.index} className="hit-card">
              <span className="hit-card-index">#{h.index}</span>
              <span className="hit-card-coord">
                ({cx}, {cy}) px
              </span>
              <span className="hit-card-offset">
                OFF: {formatOffset(ox)}, {formatOffset(oy)}
              </span>
              <span className={`hit-card-score${h.score > 0 ? " scored" : " miss-score"}`}>
                {h.score > 0 ? `${h.score} PTS` : "0 PTS"}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}
