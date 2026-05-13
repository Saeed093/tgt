import { useEffect, useState } from "react";

export default function SavedHitsGallery({ open, onClose, apiBase }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sessions, setSessions] = useState([]);

  useEffect(() => {
    if (!open) return;
    const c = new AbortController();
    setLoading(true);
    setError("");
    fetch(`${apiBase}/session/gallery`, { signal: c.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setSessions(Array.isArray(data.sessions) ? data.sessions : []);
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message || "Failed to load gallery");
      })
      .finally(() => setLoading(false));
    return () => c.abort();
  }, [open, apiBase]);

  if (!open) return null;

  const imgUrl = (sessionId, filename) =>
    `${apiBase}/session/${encodeURIComponent(sessionId)}/file/${encodeURIComponent(filename)}`;

  return (
    <div
      className="gallery-modal-overlay"
      role="presentation"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="gallery-modal-panel">
        <header className="gallery-modal-header">
          <span className="gallery-modal-title">SAVED HITS</span>
          <button type="button" className="ghost gallery-close" onClick={onClose}>
            CLOSE
          </button>
        </header>
        <div className="gallery-modal-body">
          {loading && <p className="gallery-status">Loading…</p>}
          {!loading && error && (
            <p className="gallery-status gallery-error">{error}</p>
          )}
          {!loading && !error && sessions.length === 0 && (
            <p className="gallery-status">No saved images yet. Run a session and register hits.</p>
          )}
          {!loading &&
            !error &&
            sessions.map((s) => (
              <section key={s.id} className="gallery-session-block">
                <h3 className="gallery-session-id">{s.id}</h3>
                <div className="gallery-grid">
                  {s.files.map((f) => (
                    <figure key={`${s.id}-${f}`} className="gallery-item">
                      <a
                        href={imgUrl(s.id, f)}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="gallery-thumb-link"
                      >
                        <img src={imgUrl(s.id, f)} alt={f} loading="lazy" />
                      </a>
                      <figcaption className="gallery-caption">{f}</figcaption>
                    </figure>
                  ))}
                </div>
              </section>
            ))}
        </div>
      </div>
    </div>
  );
}
