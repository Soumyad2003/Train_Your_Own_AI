import { useState } from "react";
import { api } from "./api";

const CUTOFF = 0.4;

export default function SearchTab({ onSearched }) {
  const [query, setQuery] = useState("");
  const [k, setK] = useState(3);
  const [hits, setHits] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    setError("");
    try {
      const result = await api.search(query.trim(), k);
      setHits(result);
      onSearched?.(query.trim(), result);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <form className="card row" onSubmit={submit}>
        <input
          type="text"
          placeholder="Search by meaning…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <select value={k} onChange={(e) => setK(Number(e.target.value))}>
          {[1, 3, 5, 10].map((n) => (
            <option key={n} value={n}>
              top {n}
            </option>
          ))}
        </select>
        <button className="primary" disabled={busy}>
          {busy ? "…" : "Search"}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {hits?.length === 0 && <p className="muted">No documents to search yet.</p>}
      {hits?.map((h, i) => (
        <div className="card" key={i}>
          <div className="hit-head">
            <strong>
              {h.title} <span className="muted">· chunk {h.chunk}</span>
            </strong>
            <span className={`tag ${h.distance <= CUTOFF ? "" : "far"}`}>
              distance {h.distance.toFixed(3)}
              {h.distance <= CUTOFF ? " · relevant" : " · far"}
            </span>
          </div>
          <div className="muted">
            {h.text.length > 300 ? h.text.slice(0, 300) + "…" : h.text}
          </div>
        </div>
      ))}
    </>
  );
}