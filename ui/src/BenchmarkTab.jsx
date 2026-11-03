import { useState } from "react";
import { api } from "./api";

const COLORS = ["#e57373", "#f0a35e", "#4fc3d9", "#7c6cf0"];

const fmtTime = (us) => (us < 1000 ? `${us.toFixed(0)} µs` : `${(us / 1000).toFixed(2)} ms`);
const fmtBuild = (ms) => (ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(1)} s`);

export default function BenchmarkTab() {
  const [n, setN] = useState(2000);
  const [dim, setDim] = useState(64);
  const [ef, setEf] = useState(50);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function run(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      setResult(await api.benchmark(n, dim, ef));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const maxUs = result ? Math.max(...result.results.map((r) => r.query_us), 1) : 1;

  return (
    <>
      <form className="card" onSubmit={run}>
        <div className="row" style={{ alignItems: "center", flexWrap: "wrap" }}>
          <label>
            Vectors{" "}
            <select value={n} onChange={(e) => setN(Number(e.target.value))}>
              {[1000, 2000, 5000].map((v) => (
                <option key={v} value={v}>{v.toLocaleString()}</option>
              ))}
            </select>
          </label>
          <label>
            Dimensions{" "}
            <select value={dim} onChange={(e) => setDim(Number(e.target.value))}>
              {[8, 64, 768].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </label>
          <label>
            HNSW ef_search{" "}
            <select value={ef} onChange={(e) => setEf(Number(e.target.value))}>
              {[10, 50, 200].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </label>
          <button className="primary" disabled={busy}>
            {busy ? "Running…" : "Run benchmark"}
          </button>
        </div>
        <p className="muted" style={{ marginBottom: 0 }}>
          Uses synthetic clustered vectors, not your documents. The first run for a given size
          builds all the indexes, which can take from a few seconds up to a minute for the largest
          setting. Later runs at the same size are fast (the changing ef only affects HNSW).
        </p>
      </form>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="card">
          <div className="muted" style={{ marginBottom: 8 }}>
            {result.n.toLocaleString()} vectors · {result.dim} dimensions · {result.queries} queries ·
            top {result.k}{result.cached ? " · indexes were cached" : " · indexes freshly built"}
          </div>
          <table className="bench">
            <thead>
              <tr>
                <th>Method</th>
                <th>Build</th>
                <th>Median query</th>
                <th>Distance calls</th>
                <th>Recall@{result.k}</th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((r, i) => (
                <tr key={r.name}>
                  <td>{r.name}</td>
                  <td>{fmtBuild(r.build_ms)}</td>
                  <td>
                    <div className="barcell">
                      <div className="track">
                        <div
                          className="bar"
                          style={{
                            width: `${Math.max((r.query_us / maxUs) * 100, 2)}%`,
                            background: COLORS[i % COLORS.length],
                          }}
                        />
                      </div>
                      <span>{fmtTime(r.query_us)}</span>
                    </div>
                  </td>
                  <td>
                    {Math.round(r.dist_calls).toLocaleString()}{" "}
                    <span className="muted">({((r.dist_calls / result.n) * 100).toFixed(1)}%)</span>
                  </td>
                  <td>{(r.recall * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted" style={{ marginBottom: 0 }}>
            Distance calls are the implementation-independent measure of work. Times include Python
            overhead, and the NumPy row is vectorised, so compare it fairly against the others.
          </p>
        </div>
      )}
    </>
  );
}