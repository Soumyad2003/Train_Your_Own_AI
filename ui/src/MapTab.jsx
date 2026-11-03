import { useEffect, useMemo, useState } from "react";
import { api } from "./api";

const COLORS = ["#3b6cf6", "#e67e22", "#27ae60", "#c0392b", "#8e44ad", "#16a085", "#d35400", "#2c3e50"];
const W = 760, H = 460, PAD = 50;

function starPath(cx, cy, R = 12, r = 5) {
  let d = "";
  for (let i = 0; i < 10; i++) {
    const a = (i * Math.PI) / 5 - Math.PI / 2;
    const rad = i % 2 === 0 ? R : r;
    d += `${i === 0 ? "M" : "L"}${cx + Math.cos(a) * rad},${cy + Math.sin(a) * rad}`;
  }
  return d + "Z";
}

export default function MapTab({ version, highlight }) {
  const [data, setData] = useState({ points: [], explained: [0, 0] });
  const [hover, setHover] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .map()
      .then((d) => {
        if (!cancelled) {
          setData(d);
          setError("");
        }
      })
      .catch((e) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [version]);

  const hasQuery = highlight && highlight.x != null;
  const hitSet = useMemo(() => new Set(highlight?.hitIds ?? []), [highlight]);

  const view = useMemo(() => {
    const xs = data.points.map((p) => p.x);
    const ys = data.points.map((p) => p.y);
    if (hasQuery) {
      xs.push(highlight.x);
      ys.push(highlight.y);
    }
    if (xs.length === 0) return null;
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const fx = (x) => (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0));
    const fy = (y) => (y1 === y0 ? 0.5 : (y - y0) / (y1 - y0));
    return {
      px: (x) => PAD + fx(x) * (W - 2 * PAD),
      py: (y) => H - PAD - fy(y) * (H - 2 * PAD),
    };
  }, [data, highlight, hasQuery]);

  const docIds = [...new Set(data.points.map((p) => p.doc_id))];
  const colorOf = (docId) => COLORS[docIds.indexOf(docId) % COLORS.length];
  const legend = docIds.map((id) => ({
    id,
    title: data.points.find((p) => p.doc_id === id).title,
  }));

  return (
    <>
      {error && <div className="error">{error}</div>}
      <div className="card">
        {!view ? (
          <p className="muted">Add some documents to see the map.</p>
        ) : (
          <>
            <svg viewBox={`0 0 ${W} ${H}`} className="map">
              {[...Array(9)].map((_, i) => {
                const gx = PAD + (i / 8) * (W - 2 * PAD);
                const gy = PAD + (i / 8) * (H - 2 * PAD);
                return (
                  <g key={i} stroke="#eef0f3">
                    <line x1={gx} y1={PAD} x2={gx} y2={H - PAD} />
                    <line x1={PAD} y1={gy} x2={W - PAD} y2={gy} />
                  </g>
                );
              })}
              <text x={W / 2} y={H - 14} fontSize="12" fill="#9aa1ab" textAnchor="middle">
                PC1 {(data.explained[0] * 100).toFixed(0)}%
              </text>
              <text x={16} y={H / 2} fontSize="12" fill="#9aa1ab" textAnchor="middle"
                    transform={`rotate(-90 16 ${H / 2})`}>
                PC2 {(data.explained[1] * 100).toFixed(0)}%
              </text>

              {hasQuery &&
                data.points
                  .filter((p) => hitSet.has(p.id))
                  .map((p) => (
                    <line key={`l${p.id}`}
                      x1={view.px(highlight.x)} y1={view.py(highlight.y)}
                      x2={view.px(p.x)} y2={view.py(p.y)}
                      stroke="#6c63ff" strokeOpacity="0.5" strokeDasharray="4 4" />
                  ))}

              {data.points.map((p) => {
                const hit = hitSet.has(p.id);
                const col = colorOf(p.doc_id);
                return (
                  <g key={p.id}
                     onMouseEnter={() => setHover(p)}
                     onMouseLeave={() => setHover(null)}>
                    {hit && (
                      <circle cx={view.px(p.x)} cy={view.py(p.y)} r={17}
                              fill="none" stroke={col} strokeWidth="2" strokeOpacity="0.45" />
                    )}
                    <circle cx={view.px(p.x)} cy={view.py(p.y)} r={hit ? 10 : 7}
                            fill={col} stroke="#fff" strokeWidth="1.5" />
                  </g>
                );
              })}

              {hasQuery && (
                <g>
                  <path d={starPath(view.px(highlight.x), view.py(highlight.y))}
                        fill="#f5b800" stroke="#fff" strokeWidth="1" />
                  <text x={view.px(highlight.x) + 16} y={view.py(highlight.y) + 4}
                        fontSize="12" fill="#555">query</text>
                </g>
              )}
            </svg>

            <div className="legend">
              {legend.map((l) => (
                <span key={l.id}>
                  <i style={{ background: colorOf(l.id) }} />
                  {l.title}
                </span>
              ))}
            </div>
            <div className="muted" style={{ minHeight: 20, marginTop: 6 }}>
              {hover
                ? `${hover.title} · chunk ${hover.chunk}`
                : hasQuery
                ? `Query: “${highlight.query}” (★) with its top results highlighted`
                : "Run a search to see where a query lands. Hover a point for details."}
            </div>
          </>
        )}
      </div>
    </>
  );
}