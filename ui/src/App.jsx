import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import DocumentsTab from "./DocumentsTab";
import SearchTab from "./SearchTab";
import AskTab from "./AskTab";
import MapTab from "./MapTab";
import "./App.css";

const TABS = ["Documents", "Search", "Ask AI", "Map"];

export default function App() {
  const [tab, setTab] = useState("Documents");
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [version, setVersion] = useState(0);        // bumps whenever documents change
  const [highlight, setHighlight] = useState(null); // last search, for the map

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await api.status());
      setError("");
    } catch (e) {
      setStatus(null);
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  async function changeIndex(type) {
    try {
      await api.setIndex(type);
      await refreshStatus();
    } catch (e) {
      setError(e.message);
    }
  }

  function onDocsChanged() {
    refreshStatus();
    setHighlight(null);                 // the old query position is meaningless after a change
    setVersion((v) => v + 1);
  }

  async function onSearched(query, hits) {
    try {
      const q = await api.project(query);
      setHighlight({ query, x: q.x, y: q.y, hitIds: hits.map((h) => h.id) });
    } catch {
      setHighlight(null);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Your Own AI</h1>
        <div className="status">
          <span className={`dot ${status?.ollama ? "on" : ""}`} />
          {status ? (
            <>
              <span>{status.ollama ? "Ollama online" : "Ollama offline"}</span>
              <span>
                {status.documents} docs · {status.chunks} chunks
              </span>
              <label>
                Index{" "}
                <select value={status.index} onChange={(e) => changeIndex(e.target.value)}>
                  <option value="brute">Brute force</option>
                  <option value="kdtree">KD-Tree</option>
                  <option value="hnsw">HNSW</option>
                </select>
              </label>
            </>
          ) : (
            <span>API offline</span>
          )}
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <nav>
        {TABS.map((t) => (
          <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </nav>

      <div hidden={tab !== "Documents"}>
        <DocumentsTab onChange={onDocsChanged} />
      </div>
      <div hidden={tab !== "Search"}>
        <SearchTab onSearched={onSearched} />
      </div>
      <div hidden={tab !== "Ask AI"}>
        <AskTab />
      </div>
      <div hidden={tab !== "Map"}>
        <MapTab version={version} highlight={highlight} />
      </div>
    </div>
  );
}