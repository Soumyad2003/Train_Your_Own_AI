import { useCallback, useEffect, useState } from "react";
import { api } from "./api";

export default function DocumentsTab({ onChange }) {
  const [docs, setDocs] = useState([]);
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      setDocs(await api.listDocuments());
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function add(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.addDocument(title.trim(), text);
      setTitle("");
      setText("");
      await load();
      onChange();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(id) {
    setError("");
    try {
      await api.deleteDocument(id);
      await load();
      onChange();
    } catch (err) {
      setError(err.message);
    }
  }

  function onFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    setTitle((t) => t || file.name.replace(/\.[^.]+$/, ""));
    file.text().then(setText);
  }

  return (
    <>
      <form className="card" onSubmit={add}>
        <div className="row" style={{ marginBottom: 8 }}>
          <input
            type="text"
            placeholder="Title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
          />
          <input type="file" accept=".txt,.md" onChange={onFile} />
        </div>
        <textarea
          placeholder="Paste text here, or choose a .txt / .md file above"
          value={text}
          onChange={(e) => setText(e.target.value)}
          required
        />
        <div className="row" style={{ marginTop: 8, alignItems: "center" }}>
          <button className="primary" disabled={busy}>
            {busy ? "Embedding…" : "Add document"}
          </button>
          {busy && (
            <span className="muted">
              Each ~250-word chunk is embedded by Ollama; long documents take a while.
            </span>
          )}
        </div>
      </form>

      {error && <div className="error">{error}</div>}

      {docs.length === 0 && <p className="muted">No documents yet.</p>}
      {docs.map((d) => (
        <div className="card row" key={d.doc_id} style={{ alignItems: "center" }}>
          <div style={{ flex: 1 }}>
            <strong>{d.title}</strong>
            <div className="muted">
              {d.chunks} chunk{d.chunks === 1 ? "" : "s"}
            </div>
          </div>
          <button className="danger" onClick={() => remove(d.doc_id)}>
            Delete
          </button>
        </div>
      ))}
    </>
  );
}