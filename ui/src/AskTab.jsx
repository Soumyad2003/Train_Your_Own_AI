import { useState } from "react";
import { api } from "./api";

export default function AskTab() {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    setQuestion("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", text: q }]);
    try {
      const res = await api.ask(q);
      setMessages((m) => [
        ...m,
        { role: "ai", text: res.answer, sources: res.sources },
      ]);
    } catch (err) {
      setMessages((m) => [...m, { role: "error-msg", text: err.message }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      {messages.length === 0 && (
        <p className="muted">Ask a question about your documents.</p>
      )}
      {messages.map((m, i) => (
        <div className={`msg ${m.role}`} key={i}>
          <div className="bubble">{m.text}</div>
          {m.sources?.length > 0 && (
            <div className="sources">
              {m.sources.map((s, j) => (
                <span className="chip" key={j}>
                  {s.title} · chunk {s.chunk} · {s.distance}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
      {busy && <p className="muted">Thinking… the local model can take 10–40 seconds.</p>}

      <form className="card row" onSubmit={submit}>
        <input
          type="text"
          placeholder="Ask something…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button className="primary" disabled={busy}>
          Ask
        </button>
      </form>
    </>
  );
}