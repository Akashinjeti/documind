import { useEffect, useState } from "react";
import { ask, getHealth, uploadFile, type AskResult, type Health } from "./api";

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AskResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const refreshHealth = () => getHealth().then(setHealth).catch(() => setHealth(null));
  useEffect(() => {
    void refreshHealth();
  }, []);

  async function onAsk(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      setResult(await ask(question));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    try {
      const r = await uploadFile(file);
      setNotice(`Ingested ${r.source}: ${r.chunks_added} chunks`);
      await refreshHealth();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      e.target.value = "";
    }
  }

  return (
    <main>
      <header>
        <h1>DocuMind</h1>
        <p className="muted">
          Multi-agent RAG: planner, retriever, answerer, critic.{" "}
          {health ? `${health.chunks} chunks indexed · ${health.mode} mode` : "API offline"}
        </p>
      </header>

      <label className="upload">
        Upload .txt / .md
        <input type="file" accept=".txt,.md" onChange={onUpload} />
      </label>
      {notice && <p className="notice">{notice}</p>}

      <form onSubmit={onAsk}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your documents"
          minLength={3}
          required
        />
        <button disabled={busy}>{busy ? "Thinking…" : "Ask"}</button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section>
          <h2>
            Answer <span className={result.grounded ? "badge ok" : "badge warn"}>
              {result.grounded ? "grounded" : "unverified"}
            </span>
          </h2>
          <p>{result.answer}</p>

          <h3>Sources</h3>
          <ol>
            {result.sources.map((s) => (
              <li key={s.id}>
                <strong>{s.source}</strong> <span className="muted">score {s.score}</span>
                <p className="snippet">{s.text}</p>
              </li>
            ))}
          </ol>

          <details>
            <summary>Agent trace ({result.attempts} retries)</summary>
            <pre>{result.trace.join("\n")}</pre>
          </details>
        </section>
      )}
    </main>
  );
}
