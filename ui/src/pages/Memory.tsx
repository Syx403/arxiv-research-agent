import { useCallback, useEffect, useState } from "react";

import { api } from "../api";
import type { Memory } from "../api";

/** What ARA remembers about you, quoted from you, and the papers you have read; forget any fact. */
export function MemoryPage() {
  const [memory, setMemory] = useState<Memory | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api
      .memory()
      .then(setMemory)
      .catch((e: unknown) => setError(String(e)));
  }, []);

  useEffect(load, [load]);

  const forget = async (key: string) => {
    await api.forget(key);
    load();
  };

  return (
    <>
      <div className="eyebrow">Memory</div>
      <h1>What ARA keeps between conversations.</h1>
      <p className="lede">
        Facts are kept only in your own words, with the conversation they came from. Forgetting one
        removes it from every later search.
      </p>
      {error && <div className="error-box">{error}</div>}
      <hr className="rule" />
      <div className="memory">
        <section>
          <h2 style={{ marginBottom: "1rem" }}>About you</h2>
          {memory?.facts.length === 0 && (
            <p className="notice">Nothing yet. Tell ARA, for example, “I only use hosted APIs.”</p>
          )}
          {memory?.facts.map((f) => (
            <article key={f.key} className="panel fact">
              <blockquote>“{f.quote}”</blockquote>
              <button className="button quiet" onClick={() => void forget(f.key)}>
                Forget
              </button>
              <p>{f.statement}</p>
              <span className="meta">
                {f.day} · <span className="mono">{f.key}</span>
              </span>
            </article>
          ))}
        </section>
        <section>
          <h2 style={{ marginBottom: "1rem" }}>Library</h2>
          {memory?.library.length === 0 && <p className="notice">No papers read yet.</p>}
          <table className="data">
            <tbody>
              {memory?.library.map((p) => (
                <tr key={p.arxiv_id}>
                  <td>{p.title}</td>
                  <td className="num">
                    {p.arxiv_id}v{p.version}
                  </td>
                  <td className="num">{p.published}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </>
  );
}
