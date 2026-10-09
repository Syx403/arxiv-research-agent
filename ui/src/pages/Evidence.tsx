import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import type { Claim, Evidence, Paper, PaperDocument, Turn } from "../api";

/** The answer of the last turn: each line with the sentences it cites, the source passage of a
 *  citation with that sentence marked, the lines the verifier rejected, and the papers. */
export function EvidencePage({ result }: { result: Turn | null }) {
  const answer = result?.answer ?? null;
  const evidence = useMemo(
    () => new Map((answer?.evidence ?? []).map((e) => [e.id, e])),
    [answer],
  );
  const [cited, setCited] = useState<Evidence | null>(null);
  const [document, setDocument] = useState<PaperDocument | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setCited(answer?.evidence[0] ?? null);
  }, [answer]);

  useEffect(() => {
    if (!cited) return;
    setError(null);
    api
      .document(cited.paper_id)
      .then(setDocument)
      .catch((e: unknown) => setError(String(e)));
  }, [cited?.paper_id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (cited) window.document.getElementById(`chunk-${cited.chunk_id}`)?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [cited, document]);

  if (!result) {
    return (
      <section>
        <div className="eyebrow">Evidence</div>
        <h1>Nothing answered yet.</h1>
        <p className="lede">Ask a question on the Workflow page; its answer and sources appear here.</p>
      </section>
    );
  }

  const chips = (claim: Claim) =>
    claim.citations.map((id) => (
      <button
        key={id}
        className="chip"
        aria-pressed={cited?.id === id}
        onClick={() => setCited(evidence.get(id) ?? null)}
      >
        {id}
      </button>
    ));

  return (
    <div className="evidence">
      <section>
        <div className="eyebrow">Answer · {answer?.checked ?? 0} lines checked by the verifier</div>
        {answer ? (
          <>
            {answer.context && <p className="lede">{answer.context}</p>}
            <p className="answer-short">
              {answer.short}
              {!answer.abstained && <span className="verified">✓ verified</span>}
            </p>
            {answer.sentences.map((claim) => (
              <p key={claim.index} className="answer-line">
                {claim.text}
                {chips(claim)}
                <span className="verified">✓</span>
              </p>
            ))}
            {answer.rejected.length > 0 && (
              <>
                <hr className="rule" />
                <div className="eyebrow" style={{ marginBottom: "0.6rem" }}>
                  Rejected by the verifier, not delivered
                </div>
                {answer.rejected.map((claim) => (
                  <p key={`${claim.index}-${claim.text}`} className="rejected">
                    {claim.text}
                  </p>
                ))}
              </>
            )}
          </>
        ) : (
          <p className="lede">This turn listed papers without reading them.</p>
        )}
        {result.problems.length > 0 && (
          <p className="error-box" style={{ marginTop: "1rem" }}>
            {result.problems.join("; ")}
          </p>
        )}
        <Papers title="Papers read" papers={result.read} />
        <Papers title="Papers listed" papers={result.papers} />
      </section>

      <aside className="panel source">
        <div className="eyebrow">Source</div>
        {error && <div className="error-box">{error}</div>}
        {!cited && <p className="notice">Choose a citation to see the sentence in its paper.</p>}
        {cited && document && (
          <>
            <h2 style={{ margin: "0.3rem 0 1.2rem" }}>{document.title}</h2>
            {document.passages.map((p) => (
              <p
                key={p.chunk_id}
                id={`chunk-${p.chunk_id}`}
                className={`passage ${p.chunk_id === cited.chunk_id ? "cited" : ""}`}
              >
                <small>{p.heading_path}</small>
                {p.chunk_id === cited.chunk_id ? marked(p.text, cited.text) : p.text}
              </p>
            ))}
          </>
        )}
      </aside>
    </div>
  );
}

function marked(text: string, sentence: string) {
  const at = text.indexOf(sentence);
  if (at < 0) return text;
  return (
    <>
      {text.slice(0, at)}
      <mark>{sentence}</mark>
      {text.slice(at + sentence.length)}
    </>
  );
}

function Papers({ title, papers }: { title: string; papers: Paper[] }) {
  if (papers.length === 0) return null;
  return (
    <>
      <h3 style={{ marginTop: "2rem" }}>{title}</h3>
      <div className="papers">
        {papers.map((p) => (
          <article key={p.arxiv_id} className="panel paper-card">
            <h3>{p.title}</h3>
            <div className="meta">
              <span className="mono">
                arXiv {p.arxiv_id}v{p.version}
              </span>
              {p.published && <span>{p.published}</span>}
              {p.relevance !== null && <span className="pill clay">relevance {p.relevance}</span>}
              {p.named && <span className="pill sage">named: {p.named}</span>}
              {p.read_before && <span className="pill">read before</span>}
              {p.violated.map((v) => (
                <span key={v} className="pill rust">
                  may break “{v}”
                </span>
              ))}
            </div>
            {p.reason && <p>{p.reason}</p>}
          </article>
        ))}
      </div>
    </>
  );
}
