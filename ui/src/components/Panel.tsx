import { useEffect, useState } from "react";

import { api } from "../api";
import type { Call, Evidence, Graphs, NodeEvent, Paper, PaperDocument, Turn } from "../api";
import { phases } from "../chat";
import { Graph } from "./Graph";
import type { Tab } from "./Process";
import { Timeline } from "./Timeline";

const VIEWS: { id: keyof Graphs; label: string }[] = [
  { id: "main", label: "Conversation" },
  { id: "discover", label: "Discover" },
  { id: "read", label: "Read" },
  { id: "answer", label: "Answer" },
];

export interface PanelState {
  key: string; // a turn_id, or "pending"
  tab: Tab;
  cited: string | null; // E#
  paper: string | null; // arxiv id
}

/** The side panel of one turn: its path through the graphs, its evidence with the cited
 *  sentence marked in its passage, and its papers. */
export function Panel({
  state,
  turn,
  live,
  graphs,
  set,
  close,
}: {
  state: PanelState;
  turn: { message: string; trace: NodeEvent[]; calls: Call[]; result: Turn | null };
  live: boolean;
  graphs: Graphs | null;
  set: (s: Partial<PanelState>) => void;
  close: () => void;
}) {
  const result = turn.result;
  const evidence = result?.answer?.evidence ?? [];
  const papers = [...(result?.read ?? []), ...(result?.papers ?? [])];
  return (
    <aside className="panel-side">
      <header className="panel-side-head">
        <div>
          <div className="eyebrow">This turn</div>
          <p className="panel-side-title">{turn.message}</p>
        </div>
        <button className="icon" onClick={close} aria-label="Close panel">
          ✕
        </button>
      </header>
      <nav className="panel-tabs" role="tablist">
        {(["workflow", "evidence", "papers"] as const).map((tab) => (
          <button key={tab} role="tab" aria-selected={state.tab === tab} onClick={() => set({ tab })}>
            {tab === "workflow" ? "Workflow" : tab === "evidence" ? `Evidence · ${evidence.length}` : `Papers · ${dedupe(papers).length}`}
          </button>
        ))}
      </nav>
      <div className="panel-body">
        {state.tab === "workflow" && <Workflow trace={turn.trace} calls={turn.calls} live={live} graphs={graphs} />}
        {state.tab === "evidence" && (
          <Sources evidence={evidence} cited={state.cited} choose={(cited) => set({ cited })} />
        )}
        {state.tab === "papers" && <Papers papers={dedupe(papers)} read={result?.read ?? []} focus={state.paper} />}
      </div>
    </aside>
  );
}

function Workflow({ trace, calls, live, graphs }: { trace: NodeEvent[]; calls: Call[]; live: boolean; graphs: Graphs | null }) {
  const [view, setView] = useState<keyof Graphs>("main");
  const coloured = phases(trace);
  const running = (g: string) => live && Object.values(coloured[g] ?? {}).some((p) => p === "start");
  return (
    <>
      <div className="seg">
        {VIEWS.map((v) => (
          <button key={v.id} aria-selected={view === v.id} onClick={() => setView(v.id)}>
            {v.label}
            {running(v.id) && <span className="dot" />}
            {!running(v.id) && coloured[v.id] && v.id !== "main" && <span className="ran" />}
          </button>
        ))}
      </div>
      {graphs ? <Graph definition={graphs[view]} phases={coloured[view] ?? {}} /> : <p className="notice">The server is not reachable.</p>}
      <div className="legend">
        <span><i className="l-run" />running</span>
        <span><i className="l-done" />done</span>
        <span><i className="l-deg" />degraded</span>
        <span><i className="l-fail" />failed</span>
      </div>
      <Timeline calls={calls} />
    </>
  );
}

function Sources({ evidence, cited, choose }: { evidence: Evidence[]; cited: string | null; choose: (id: string) => void }) {
  const chosen = evidence.find((e) => e.id === cited) ?? evidence[0] ?? null;
  const [document, setDocument] = useState<PaperDocument | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!chosen) return;
    setError(null);
    api.document(chosen.paper_id).then(setDocument).catch((e: unknown) => setError(String(e)));
  }, [chosen?.paper_id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (chosen) window.document.getElementById(`chunk-${chosen.chunk_id}`)?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [chosen, document]);

  if (evidence.length === 0) return <p className="notice">This turn delivered no cited sentence.</p>;
  return (
    <>
      <div className="evidence-chips">
        {evidence.map((e) => (
          <button key={e.id} className="chip" aria-pressed={chosen?.id === e.id} onClick={() => choose(e.id)} title={e.text}>
            {e.id}
          </button>
        ))}
      </div>
      {chosen && <blockquote className="quote">{chosen.text}</blockquote>}
      {error && <div className="error-box">{error}</div>}
      {chosen && document && document.id === chosen.paper_id && (
        <article className="source">
          <h3>{document.title}</h3>
          {document.passages.map((p) => (
            <p key={p.chunk_id} id={`chunk-${p.chunk_id}`} className={`passage ${p.chunk_id === chosen.chunk_id ? "cited" : ""}`}>
              <small>{p.heading_path.split(" › ").slice(1).join(" › ") || p.heading_path}</small>
              {p.chunk_id === chosen.chunk_id ? marked(p.text, chosen.text) : p.text}
            </p>
          ))}
        </article>
      )}
    </>
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

function dedupe(papers: Paper[]): Paper[] {
  return [...new Map(papers.map((p) => [p.arxiv_id, p])).values()];
}

function Papers({ papers, read, focus }: { papers: Paper[]; read: Paper[]; focus: string | null }) {
  useEffect(() => {
    if (focus) window.document.getElementById(`paper-${focus}`)?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [focus]);
  if (papers.length === 0) return <p className="notice">No papers in this turn.</p>;
  const wasRead = new Set(read.map((p) => p.arxiv_id));
  return (
    <div className="papers">
      {papers.map((p) => (
        <article key={p.arxiv_id} id={`paper-${p.arxiv_id}`} className={`paper-card ${focus === p.arxiv_id ? "focus" : ""}`}>
          <h3>{p.title}</h3>
          <div className="meta">
            <a className="mono" href={`https://arxiv.org/abs/${p.arxiv_id}v${p.version}`} target="_blank" rel="noreferrer">
              arXiv {p.arxiv_id}v{p.version}
            </a>
            {p.published && <span>{p.published}</span>}
            {wasRead.has(p.arxiv_id) && <span className="pill sage">read</span>}
            {p.relevance !== null && <span className="pill clay">relevance {p.relevance}</span>}
            {p.named && <span className="pill">named “{p.named}”</span>}
            {p.read_before && <span className="pill">read before</span>}
            {p.violated.map((v) => (
              <span key={v} className="pill rust">may break “{v}”</span>
            ))}
          </div>
          {p.reason && <p>{p.reason}</p>}
          <details>
            <summary>Abstract</summary>
            <p>{p.abstract}</p>
          </details>
        </article>
      ))}
    </div>
  );
}
