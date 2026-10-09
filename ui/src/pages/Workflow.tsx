import { useEffect, useRef, useState } from "react";

import { api } from "../api";
import type { Graphs } from "../api";
import { Graph } from "../components/Graph";
import { Timeline } from "../components/Timeline";
import type { useTurn } from "../turn";

type Turn = ReturnType<typeof useTurn>;

const SUGGESTIONS = [
  "Read 2210.03629 and explain how it interleaves reasoning and actions.",
  "Compare 2305.18323 and 2312.04511: how does each avoid calling the LLM after every tool call?",
  "Find recent papers on KV-cache eviction for long-context inference.",
  "Which papers have we read about LLM agents that call tools?",
];

const VIEWS: { id: keyof Graphs; label: string }[] = [
  { id: "main", label: "Conversation" },
  { id: "discover", label: "Discover" },
  { id: "read", label: "Read" },
  { id: "answer", label: "Answer" },
];

export function WorkflowPage({ turn }: { turn: Turn }) {
  const [graphs, setGraphs] = useState<Graphs | null>(null);
  const [view, setView] = useState<keyof Graphs>("main");
  const [draft, setDraft] = useState("");
  const transcript = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.graph().then(setGraphs).catch(() => setGraphs(null));
  }, []);

  useEffect(() => {
    transcript.current?.scrollTo({ top: transcript.current.scrollHeight, behavior: "smooth" });
  }, [turn.messages.length, turn.busy]);

  const submit = () => {
    const text = draft.trim();
    if (!text || turn.busy) return;
    setDraft("");
    void (turn.waiting ? turn.resume(text) : turn.send(text));
  };

  const running = (graph: keyof Graphs) =>
    Object.values(turn.nodes[graph] ?? {}).some((phase) => phase === "start");

  return (
    <div className="workflow">
      <section className="chat">
        <div className="chat-head">
          <div>
            <div className="eyebrow">Conversation</div>
            <h2>Ask about papers</h2>
          </div>
          <button className="button quiet" onClick={turn.reset} disabled={turn.busy}>
            New conversation
          </button>
        </div>
        <div className="transcript" ref={transcript}>
          {turn.messages.length === 0 && (
            <div className="empty">
              <h2>Find, read and cite arXiv papers.</h2>
              <p>
                Every answer line is checked against the sentence it cites before you see it. Try
                one of these, or ask your own:
              </p>
              <div className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <button key={s} className="suggestion" onClick={() => void turn.send(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {turn.messages.map((m, i) => (
            <div key={i} className={`bubble ${m.role}`}>
              {m.text}
            </div>
          ))}
          {turn.busy && <div className="notice">Working… the graph on the right shows where.</div>}
          {turn.waiting && !turn.busy && (
            <div className="waiting">
              <div className="eyebrow">A question before I go on</div>
              {turn.waiting.question}
            </div>
          )}
          {turn.error && <div className="error-box">{turn.error}</div>}
        </div>
        <div className="composer">
          <textarea
            value={draft}
            placeholder={turn.waiting ? "Your answer…" : "Ask for papers, or about a paper…"}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
          />
          <button className="button" onClick={submit} disabled={turn.busy || !draft.trim()}>
            {turn.waiting ? "Answer" : "Send"}
          </button>
        </div>
      </section>

      <section>
        <div className="panel">
          <div className="panel-head">
            <div>
              <div className="eyebrow">Live workflow</div>
              <h3>Drawn from the compiled LangGraph</h3>
            </div>
            <div className="tabs" role="tablist">
              {VIEWS.map((v) => (
                <button
                  key={v.id}
                  role="tab"
                  className="tab"
                  aria-selected={view === v.id}
                  onClick={() => setView(v.id)}
                >
                  {v.label}
                  {running(v.id) && <span className="dot" />}
                </button>
              ))}
            </div>
          </div>
          {graphs ? (
            <Graph definition={graphs[view]} phases={turn.nodes[view] ?? {}} />
          ) : (
            <p className="notice" style={{ padding: "1rem 1.2rem" }}>
              The server is not reachable.
            </p>
          )}
          <div className="legend">
            <span>
              <i style={{ background: "var(--clay-wash)", borderColor: "var(--clay)" }} />
              running
            </span>
            <span>
              <i style={{ background: "var(--sage-wash)", borderColor: "var(--sage)" }} />
              done
            </span>
            <span>
              <i style={{ background: "var(--amber-wash)", borderColor: "var(--amber)" }} />
              degraded
            </span>
            <span>
              <i style={{ background: "var(--rust-wash)", borderColor: "var(--rust)" }} />
              failed
            </span>
          </div>
        </div>
        <Timeline calls={turn.calls} />
      </section>
    </div>
  );
}
