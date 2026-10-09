import { useState } from "react";

import type { Call, NodeEvent } from "../api";
import { LABELS, doing } from "../chat";

export type Tab = "workflow" | "evidence" | "papers";

const seconds = (from: string, to: string) =>
  Math.max(0, (new Date(to).getTime() - new Date(from).getTime()) / 1000);

/** The fold under each reply: how the turn ran, in one line, opening to its steps; and the way
 *  into the side panel for this turn's workflow, evidence and papers. */
export function Process({
  trace,
  calls,
  live,
  timing,
  counts,
  open,
}: {
  trace: NodeEvent[];
  calls: Call[];
  live: boolean;
  timing?: { started_at: string; finished_at: string };
  counts?: { read: number; listed: number; evidence: number };
  open: (tab: Tab) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const cost = calls.reduce((sum, c) => sum + (c.cost_usd ?? 0), 0);
  const degraded = trace.filter((t) => t.phase === "degraded" || t.phase === "failed").length;
  const steps = ordered(trace);
  const facts = [
    counts?.read ? `read ${counts.read} paper${counts.read > 1 ? "s" : ""}` : "",
    counts?.listed ? `listed ${counts.listed}` : "",
    `${calls.length} model call${calls.length === 1 ? "" : "s"}`,
    `$${cost.toFixed(4)}`,
    timing ? `${seconds(timing.started_at, timing.finished_at).toFixed(0)} s` : "",
  ].filter(Boolean);

  return (
    <div className={`process ${live ? "live" : ""}`}>
      <div className="process-bar">
        <button className="process-toggle" onClick={() => setExpanded((e) => !e)} aria-expanded={expanded}>
          {live ? <span className="spinner" /> : <span className="caret">{expanded ? "▾" : "▸"}</span>}
          <span>{live ? `${doing(trace)}…` : facts.join(" · ")}</span>
          {degraded > 0 && <span className="pill amber">{degraded} degraded</span>}
        </button>
        <span className="process-links">
          <button onClick={() => open("workflow")}>Workflow</button>
          {!!counts?.evidence && <button onClick={() => open("evidence")}>Evidence</button>}
          {!!(counts?.read || counts?.listed) && <button onClick={() => open("papers")}>Papers</button>}
        </span>
      </div>
      {expanded && (
        <ol className="steps">
          {steps.map((s) => (
            <li key={`${s.graph}/${s.node}`} className={s.phase}>
              <span className="step-dot" />
              <span>{LABELS[s.node] ?? s.node}</span>
              <span className="mono step-node">
                {s.graph === "main" ? "" : `${s.graph} › `}
                {s.node}
              </span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

/** Nodes in the order they first ran, each with its last phase. */
function ordered(trace: NodeEvent[]): NodeEvent[] {
  const found = new Map<string, NodeEvent>();
  for (const event of trace) {
    const key = `${event.graph}/${event.node}`;
    const before = found.get(key);
    const sticky = before && (before.phase === "degraded" || before.phase === "failed");
    found.set(key, sticky && event.phase === "end" ? before : event);
  }
  return [...found.values()].filter((e) => !LABELS[e.node] || e.node !== "__start__");
}
