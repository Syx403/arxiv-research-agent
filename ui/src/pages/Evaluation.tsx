import { useEffect, useState } from "react";

import { api } from "../api";
import type { Metric, Run, RunSummary } from "../api";

const SUITES: Record<string, string> = {
  s1: "Retrieval",
  s2: "Reading QA",
  s3: "Discovery",
  s4: "Understand",
  s5: "Verifier",
  s6: "Multi-turn + memory",
  s7: "Robustness",
};

/** Every evaluation round: metrics with 95% bootstrap intervals by split, items, cost. */
export function EvaluationPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [chosen, setChosen] = useState<string | null>(null);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .runs()
      .then((r) => {
        setRuns(r);
        setChosen((c) => c ?? r[0]?.id ?? null);
      })
      .catch((e: unknown) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!chosen) return;
    setSummary(null);
    api
      .run(chosen)
      .then(setSummary)
      .catch((e: unknown) => setError(String(e)));
  }, [chosen]);

  const spent = runs.reduce((sum, r) => sum + r.cost_usd, 0);

  return (
    <>
      <div className="eyebrow">Evaluation</div>
      <h1>How do we know it works?</h1>
      <p className="lede">
        Small, honest suites. Each round reports held-out items separately from the dev items used
        for choices; intervals are 95% bootstrap, so read small n as directional.
      </p>
      <div className="totals" style={{ margin: "1.4rem 0 2rem" }}>
        <div className="figure">
          <b>{runs.length}</b>
          <span>rounds</span>
        </div>
        <div className="figure">
          <b>${spent.toFixed(3)}</b>
          <span>spent on evaluation</span>
        </div>
      </div>
      {error && <div className="error-box">{error}</div>}
      <div className="runs-layout">
        <table className="data">
          <thead>
            <tr>
              <th>Suite</th>
              <th>Round</th>
              <th className="num">Items</th>
              <th className="num">Cost</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr
                key={r.id}
                className="clickable"
                onClick={() => setChosen(r.id)}
                style={r.id === chosen ? { background: "var(--clay-wash)" } : undefined}
              >
                <td>
                  {SUITES[r.suite] ?? r.suite}
                  {r.status !== "complete" && <span className="pill amber"> {r.status}</span>}
                </td>
                <td className="mono">{r.id.replace(/^s\d-/, "")}</td>
                <td className="num">{r.items}</td>
                <td className="num">${r.cost_usd.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <section>{summary ? <Summary summary={summary} /> : <p className="notice">Loading…</p>}</section>
      </div>
    </>
  );
}

function Summary({ summary }: { summary: RunSummary }) {
  const { run } = summary;
  const errors = summary.items.filter((i) => i.error).length;
  return (
    <>
      <div className="meta">
        <span className="pill clay">{SUITES[run.suite] ?? run.suite}</span>
        <span className="mono">{run.id}</span>
        <span>
          {summary.items.length} results, {errors} errors
        </span>
        {run.langsmith_experiment && <span className="mono">{run.langsmith_experiment}</span>}
      </div>
      {(["test", "dev"] as const).map((split) => (
        <Split key={split} split={split} metrics={summary.metrics.filter((m) => m.split === split)} />
      ))}
      {summary.usage.length > 0 && (
        <>
          <div className="split-head">
            <h3>Efficiency</h3>
            <span className="eyebrow">requests, tokens, cost and latency per stage</span>
          </div>
          <table className="data">
            <thead>
              <tr>
                <th>Stage</th>
                <th>Model</th>
                <th className="num">Requests</th>
                <th className="num">Cached</th>
                <th className="num">Cost</th>
                <th className="num">p50 / p95</th>
              </tr>
            </thead>
            <tbody>
              {summary.usage.map((u) => (
                <tr key={`${u.stage}-${u.model}`}>
                  <td>{u.stage}</td>
                  <td className="mono">{u.model}</td>
                  <td className="num">{u.requests}</td>
                  <td className="num">
                    {u.input_tokens ? `${Math.round((100 * u.cached_tokens) / u.input_tokens)}%` : "—"}
                  </td>
                  <td className="num">${Number(u.usd).toFixed(4)}</td>
                  <td className="num">
                    {Math.round(u.p50_ms)} / {Math.round(u.p95_ms)} ms
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </>
  );
}

function Split({ split, metrics }: { split: "test" | "dev"; metrics: Metric[] }) {
  if (metrics.length === 0) return null;
  const top = Math.max(1, ...metrics.map((m) => m.high));
  return (
    <>
      <div className="split-head">
        <h3>{split === "test" ? "Held-out" : "Dev"}</h3>
        <span className="eyebrow">
          {split === "test" ? "the reported numbers" : "used for choices"}
        </span>
      </div>
      <table className="data">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Arm</th>
            <th className="num">n</th>
            <th className="num">Mean [95% CI]</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {metrics.map((m) => (
            <tr key={`${m.arm}-${m.metric}`}>
              <td>{m.metric.replaceAll("_", " ")}</td>
              <td className="mono">{m.arm}</td>
              <td className="num">{m.n}</td>
              <td className="num">
                {m.mean.toFixed(2)} [{m.low.toFixed(2)}, {m.high.toFixed(2)}]
              </td>
              <td>
                <div className="ci" title={`${m.low.toFixed(2)} – ${m.high.toFixed(2)}`}>
                  <span
                    className="band"
                    style={{
                      left: `${(100 * m.low) / top}%`,
                      width: `${Math.max(1.5, (100 * (m.high - m.low)) / top)}%`,
                    }}
                  />
                  <span className="mean" style={{ left: `${(100 * m.mean) / top}%` }} />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
