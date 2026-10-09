import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import type { Catalog, Metric, Run, RunSummary, SuiteInfo } from "../api";

const SPLITS = [
  { id: "test", name: "Held-out", note: "reported" },
  { id: "dev", name: "Dev", note: "used for choices" },
] as const;

const when = (iso: string) =>
  new Date(iso).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

/** How the agent is evaluated: the shared protocol, one card per suite (what it tests and its
 *  latest held-out result), then a suite's rounds, metrics with their meaning, and items (D38). */
export function EvaluationPage() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [latest, setLatest] = useState<Record<string, RunSummary>>({});
  const [suite, setSuite] = useState<string>("s6");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.suites(), api.runs()])
      .then(([c, r]) => {
        setCatalog(c);
        setRuns(r);
        const newest = c.suites
          .map((s) => r.find((x) => x.suite === s.id && x.status === "complete"))
          .filter((x): x is Run => !!x);
        return Promise.all(newest.map((x) => api.run(x.id)));
      })
      .then((summaries) => setLatest(Object.fromEntries(summaries.map((s) => [s.run.suite, s]))))
      .catch((e: unknown) => setError(String(e)));
  }, []);

  const spent = runs.reduce((sum, r) => sum + r.cost_usd, 0);
  const chosen = catalog?.suites.find((s) => s.id === suite);

  return (
    <>
      <div className="eyebrow">Evaluation</div>
      <h1>How do we know it works?</h1>
      <p className="lede">
        Seven small suites, one for each part of the agent, each graded by code against labelled
        data. {runs.length} rounds so far, US${spent.toFixed(3)} in all.
      </p>
      {error && <div className="error-box">{error}</div>}
      {catalog && (
        <ol className="protocol">
          {catalog.protocol.map((p, i) => (
            <li key={i}>{p}</li>
          ))}
        </ol>
      )}
      <div className="suites">
        {catalog?.suites.map((s) => (
          <SuiteCard
            key={s.id}
            suite={s}
            summary={latest[s.id]}
            rounds={runs.filter((r) => r.suite === s.id).length}
            chosen={s.id === suite}
            choose={() => setSuite(s.id)}
          />
        ))}
      </div>
      {chosen && <SuiteDetail key={chosen.id} suite={chosen} runs={runs.filter((r) => r.suite === chosen.id)} />}
    </>
  );
}

function headline(suite: SuiteInfo, summary: RunSummary | undefined, split: string): Metric | undefined {
  return summary?.metrics.find((m) => m.split === split && m.metric === suite.headline && m.arm === suite.arm);
}

function SuiteCard({
  suite,
  summary,
  rounds,
  chosen,
  choose,
}: {
  suite: SuiteInfo;
  summary: RunSummary | undefined;
  rounds: number;
  chosen: boolean;
  choose: () => void;
}) {
  const held = headline(suite, summary, "test");
  const shown = held ?? headline(suite, summary, "dev");
  return (
    <button className={`suite-card ${chosen ? "chosen" : ""}`} onClick={choose} aria-pressed={chosen}>
      <span className="suite-id mono">{suite.id.toUpperCase()}</span>
      <b>{suite.name}</b>
      <span className="suite-covers">{suite.covers}</span>
      <span className="suite-question">{suite.question}</span>
      <span className="suite-result">
        {shown ? (
          <>
            <span className="suite-value">{shown.mean.toFixed(2)}</span>
            <span>
              {suite.headline.replaceAll("_", " ")}, {held ? "held-out" : "dev only (no held-out round yet)"}
              <br />[{shown.low.toFixed(2)}, {shown.high.toFixed(2)}] · n {shown.n}
            </span>
          </>
        ) : (
          <span>{rounds ? "no complete round" : "not run yet"}</span>
        )}
      </span>
      <span className="suite-foot">
        {suite.items.dev} dev · {suite.items.test} held-out · {rounds} {rounds === 1 ? "round" : "rounds"}
      </span>
    </button>
  );
}

function SuiteDetail({ suite, runs }: { suite: SuiteInfo; runs: Run[] }) {
  const [chosen, setChosen] = useState<string | null>(runs.find((r) => r.status === "complete")?.id ?? runs[0]?.id ?? null);
  const [summary, setSummary] = useState<RunSummary | null>(null);

  useEffect(() => {
    if (!chosen) return;
    setSummary(null);
    api.run(chosen).then(setSummary).catch(() => setSummary(null));
  }, [chosen]);

  return (
    <section className="suite-detail">
      <header>
        <span className="eyebrow">
          {suite.id.toUpperCase()} · tests {suite.covers}
        </span>
        <h2>{suite.name}</h2>
        <p className="lede">{suite.question}</p>
      </header>
      <dl className="facts">
        <dt>Data</dt>
        <dd>{suite.data}</dd>
        <dt>Grading</dt>
        <dd>{suite.grading}</dd>
        <dt>Items</dt>
        <dd>
          {suite.items.dev} dev, {suite.items.test} held-out
        </dd>
      </dl>
      <h3 className="detail-head">Rounds</h3>
      {runs.length === 0 ? (
        <p className="notice">This suite has not been run yet.</p>
      ) : (
        <div className="rounds">
          {runs.map((r) => (
            <button key={r.id} className={`round ${r.id === chosen ? "chosen" : ""}`} onClick={() => setChosen(r.id)}>
              <b>{when(r.created_at)}</b>
              <span>
                {r.items} items · ${r.cost_usd.toFixed(3)}
              </span>
              {r.status !== "complete" && <span className="pill amber">{r.status}</span>}
            </button>
          ))}
        </div>
      )}
      {chosen && !summary && <p className="notice">Loading the round…</p>}
      {summary && <RoundView suite={suite} summary={summary} />}
    </section>
  );
}

function RoundView({ suite, summary }: { suite: SuiteInfo; summary: RunSummary }) {
  const arms = [...new Set(summary.metrics.map((m) => m.arm))];
  return (
    <>
      <p className="round-meta">
        <span className="mono">{summary.run.id}</span>
        <span>{summary.items.length} results</span>
        <span>{summary.items.filter((i) => i.error).length} errors</span>
        {summary.run.langsmith_experiment && <span className="mono">LangSmith: {summary.run.langsmith_experiment}</span>}
      </p>
      <div className={`results ${arms.length > 2 ? "stacked" : ""}`}>
        {SPLITS.map((split) => (
          <Results key={split.id} suite={suite} split={split} arms={arms} metrics={summary.metrics.filter((m) => m.split === split.id)} />
        ))}
      </div>
      <Items suite={suite} summary={summary} />
      {summary.usage.length > 0 && (
        <details className="more">
          <summary>Cost and latency by stage</summary>
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
                  <td className="num">{u.input_tokens ? `${Math.round((100 * u.cached_tokens) / u.input_tokens)}%` : "—"}</td>
                  <td className="num">${Number(u.usd).toFixed(4)}</td>
                  <td className="num">
                    {Math.round(u.p50_ms)} / {Math.round(u.p95_ms)} ms
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </>
  );
}

/** One split's metrics: each named and explained, the headline first, one column per arm. */
function Results({
  suite,
  split,
  arms,
  metrics,
}: {
  suite: SuiteInfo;
  split: (typeof SPLITS)[number];
  arms: string[];
  metrics: Metric[];
}) {
  if (metrics.length === 0) return null;
  const names = [...new Set(metrics.map((m) => m.metric))].sort(
    (a, b) => Number(b === suite.headline) - Number(a === suite.headline),
  );
  const cell = (name: string, arm: string) => metrics.find((m) => m.metric === name && m.arm === arm);
  return (
    <div className="split">
      <div className="split-head">
        <h3>{split.name}</h3>
        <span className="eyebrow">{split.note}</span>
      </div>
      <table className="data">
        <thead>
          <tr>
            <th>Metric</th>
            {arms.map((a) => (
              <th key={a} className="num">
                {arms.length > 1 ? a : "mean [95% CI]"}
                {a === suite.arm && arms.length > 1 ? " ★" : ""}
              </th>
            ))}
            <th className="num">n</th>
          </tr>
        </thead>
        <tbody>
          {names.map((name) => (
            <tr key={name} className={name === suite.headline ? "headline" : undefined}>
              <td>
                <b>{name.replaceAll("_", " ")}</b>
                {suite.metrics[name] && <small>{suite.metrics[name]}</small>}
              </td>
              {arms.map((a) => {
                const m = cell(name, a);
                return (
                  <td key={a} className="num">
                    {m ? (
                      <>
                        {m.mean.toFixed(2)}
                        <small>
                          [{m.low.toFixed(2)}, {m.high.toFixed(2)}]
                        </small>
                      </>
                    ) : (
                      "—"
                    )}
                  </td>
                );
              })}
              <td className="num">{Math.max(...metrics.filter((m) => m.metric === name).map((m) => m.n))}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Every item's result, those that missed the headline (or errored) first. */
function Items({ suite, summary }: { suite: SuiteInfo; summary: RunSummary }) {
  const items = useMemo(() => {
    const score = (i: RunSummary["items"][number]) => (i.error ? -1 : (i.metrics[suite.headline] ?? 1));
    return summary.items.filter((i) => i.arm === suite.arm || !summary.items.some((x) => x.arm === suite.arm)).sort((a, b) => score(a) - score(b));
  }, [suite, summary]);
  const missed = items.filter((i) => i.error || (i.metrics[suite.headline] ?? 1) < 1).length;
  return (
    <details className="more">
      <summary>
        Items ({items.length}) · {missed} below 1 on {suite.headline.replaceAll("_", " ")}
      </summary>
      <table className="data items">
        <thead>
          <tr>
            <th>Item</th>
            <th>Split</th>
            <th className="num">{suite.headline.replaceAll("_", " ")}</th>
            <th>Other metrics</th>
          </tr>
        </thead>
        <tbody>
          {items.map((i) => (
            <tr key={`${i.arm}-${i.item_id}`}>
              <td className="mono">{i.item_id}</td>
              <td>{i.split === "test" ? "held-out" : i.split}</td>
              <td className="num">{i.error ? "error" : (i.metrics[suite.headline]?.toFixed(2) ?? "—")}</td>
              <td className="item-metrics">
                {i.error ??
                  Object.entries(i.metrics)
                    .filter(([k]) => k !== suite.headline)
                    .map(([k, v]) => `${k.replaceAll("_", " ")} ${Number.isInteger(v) ? v : v.toFixed(2)}`)
                    .join(" · ")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}
