import type { Call } from "../api";

const usd = (v: number) => `$${v.toFixed(v < 0.01 ? 5 : 4)}`;
const ms = (v: number | null) => (v === null ? "—" : v < 1000 ? `${v} ms` : `${(v / 1000).toFixed(1)} s`);
const short = (model: string) => model.replace("gpt-6-", "").replace("text-embedding-3-", "emb-");

/** Every model call of the turn, from the ledger: what each step cost and how much it reused. */
export function Timeline({ calls }: { calls: Call[] }) {
  const cost = calls.reduce((sum, c) => sum + (c.cost_usd ?? 0), 0);
  const input = calls.reduce((sum, c) => sum + (c.input_tokens ?? 0), 0);
  const cached = calls.reduce((sum, c) => sum + (c.cached_tokens ?? 0), 0);
  return (
    <section className="timeline">
      <div className="eyebrow">Model calls, from the cost ledger</div>
      <div className="totals">
        <div className="figure">
          <b>{calls.length}</b>
          <span>model calls</span>
        </div>
        <div className="figure">
          <b>{usd(cost)}</b>
          <span>this turn</span>
        </div>
        <div className="figure">
          <b>{input ? Math.round((100 * cached) / input) : 0}%</b>
          <span>input from cache</span>
        </div>
      </div>
      {calls.length === 0 ? (
        <p className="notice">Model calls appear here as the turn runs.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Stage</th>
              <th>Model</th>
              <th className="num">Latency</th>
              <th className="num">In / cached</th>
              <th className="num">Out</th>
              <th className="num">Cost</th>
            </tr>
          </thead>
          <tbody>
            {calls.map((c) => (
              <tr key={c.id}>
                <td>
                  {c.stage}
                  {c.status !== "settled" && <span className="pill rust"> {c.status}</span>}
                </td>
                <td className="mono">{short(c.model)}</td>
                <td className="num">{ms(c.latency_ms)}</td>
                <td className="num">
                  {c.input_tokens ?? "—"}
                  {c.cached_tokens ? ` / ${c.cached_tokens}` : ""}
                </td>
                <td className="num">{c.output_tokens ?? "—"}</td>
                <td className="num">{c.cost_usd === null ? "—" : usd(c.cost_usd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
