import type { ReactNode } from "react";

import type { Paper, Turn } from "../api";

const CITATION = /\[(E\d+(?:\s*,\s*E\d+)*)\]/g;
const SOURCE_LINE = /^\[E\d+\] /;
const LISTING = /^(\d+)\. (.+) \(arXiv ([^,)]+)(?:, ([^)]*))?\) — (.*)$/;

/** The agent's reply as written, with citations as chips and listed papers as rows. The
 *  "[E1] section (arXiv …)" source lines are left to the side panel. */
export function Reply({
  turn,
  onCite,
  onPaper,
}: {
  turn: Turn;
  onCite: (id: string) => void;
  onPaper: (paper: Paper | undefined) => void;
}) {
  const blocks = turn.reply.split(/\n\n+/).map((b) => b.split("\n").filter((l) => !SOURCE_LINE.test(l)));
  const lead = turn.answer && !turn.answer.abstained ? turn.answer.short : null;
  return (
    <div className="reply">
      {blocks.map((lines, b) => {
        if (lines.length === 0) return null;
        if (lines.every((l) => LISTING.test(l))) {
          return (
            <ol key={b} className="listing">
              {lines.map((line) => {
                const [, n, title, id, date, reason] = LISTING.exec(line)!;
                const paper = turn.papers[Number(n) - 1] ?? turn.read[Number(n) - 1];
                return (
                  <li key={n}>
                    <button className="listing-item" onClick={() => onPaper(paper)}>
                      <span className="listing-title">{title}</span>
                      <span className="listing-meta">
                        <span className="mono">{id}</span>
                        {date && <span>{date}</span>}
                        {paper?.relevance != null && <span>relevance {paper.relevance}</span>}
                      </span>
                      <span className="listing-reason">{reason}</span>
                    </button>
                  </li>
                );
              })}
            </ol>
          );
        }
        return (
          <div key={b} className="reply-block">
            {lines.map((line, i) => {
              if (line.startsWith("Note: ")) {
                return (
                  <p key={i} className="note">
                    {line.slice(6)}
                  </p>
                );
              }
              const isLead = lead !== null && line.replace(CITATION, "").trim() === lead;
              return (
                <p key={i} className={isLead ? "lead" : undefined}>
                  {chipped(line, onCite)}
                </p>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

function chipped(line: string, onCite: (id: string) => void): ReactNode[] {
  const parts: ReactNode[] = [];
  let at = 0;
  for (const match of line.matchAll(CITATION)) {
    parts.push(line.slice(at, match.index).trimEnd());
    for (const id of match[1].split(/\s*,\s*/)) {
      parts.push(
        <button key={`${match.index}-${id}`} className="chip" onClick={() => onCite(id)}>
          {id}
        </button>,
      );
    }
    at = match.index + match[0].length;
  }
  parts.push(line.slice(at));
  return parts;
}
