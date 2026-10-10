import type { ReactNode } from "react";

import type { Claim, Paper, Part, Turn } from "../api";

const CITATION = /\[(E\d+(?:\s*,\s*E\d+)*)\]/g;

/** The agent's reply, from the turn's own data (D39): listed papers from `turn.papers`, the
 *  answer from `turn.answer` (the direct answer, then its explanation in paragraphs, every
 *  sentence with its citation chips), notes and warnings as such. Turns recorded before D39 have
 *  no parts and are shown as their text. */
export function Reply({
  turn,
  onCite,
  onPaper,
}: {
  turn: Turn;
  onCite: (id: string) => void;
  onPaper: (paper: Paper | undefined) => void;
}) {
  const parts: Part[] = turn.parts?.length ? turn.parts : [{ kind: "text", text: turn.reply }];
  return (
    <div className="reply">
      {parts.map((part, i) => (
        <PartView key={i} part={part} turn={turn} onCite={onCite} onPaper={onPaper} />
      ))}
    </div>
  );
}

function PartView({
  part,
  turn,
  onCite,
  onPaper,
}: {
  part: Part;
  turn: Turn;
  onCite: (id: string) => void;
  onPaper: (paper: Paper | undefined) => void;
}) {
  switch (part.kind) {
    case "listing":
      return (
        <div className="reply-block">
          {part.source === "library" && <p className="listing-head">From your library</p>}
          <ol className="listing">
            {turn.papers.map((paper) => (
              <li key={paper.arxiv_id}>
                <button className="listing-item" onClick={() => onPaper(paper)}>
                  <span className="listing-title">{paper.title}</span>
                  <span className="listing-meta">
                    <span className="mono">
                      {paper.arxiv_id}v{paper.version}
                    </span>
                    {paper.published && <span>{paper.published}</span>}
                    {paper.relevance != null && <span>relevance {paper.relevance}</span>}
                    {paper.read_before && <span>read before</span>}
                  </span>
                  <span className="listing-reason">{paper.reason}</span>
                </button>
              </li>
            ))}
          </ol>
        </div>
      );
    case "answer":
      return turn.answer ? <AnswerView answer={turn.answer} onCite={onCite} /> : null;
    case "note":
    case "warning":
      return <p className="note">{part.text.replace(/^Note: /, "")}</p>;
    case "context":
      return <p className="context">{chipped(part.text, onCite)}</p>;
    default:
      return (
        <div className="reply-block">
          {part.text
            .split("\n")
            .filter(Boolean)
            .map((line, i) => (
              <p key={i}>{chipped(line, onCite)}</p>
            ))}
        </div>
      );
  }
}

/** The direct answer (an abstention in plain type), then each paragraph of verified sentences. */
function AnswerView({ answer, onCite }: { answer: NonNullable<Turn["answer"]>; onCite: (id: string) => void }) {
  const paragraphs = new Map<number, Claim[]>();
  for (const sentence of answer.sentences) {
    paragraphs.set(sentence.paragraph, [...(paragraphs.get(sentence.paragraph) ?? []), sentence]);
  }
  return (
    <div className="reply-block answer">
      <p className={answer.abstained ? undefined : "lead"}>{answer.short}</p>
      {[...paragraphs.values()].map((claims, i) => (
        <p key={i}>
          {claims.map((c) => (
            <span key={c.index}>
              {c.text}
              {c.citations.map((id) => (
                <button key={id} className="chip" onClick={() => onCite(id)}>
                  {id}
                </button>
              ))}{" "}
            </span>
          ))}
        </p>
      ))}
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
