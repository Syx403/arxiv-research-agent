import { useEffect, useRef, useState } from "react";

import type { Paper, Turn } from "../api";
import type { Pending, useChat } from "../chat";
import { Process } from "./Process";
import type { Tab } from "./Process";
import { Reply } from "./Reply";

const SUGGESTIONS = [
  { title: "Read a paper", text: "Read 2210.03629 and explain how it interleaves reasoning and actions." },
  { title: "Compare two papers", text: "Compare 2305.18323 and 2312.04511: how does each avoid calling the LLM after every tool call?" },
  { title: "Find recent work", text: "Find recent papers on KV-cache eviction for long-context inference." },
  { title: "Ask what we read", text: "Which papers have we read about LLM agents that call tools?" },
];

type Chat = ReturnType<typeof useChat>;
export type Open = (key: string, tab: Tab, extra?: { cited?: string; paper?: string }) => void;

/** The conversation: a centred column of turns, each user message followed by the agent's reply
 *  and the fold that leads to its workflow and evidence; the composer below. */
export function ChatView({ chat, open }: { chat: Chat; open: Open }) {
  const end = useRef<HTMLDivElement>(null);
  const turns = chat.conversation?.turns ?? [];
  const empty = turns.length === 0 && !chat.pending;

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length, chat.pending?.trace.length]);

  if (chat.missing) {
    return (
      <div className="chat-empty">
        <h1>This conversation no longer exists.</h1>
        <a href="#/">Start a new one</a>
      </div>
    );
  }

  if (empty) {
    return (
      <div className="chat-empty">
        <div className="greeting">
          <span className="mark" aria-hidden>
            ✳
          </span>
          <h1>What are you researching?</h1>
        </div>
        <p className="lede">
          I find arXiv papers, read them, and answer with every line checked against the sentence
          it cites.
        </p>
        <Composer send={chat.send} busy={false} waiting={false} autoFocus />
        <div className="starters">
          {SUGGESTIONS.map((s) => (
            <button key={s.title} className="starter" onClick={() => void chat.send(s.text)}>
              <b>{s.title}</b>
              <span>{s.text}</span>
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="chat">
      <header className="chat-title">
        <h2>{chat.conversation?.title ?? chat.pending?.message}</h2>
      </header>
      <div className="thread">
        {turns.map((turn) => (
          <TurnView key={turn.turn_id} turn={turn} open={open} />
        ))}
        {chat.pending && <PendingView pending={chat.pending} open={open} />}
        <div ref={end} />
      </div>
      <div className="dock">
        <Composer send={chat.send} busy={!!chat.pending && !chat.pending.error} waiting={!!chat.waiting} />
        <p className="dock-note">Answers cite arXiv text; each delivered line passed a verifier.</p>
      </div>
    </div>
  );
}

function TurnView({ turn, open }: { turn: Turn; open: Open }) {
  const key = turn.turn_id;
  const cite = (id: string) => open(key, "evidence", { cited: id });
  const paper = (p: Paper | undefined) => open(key, "papers", { paper: p?.arxiv_id });
  return (
    <>
      <div className="msg user">{turn.message}</div>
      <div className="msg agent">
        {turn.waiting ? (
          <div className="asking">
            <span className="eyebrow">A question before I go on</span>
            <p>{turn.waiting.question}</p>
          </div>
        ) : (
          <Reply turn={turn} onCite={cite} onPaper={paper} />
        )}
        <Process
          trace={turn.trace}
          calls={turn.calls}
          live={false}
          timing={turn}
          counts={{
            read: turn.read.length,
            listed: turn.papers.length,
            evidence: turn.answer?.evidence.length ?? 0,
          }}
          open={(tab) => open(key, tab)}
        />
      </div>
    </>
  );
}

function PendingView({ pending, open }: { pending: Pending; open: Open }) {
  return (
    <>
      <div className="msg user">{pending.message}</div>
      <div className="msg agent">
        {pending.error ? (
          <div className="error-box">{pending.error}</div>
        ) : (
          <Process trace={pending.trace} calls={pending.calls} live open={(tab) => open("pending", tab)} />
        )}
      </div>
    </>
  );
}

function Composer({
  send,
  busy,
  waiting,
  autoFocus,
}: {
  send: (text: string) => Promise<void>;
  busy: boolean;
  waiting: boolean;
  autoFocus?: boolean;
}) {
  const [draft, setDraft] = useState("");
  const submit = () => {
    const text = draft.trim();
    if (!text || busy) return;
    setDraft("");
    void send(text);
  };
  return (
    <div className="composer">
      <textarea
        autoFocus={autoFocus}
        rows={1}
        value={draft}
        placeholder={waiting ? "Answer the question…" : "Ask for papers, or about a paper…"}
        onChange={(e) => {
          setDraft(e.target.value);
          e.target.style.height = "auto";
          e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
      />
      <button className="send" onClick={submit} disabled={busy || !draft.trim()} aria-label="Send">
        ↑
      </button>
    </div>
  );
}
