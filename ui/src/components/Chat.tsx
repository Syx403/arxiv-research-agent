import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";

import type { Paper, Turn } from "../api";
import type { Draft, Pending, useChat } from "../chat";
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
  const running = !!chat.pending && !chat.pending.error;

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length, chat.pending?.trace.length, chat.pending?.problems.length]);

  const composer = (
    <Composer
      send={chat.send}
      stop={chat.stop}
      running={running}
      waiting={!!chat.waiting}
      draft={chat.draft}
      taken={() => chat.setDraft(null)}
      autoFocus
    />
  );

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
        {composer}
        <div className="starters">
          {SUGGESTIONS.map((s) => (
            <button key={s.title} className="starter" onClick={() => chat.setDraft({ text: s.text, n: Date.now() })}>
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
        <h2>{chat.conversation?.title || chat.pending?.message}</h2>
      </header>
      <div className="thread">
        {turns.map((turn, i) => (
          <TurnView key={turn.turn_id} turn={turn} answered={i < turns.length - 1 || !!chat.pending} open={open} />
        ))}
        {chat.pending && <PendingView pending={chat.pending} open={open} />}
        <div ref={end} />
      </div>
      <div className="dock">
        {composer}
        <p className="dock-note">
          {running
            ? "The turn runs on even if you leave this conversation. Esc or ■ stops it."
            : "Answers cite arXiv text; each delivered line passed a verifier."}
        </p>
      </div>
    </div>
  );
}

function TurnView({ turn, answered, open }: { turn: Turn; answered: boolean; open: Open }) {
  const key = turn.turn_id;
  const cite = (id: string) => open(key, "evidence", { cited: id });
  const paper = (p: Paper | undefined) => open(key, "papers", { paper: p?.arxiv_id });
  return (
    <>
      <div className="msg user">{turn.message}</div>
      <div className="msg agent">
        {turn.waiting ? (
          <div className={`asking ${answered ? "answered" : ""}`}>
            <span className="eyebrow">{answered ? "I asked" : "A question before I go on"}</span>
            <p>{turn.waiting.question}</p>
          </div>
        ) : (
          <Reply turn={turn} onCite={cite} onPaper={paper} />
        )}
        <Process
          trace={turn.trace}
          calls={turn.calls}
          problems={[...new Set(turn.problems)]}
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
          <Process
            trace={pending.trace}
            calls={pending.calls}
            problems={pending.problems}
            live
            since={pending.since}
            open={(tab) => open("pending", tab)}
          />
        )}
      </div>
    </>
  );
}

/** The message box. Enter sends, Shift+Enter starts a line; Enter that confirms an input
 *  method's composition (Chinese, Japanese, ...) only confirms it. While a turn runs the button
 *  stops it (Esc too), and a stopped turn's message comes back here to be edited. */
function Composer({
  send,
  stop,
  running,
  waiting,
  draft,
  taken,
  autoFocus,
}: {
  send: (text: string) => Promise<void>;
  stop: () => Promise<void>;
  running: boolean;
  waiting: boolean;
  draft: Draft | null;
  taken: () => void;
  autoFocus?: boolean;
}) {
  const [text, setText] = useState("");
  const composing = useRef(false);
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!draft) return;
    setText(draft.text);
    box.current?.focus();
    taken(); // used once: a later composer starts empty
  }, [draft, taken]);

  useEffect(() => {
    const area = box.current;
    if (!area) return;
    area.style.height = "auto";
    area.style.height = `${Math.min(area.scrollHeight, 200)}px`;
  }, [text]);

  const submit = () => {
    const message = text.trim();
    if (!message || running) return;
    setText("");
    void send(message);
  };

  const keyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // keyCode 229: Safari reports the Enter that ends a composition after compositionend
    if (composing.current || e.nativeEvent.isComposing || e.keyCode === 229) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
    if (e.key === "Escape" && running) void stop();
  };

  return (
    <div className="composer">
      <textarea
        ref={box}
        autoFocus={autoFocus}
        rows={1}
        value={text}
        placeholder={waiting ? "Answer the question…" : "Ask for papers, or about a paper…"}
        onChange={(e) => setText(e.target.value)}
        onCompositionStart={() => (composing.current = true)}
        onCompositionEnd={() => (composing.current = false)}
        onKeyDown={keyDown}
      />
      {running ? (
        <button className="send stop" onClick={() => void stop()} aria-label="Stop" title="Stop (Esc)">
          ■
        </button>
      ) : (
        <button className="send" onClick={submit} disabled={!text.trim()} aria-label="Send" title="Send (Enter)">
          ↑
        </button>
      )}
    </div>
  );
}
