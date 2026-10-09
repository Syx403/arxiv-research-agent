import { useCallback, useEffect, useState } from "react";

import { api, stream } from "./api";
import type { Call, Message, Phase, StreamEvent, Turn, Waiting } from "./api";

const THREAD = "ara.thread";

export type Nodes = Record<string, Record<string, Phase>>;

export interface TurnState {
  thread: string | null;
  messages: Message[];
  nodes: Nodes;
  calls: Call[];
  result: Turn | null;
  waiting: Waiting | null;
  busy: boolean;
  error: string | null;
}

const empty: Omit<TurnState, "thread"> = {
  messages: [],
  nodes: {},
  calls: [],
  result: null,
  waiting: null,
  busy: false,
  error: null,
};

function remembered(): string | null {
  try {
    return localStorage.getItem(THREAD);
  } catch {
    return null;
  }
}

function keep(thread: string): void {
  try {
    localStorage.setItem(THREAD, thread);
  } catch {
    // private window: the conversation simply is not restored on reload
  }
}

/** The conversation and the turn in progress: what every page reads. */
export function useTurn() {
  const [state, setState] = useState<TurnState>({ ...empty, thread: remembered() });

  useEffect(() => {
    const thread = state.thread;
    if (!thread) return;
    api
      .thread(thread)
      .then((t) =>
        setState((s) => ({
          ...s,
          messages: t.messages,
          result: t.answer || t.papers.length ? t : null,
          waiting: t.waiting,
        })),
      )
      .catch(() => undefined); // a thread the server no longer knows starts empty
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const run = useCallback(
    async (kind: "messages" | "resume", body: Record<string, string>, shown: Message | null) => {
      let thread = state.thread;
      if (!thread) {
        thread = (await api.newThread()).id;
        keep(thread);
      }
      setState((s) => ({
        ...s,
        thread,
        messages: shown ? [...s.messages, shown] : s.messages,
        nodes: kind === "messages" ? {} : s.nodes,
        calls: kind === "messages" ? [] : s.calls,
        waiting: null,
        busy: true,
        error: null,
      }));
      try {
        for await (const event of stream(thread, kind, body)) setState((s) => apply(s, event));
      } catch (error) {
        setState((s) => ({ ...s, error: String(error) }));
      } finally {
        setState((s) => ({ ...s, busy: false }));
      }
    },
    [state.thread],
  );

  const send = useCallback(
    (text: string) => run("messages", { text }, { role: "user", text }),
    [run],
  );
  const resume = useCallback(
    (answer: string) => run("resume", { answer }, { role: "user", text: answer }),
    [run],
  );
  const reset = useCallback(() => {
    try {
      localStorage.removeItem(THREAD);
    } catch {
      // nothing stored
    }
    setState({ ...empty, thread: null });
  }, []);

  return { ...state, send, resume, reset };
}

function apply(s: TurnState, event: StreamEvent): TurnState {
  switch (event.event) {
    case "node": {
      const { graph, node, phase } = event.data;
      const before = s.nodes[graph]?.[node];
      const settled = before === "degraded" || before === "failed";
      const next = settled && phase === "end" ? before : phase;
      return { ...s, nodes: { ...s.nodes, [graph]: { ...s.nodes[graph], [node]: next } } };
    }
    case "call":
      return { ...s, calls: [...s.calls, event.data] };
    case "interrupt":
      return {
        ...s,
        waiting: event.data,
        messages: [...s.messages, { role: "assistant", text: event.data.question }],
      };
    case "done":
      return { ...s, messages: event.data.messages, result: event.data };
    case "error":
      return { ...s, error: event.data.error };
  }
}
