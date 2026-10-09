import { useCallback, useEffect, useRef, useState } from "react";

import { api, stream } from "./api";
import type { Call, Conversation, NodeEvent, Phase, StreamEvent } from "./api";

/** The turn in progress: what was sent and what has happened so far. */
export interface Pending {
  message: string;
  trace: NodeEvent[];
  calls: Call[];
  error: string | null;
}

/** One conversation: its recorded turns, and the turn running now. `id` follows the route;
 *  a new conversation gets its id from its first message. */
export function useChat(routeId: string | null, go: (id: string) => void, changed: () => void) {
  const [id, setId] = useState<string | null>(routeId);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const [missing, setMissing] = useState(false);
  const own = useRef<string | null>(null); // an id this hook just created: nothing to load yet

  useEffect(() => {
    if (routeId === own.current && routeId !== null) return;
    setId(routeId);
    setPending(null);
    setMissing(false);
    setConversation(null);
    if (!routeId) return;
    let live = true;
    api
      .conversation(routeId)
      .then((c) => live && setConversation(c))
      .catch(() => live && setMissing(true));
    return () => {
      live = false;
    };
  }, [routeId]);

  const run = useCallback(
    async (kind: "messages" | "resume", text: string) => {
      let current = id;
      if (!current) {
        current = (await api.newConversation()).id;
        own.current = current;
        setId(current);
        go(current);
      }
      setPending({ message: text, trace: [], calls: [], error: null });
      try {
        for await (const event of stream(current, kind, kind === "messages" ? { text } : { answer: text }))
          handle(event);
      } catch (error) {
        setPending((p) => p && { ...p, error: String(error) });
        return;
      }
      changed();

      function handle(event: StreamEvent) {
        switch (event.event) {
          case "node":
            setPending((p) => p && { ...p, trace: [...p.trace, event.data] });
            break;
          case "call":
            setPending((p) => p && { ...p, calls: [...p.calls, event.data] });
            break;
          case "error":
            setPending((p) => p && { ...p, error: event.data.error });
            break;
          case "done": {
            const turn = event.data;
            const now = new Date().toISOString();
            setConversation((c) =>
              c
                ? { ...c, turns: [...c.turns, turn], updated_at: now }
                : { id: current!, title: text, created_at: now, updated_at: now, turns: [turn] },
            );
            setPending(null);
          }
        }
      }
    },
    [id, go, changed],
  );

  const waiting = conversation?.turns.at(-1)?.waiting ?? null;
  const send = useCallback((text: string) => run(waiting ? "resume" : "messages", text), [run, waiting]);

  return { id, conversation, pending, missing, waiting, send };
}

/** Each node's last phase in a turn; a node its error handler took over stays "degraded". */
export function phases(trace: NodeEvent[]): Record<string, Record<string, Phase>> {
  const found: Record<string, Record<string, Phase>> = {};
  for (const { graph, node, phase } of trace) {
    const before = found[graph]?.[node];
    const sticky = (before === "degraded" || before === "failed") && phase === "end";
    found[graph] = { ...found[graph], [node]: sticky ? before : phase };
  }
  return found;
}

export const LABELS: Record<string, string> = {
  load_context: "Recalling your memory",
  understand: "Understanding the request",
  clarify: "Asking a question",
  resolve: "Finding the papers you named",
  discover: "Searching arXiv",
  researcher: "Planning searches",
  arxiv_tools: "Searching arXiv",
  prerank: "Ranking candidates",
  prewarm: "Preparing the cache",
  screen: "Screening abstracts",
  rank: "Ranking papers",
  choose_papers: "Choosing papers to read",
  library: "Looking through your library",
  read: "Reading papers",
  ingest: "Fetching full text",
  search: "Searching the papers",
  select: "Selecting evidence",
  collect: "Collecting evidence",
  conflicts: "Checking your requirements",
  answer: "Writing the answer",
  synthesize: "Writing the answer",
  verify: "Verifying each line",
  repair: "Repairing rejected lines",
  finalize: "Keeping verified lines",
  search_instead: "Searching arXiv instead",
  respond: "Replying",
  remember: "Updating memory",
};

/** What the turn is doing now: the latest node that started and has not ended. */
export function doing(trace: NodeEvent[]): string {
  const open = new Map<string, NodeEvent>();
  for (const event of trace) {
    const key = `${event.graph}/${event.node}`;
    if (event.phase === "start") open.set(key, event);
    else open.delete(key);
  }
  const last = [...open.values()].at(-1);
  return last ? (LABELS[last.node] ?? last.node) : "Thinking";
}
