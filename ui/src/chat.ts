import { useCallback, useEffect, useRef, useState } from "react";

import { api, stream } from "./api";
import type { Call, Conversation, NodeEvent, Phase, StreamEvent } from "./api";

/** A turn running on the server: what was sent and what has happened so far. */
export interface Pending {
  message: string;
  kind: "messages" | "resume";
  trace: NodeEvent[];
  calls: Call[];
  problems: string[];
  since: number; // when this page saw it start (ms)
}

/** Text put back in the composer: a stopped turn's message, or a starter (`n` makes each new). */
export interface Draft {
  text: string;
  n: number;
}

/** The open conversation and every turn this page is following. A turn runs on the server
 *  whatever the page does (D38): switching away and back, or reloading, rejoins it; only Stop
 *  ends it early, and its message comes back to the composer. */
export function useChat(routeId: string | null, go: (id: string | null) => void, changed: () => void) {
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [missing, setMissing] = useState(false);
  const [runs, setRuns] = useState<Record<string, Pending>>({});
  const [draft, setDraft] = useState<Draft | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const followed = useRef(new Set<string>());
  const route = useRef(routeId);
  route.current = routeId;
  const runsNow = useRef(runs); // the runs as of the last render, for a message to give back
  runsNow.current = runs;

  const update = useCallback((id: string, change: (p: Pending) => Pending | null) => {
    setRuns((all) => {
      const before = all[id];
      if (!before) return all;
      const after = change(before);
      const { [id]: _, ...rest } = all;
      return after ? { ...rest, [id]: after } : rest;
    });
  }, []);

  /** Read a turn's events until it ends; each goes to its own conversation, open or not. A
   *  stopped or failed turn has been undone on the server: its message returns to the composer
   *  with a notice (D38, D39). A stream that breaks without an end rejoins the turn, or, if it
   *  has ended meanwhile, reloads the conversation. */
  const follow = useCallback(
    async (id: string, events: AsyncGenerator<StreamEvent>): Promise<void> => {
      followed.current.add(id);
      let ended = false;
      let started = false;
      let failure = "";
      try {
        for await (const event of events) {
          switch (event.event) {
            case "started":
              started = true;
              update(id, (p) => ({ ...p, since: Date.parse(event.data.started_at) }));
              changed(); // listed (and marked running) from its first message
              break;
            case "node":
              update(id, (p) => ({ ...p, trace: [...p.trace, event.data] }));
              break;
            case "call":
              update(id, (p) => ({ ...p, calls: [...p.calls, event.data] }));
              break;
            case "problem":
              update(id, (p) => ({ ...p, problems: [...p.problems, event.data.text] }));
              break;
            case "done": {
              ended = true;
              const turn = event.data;
              setConversation((c) => {
                if (c?.id === id) return { ...c, turns: [...c.turns, turn], running: null };
                if (c === null && route.current === id) {
                  const at = turn.started_at;
                  return { id, title: turn.message, created_at: at, updated_at: at, turns: [turn], running: null };
                }
                return c;
              });
              update(id, () => null);
              break;
            }
            case "stopped":
            case "error": {
              ended = true;
              const undone = event.data;
              update(id, () => null);
              if (route.current === id) {
                setDraft({ text: undone.message, n: Date.now() });
                setNotice(
                  event.event === "error"
                    ? `${undone.error} Your message is back in the box; nothing was kept.`
                    : `Stopped. Your message is back in the box ($${undone.spent_usd.toFixed(4)} spent).`,
                );
                if (undone.conversation_removed) go(null);
                else setConversation((c) => (c?.id === id ? { ...c, running: null } : c));
              }
            }
          }
        }
      } catch (error) {
        failure = String(error); // the connection broke: handled below like an early end
      } finally {
        followed.current.delete(id);
        changed();
      }
      if (ended) return;
      const message = runsNow.current[id]?.message ?? "";
      const now = await api.conversation(id).catch(() => null);
      if (now?.running) return follow(id, stream(id, "live"));
      update(id, () => null);
      if (route.current !== id) return;
      if (now) setConversation(now);
      if (!started) {
        // refused before it began (another turn is running, or the server is unreachable)
        setDraft({ text: message, n: Date.now() });
        setNotice(`The message was not sent (${failure || "no answer"}). It is back in the box.`);
      }
    },
    [changed, go, update],
  );

  useEffect(() => {
    setMissing(false);
    setConversation(null);
    if (routeId) setNotice(null);
    if (!routeId) return;
    let live = true;
    api
      .conversation(routeId)
      .then((c) => {
        if (!live) return;
        setConversation(c);
        if (c.running && !followed.current.has(routeId)) {
          const { message, kind } = c.running;
          setRuns((all) => ({ ...all, [routeId]: fresh(message, kind) }));
          void follow(routeId, stream(routeId, "live"));
        }
      })
      .catch(() => {
        // a conversation this page has just started is not stored until its turn begins
        if (live && !followed.current.has(routeId)) setMissing(true);
      });
    return () => {
      live = false;
    };
  }, [routeId, follow]);

  const pending = routeId ? (runs[routeId] ?? null) : null;
  const waiting = pending ? null : (conversation?.turns.at(-1)?.waiting ?? null);

  const send = useCallback(
    async (text: string) => {
      const id = routeId ?? (await api.newConversation()).id;
      const kind = waiting ? "resume" : "messages";
      setNotice(null);
      setRuns((all) => ({ ...all, [id]: fresh(text, kind) }));
      followed.current.add(id);
      if (!routeId) {
        setConversation({ id, title: text, created_at: "", updated_at: "", turns: [], running: null });
        go(id);
      }
      await follow(id, stream(id, kind, text));
    },
    [routeId, waiting, go, follow],
  );

  const stop = useCallback(async () => {
    if (!routeId) return;
    try {
      await api.stop(routeId); // the turn's own stream reports "stopped"
    } catch {
      // it finished as the stop was sent: its record arrives on the stream
    }
  }, [routeId]);

  return { id: routeId, conversation, pending, runs, missing, waiting, draft, setDraft, notice, send, stop };
}

function fresh(message: string, kind: "messages" | "resume"): Pending {
  return { message, kind, trace: [], calls: [], problems: [], since: Date.now() };
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
