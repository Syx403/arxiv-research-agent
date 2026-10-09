import { useCallback, useEffect, useState } from "react";

import { api } from "./api";
import type { ConversationSummary, Graphs } from "./api";
import { useChat } from "./chat";
import { ChatView } from "./components/Chat";
import type { Open } from "./components/Chat";
import { Panel } from "./components/Panel";
import type { PanelState } from "./components/Panel";
import { Sidebar } from "./components/Sidebar";
import { EvaluationPage } from "./pages/Evaluation";
import { MemoryPage } from "./pages/Memory";

type Route = { page: "chat"; id: string | null } | { page: "evaluation" } | { page: "memory" };

function route(): Route {
  const hash = location.hash.replace(/^#\/?/, "");
  if (hash === "evaluation" || hash === "memory") return { page: hash };
  const id = /^c\/([\w-]+)$/.exec(hash)?.[1] ?? null;
  return { page: "chat", id };
}

function stored(key: string, fallback: boolean): boolean {
  try {
    return (localStorage.getItem(key) ?? String(fallback)) === "true";
  } catch {
    return fallback;
  }
}

export function App() {
  const [at, setAt] = useState<Route>(route);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [graphs, setGraphs] = useState<Graphs | null>(null);
  const [panel, setPanel] = useState<PanelState | null>(null);
  const [collapsed, setCollapsed] = useState(() => stored("ara.sidebar.collapsed", false));

  const refresh = useCallback(() => {
    api.conversations().then(setConversations).catch(() => undefined);
  }, []);
  // a new conversation takes its id without a history entry; a removed one goes back to new
  const go = useCallback((id: string | null) => {
    history.replaceState(null, "", id ? `#/c/${id}` : "#/");
    setAt({ page: "chat", id });
  }, []);
  const chat = useChat(at.page === "chat" ? at.id : null, go, refresh);

  useEffect(() => {
    refresh();
    api.graph().then(setGraphs).catch(() => setGraphs(null));
    const follow = () => {
      setAt(route());
      setPanel(null);
    };
    window.addEventListener("hashchange", follow);
    return () => window.removeEventListener("hashchange", follow);
  }, [refresh]);

  useEffect(() => {
    try {
      localStorage.setItem("ara.sidebar.collapsed", String(collapsed));
    } catch {
      // the choice is not remembered in a private window
    }
  }, [collapsed]);

  // a panel opened on the running turn follows it into its record when it finishes
  const last = chat.conversation?.turns.at(-1)?.turn_id;
  useEffect(() => {
    if (!chat.pending && last) setPanel((p) => (p?.key === "pending" ? { ...p, key: last } : p));
  }, [chat.pending, last]);

  const open: Open = (key, tab, extra) =>
    setPanel({ key, tab, cited: extra?.cited ?? null, paper: extra?.paper ?? null });

  const turns = chat.conversation?.turns ?? [];
  const shown =
    panel?.key === "pending" && chat.pending
      ? { message: chat.pending.message, trace: chat.pending.trace, calls: chat.pending.calls, result: null }
      : (() => {
          const t = turns.find((x) => x.turn_id === panel?.key);
          return t ? { message: t.message, trace: t.trace, calls: t.calls, result: t } : null;
        })();

  return (
    <div className={`shell ${collapsed ? "narrow" : ""} ${panel && shown ? "with-panel" : ""}`}>
      <Sidebar
        conversations={conversations}
        running={new Set(Object.keys(chat.runs))}
        current={at.page === "chat" ? chat.id : null}
        page={at.page}
        collapsed={collapsed}
        toggle={() => setCollapsed((c) => !c)}
        changed={(removed) => {
          refresh();
          if (removed && removed === chat.id) location.hash = "#/";
        }}
      />
      <main className="stage" key={at.page === "chat" ? "chat" : at.page}>
        {at.page === "chat" && <ChatView chat={chat} open={open} />}
        {at.page === "evaluation" && (
          <div className="page">
            <EvaluationPage />
          </div>
        )}
        {at.page === "memory" && (
          <div className="page">
            <MemoryPage />
          </div>
        )}
      </main>
      {panel && shown && (
        <Panel
          state={panel}
          turn={shown}
          live={panel.key === "pending"}
          graphs={graphs}
          set={(s) => setPanel((p) => p && { ...p, ...s })}
          close={() => setPanel(null)}
        />
      )}
    </div>
  );
}
