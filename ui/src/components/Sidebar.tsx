import { useState } from "react";

import { api } from "../api";
import type { ConversationSummary } from "../api";

const DAY = 86_400_000;

function group(updated: string, now: Date): string {
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const at = new Date(updated).getTime();
  if (at >= start) return "Today";
  if (at >= start - DAY) return "Yesterday";
  if (at >= start - 7 * DAY) return "Previous 7 days";
  return "Older";
}

/** Conversations, newest first, grouped by day; rename and delete in place; the global pages. */
export function Sidebar({
  conversations,
  running,
  current,
  page,
  collapsed,
  toggle,
  changed,
}: {
  conversations: ConversationSummary[];
  running: Set<string>; // turns this page follows; the list marks the server's own
  current: string | null;
  page: string;
  collapsed: boolean;
  toggle: () => void;
  changed: (removed?: string) => void;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const now = new Date();
  const groups = new Map<string, ConversationSummary[]>();
  for (const c of conversations) {
    const g = group(c.updated_at, now);
    groups.set(g, [...(groups.get(g) ?? []), c]);
  }

  const save = async (id: string) => {
    if (draft.trim()) await api.rename(id, draft.trim());
    setEditing(null);
    changed();
  };
  const remove = async (c: ConversationSummary) => {
    if (!window.confirm(`Delete “${c.title}”? Its turns, evidence and checkpoints are removed.`)) return;
    await api.remove(c.id);
    changed(c.id);
  };

  if (collapsed) {
    return (
      <aside className="sidebar collapsed">
        <button className="icon" onClick={toggle} aria-label="Open sidebar">
          ☰
        </button>
        <a className="icon new" href="#/" aria-label="New conversation">
          +
        </a>
      </aside>
    );
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <a className="wordmark" href="#/">
          <b>ARA</b>
          <span>research agent</span>
        </a>
        <button className="icon" onClick={toggle} aria-label="Close sidebar">
          ⟨
        </button>
      </div>
      <a className="new-chat" href="#/">
        <span>+</span> New conversation
      </a>
      <nav className="history">
        {conversations.length === 0 && <p className="notice small">Your conversations will appear here.</p>}
        {[...groups].map(([label, items]) => (
          <section key={label}>
            <h4>{label}</h4>
            {items.map((c) => (
              <div key={c.id} className={`history-item ${c.id === current && page === "chat" ? "current" : ""}`}>
                {editing === c.id ? (
                  <input
                    autoFocus
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onBlur={() => void save(c.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") void save(c.id);
                      if (e.key === "Escape") setEditing(null);
                    }}
                  />
                ) : (
                  <a href={`#/c/${c.id}`} title={c.title}>
                    {(c.running || running.has(c.id)) && <span className="spinner" aria-label="Running" />}
                    <span className="history-title">{c.title}</span>
                  </a>
                )}
                <span className="history-actions">
                  <button
                    aria-label="Rename"
                    onClick={() => {
                      setEditing(c.id);
                      setDraft(c.title);
                    }}
                  >
                    ✎
                  </button>
                  <button aria-label="Delete" onClick={() => void remove(c)}>
                    ✕
                  </button>
                </span>
              </div>
            ))}
          </section>
        ))}
      </nav>
      <footer className="sidebar-foot">
        <a href="#/evaluation" aria-current={page === "evaluation" ? "page" : undefined}>
          Evaluation
        </a>
        <a href="#/memory" aria-current={page === "memory" ? "page" : undefined}>
          Memory
        </a>
      </footer>
    </aside>
  );
}
