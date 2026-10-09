import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { createPortal } from "react-dom";

import { api } from "../api";
import type { Fact, LibraryPaper, Page, Shelved } from "../api";

const SHOWN = 8; // items a list shows as plain rows; past that it gets a frame and scrolls

/** What ARA remembers about you, quoted from you, and the papers you have read. Both lists are
 *  built for many items: plain rows while short, a framed list that scrolls and loads more at its
 *  end once long; new items are added at the end (D38). */
export function MemoryPage() {
  const [query, setQuery] = useState("");
  const [paper, setPaper] = useState<string | null>(null);
  const facts = usePaged<Fact>(useCallback((offset) => api.facts(offset), []));
  const shelf = usePaged<Shelved>(useCallback((offset) => api.library(query, offset), [query]));

  const forget = async (key: string) => {
    await api.forget(key);
    facts.reload();
  };

  return (
    <>
      <div className="eyebrow">Memory</div>
      <h1>What ARA keeps between conversations.</h1>
      <p className="lede">
        Facts are kept only in your own words, with the conversation they came from. Forgetting one
        removes it from every later search.
      </p>
      <hr className="rule" />
      <div className="memory">
        <section>
          <Heading title="About you" total={facts.total} />
          {facts.error && <div className="error-box">{facts.error}</div>}
          {facts.total === 0 && !facts.loading && (
            <p className="notice">Nothing yet. Tell ARA, for example, “I only use hosted APIs.”</p>
          )}
          <Shelf paged={facts}>
            {facts.items.map((f) => (
              <article key={f.key} className="fact">
                <blockquote>“{f.quote}”</blockquote>
                <button className="button quiet small" onClick={() => void forget(f.key)}>
                  Forget
                </button>
                <p>{f.statement}</p>
                <span className="meta">
                  {f.day}
                  {f.conversation ? (
                    <a href={`#/c/${f.conversation.id}`}>from “{f.conversation.title}”</a>
                  ) : (
                    <span>from a conversation no longer kept</span>
                  )}
                </span>
              </article>
            ))}
          </Shelf>
        </section>
        <section>
          <Heading title="Library" total={shelf.total} />
          {(shelf.total > SHOWN || query) && (
            <input
              className="search"
              type="search"
              placeholder="Search by title or arXiv id"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          )}
          {shelf.error && <div className="error-box">{shelf.error}</div>}
          {shelf.total === 0 && !shelf.loading && (
            <p className="notice">{query ? "No paper matches." : "No papers read yet."}</p>
          )}
          <Shelf paged={shelf}>
            {shelf.items.map((p) => (
              <button
                key={p.arxiv_id}
                className={`shelf-row ${paper === p.arxiv_id ? "open" : ""}`}
                onClick={() => setPaper(p.arxiv_id)}
              >
                <span className="shelf-title">{p.title}</span>
                <span className="shelf-meta">
                  <span className="mono">
                    {p.arxiv_id}v{p.version}
                  </span>
                  {p.published && <span>{p.published}</span>}
                </span>
              </button>
            ))}
          </Shelf>
        </section>
      </div>
      {paper && <PaperDrawer arxivId={paper} close={() => setPaper(null)} />}
    </>
  );
}

interface Paged<T> {
  items: T[];
  total: number;
  loading: boolean;
  error: string | null;
  more: () => void;
  reload: () => void;
}

/** A list read a page at a time; `load` changing (a new search) starts it again. */
function usePaged<T>(load: (offset: number) => Promise<Page<T>>): Paged<T> {
  const [items, setItems] = useState<T[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [round, setRound] = useState(0);

  useEffect(() => {
    let live = true;
    setLoading(true);
    load(0)
      .then((page) => {
        if (!live) return;
        setItems(page.items);
        setTotal(page.total);
        setError(null);
      })
      .catch((e: unknown) => live && setError(String(e)))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [load, round]);

  const more = () => {
    if (loading || items.length >= total) return;
    setLoading(true);
    load(items.length)
      .then((page) => {
        setItems((before) => [...before, ...page.items]);
        setTotal(page.total);
      })
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setLoading(false));
  };

  return { items, total, loading, error, more, reload: () => setRound((r) => r + 1) };
}

function Heading({ title, total }: { title: string; total: number }) {
  return (
    <div className="shelf-head">
      <h2>{title}</h2>
      {total > 0 && <span className="count">{total}</span>}
    </div>
  );
}

/** Plain rows while the list is short; past SHOWN items a framed list that scrolls, reading the
 *  next page as its end comes into view. */
function Shelf<T>({ paged, children }: { paged: Paged<T>; children: ReactNode }) {
  const full = paged.total > SHOWN;
  const rest = paged.total - paged.items.length;
  return (
    <div
      className={`shelf ${full ? "full" : ""}`}
      onScroll={(e) => {
        const el = e.currentTarget;
        if (el.scrollHeight - el.scrollTop - el.clientHeight < 80) paged.more();
      }}
    >
      {children}
      {rest > 0 && (
        <button className="shelf-more" onClick={paged.more} disabled={paged.loading}>
          {paged.loading ? "Loading…" : `Show ${Math.min(rest, 50)} more of ${rest}`}
        </button>
      )}
    </div>
  );
}

/** One paper of the library: what it is, when it was read, and where — each conversation that
 *  read it opens at that conversation. */
function PaperDrawer({ arxivId, close }: { arxivId: string; close: () => void }) {
  const [paper, setPaper] = useState<LibraryPaper | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setPaper(null);
    api
      .libraryPaper(arxivId)
      .then(setPaper)
      .catch((e: unknown) => setError(String(e)));
  }, [arxivId]);

  useEffect(() => {
    const key = (e: KeyboardEvent) => e.key === "Escape" && close();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [close]);

  const day = (iso: string) => new Date(iso).toLocaleDateString(undefined, { dateStyle: "medium" });

  // into <body>: the page's entrance animation transforms it, which would hold a fixed drawer
  return createPortal(
    <>
      <div className="drawer-veil" onClick={close} />
      <aside className="drawer" aria-label="Paper">
        <header className="drawer-head">
          <span className="eyebrow">From your library</span>
          <button className="icon" onClick={close} aria-label="Close">
            ✕
          </button>
        </header>
        {error && <div className="error-box">{error}</div>}
        {!paper && !error && <p className="notice">Loading…</p>}
        {paper && (
          <div className="drawer-body">
            <h2>{paper.title}</h2>
            <div className="meta">
              <span className="mono">
                arXiv {paper.arxiv_id}v{paper.version}
              </span>
              {paper.published && <span>published {paper.published}</span>}
              <span>first read {day(paper.first_read_at)}</span>
            </div>
            <div className="drawer-links">
              <a className="button" href={`https://arxiv.org/abs/${paper.arxiv_id}v${paper.version}`} target="_blank" rel="noreferrer">
                Open on arXiv ↗
              </a>
              <a className="button quiet" href={`https://arxiv.org/pdf/${paper.arxiv_id}v${paper.version}`} target="_blank" rel="noreferrer">
                PDF ↗
              </a>
            </div>
            <h3>Abstract</h3>
            <p className="abstract">{paper.abstract}</p>
            <h3>Read in</h3>
            {paper.read_in.length === 0 ? (
              <p className="notice">Read before conversations were kept (or in an evaluation).</p>
            ) : (
              <ul className="read-in">
                {paper.read_in.map((r) => (
                  <li key={r.turn_id}>
                    <a href={`#/c/${r.conversation_id}`}>
                      <b>{r.title}</b>
                      <span>{r.message}</span>
                      <small>{day(r.finished_at)}</small>
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </aside>
    </>,
    document.body,
  );
}
