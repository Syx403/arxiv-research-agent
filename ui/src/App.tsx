import { useEffect, useState } from "react";

import { EvidencePage } from "./pages/Evidence";
import { EvaluationPage } from "./pages/Evaluation";
import { MemoryPage } from "./pages/Memory";
import { WorkflowPage } from "./pages/Workflow";
import { useTurn } from "./turn";

const PAGES = [
  ["workflow", "Workflow"],
  ["evidence", "Evidence"],
  ["evaluation", "Evaluation"],
  ["memory", "Memory"],
] as const;
type Page = (typeof PAGES)[number][0];

function current(): Page {
  const hash = location.hash.replace("#/", "");
  return PAGES.some(([id]) => id === hash) ? (hash as Page) : "workflow";
}

export function App() {
  const [page, setPage] = useState<Page>(current);
  const turn = useTurn();

  useEffect(() => {
    const follow = () => setPage(current());
    window.addEventListener("hashchange", follow);
    return () => window.removeEventListener("hashchange", follow);
  }, []);

  return (
    <div className="app">
      <header className="masthead">
        <a className="wordmark" href="#/workflow">
          <b>ARA</b>
          <span>arXiv research agent</span>
        </a>
        <nav className="nav">
          {PAGES.map(([id, label]) => (
            <a key={id} href={`#/${id}`} aria-current={page === id ? "page" : undefined}>
              {label}
            </a>
          ))}
        </nav>
      </header>
      <main key={page}>
        {page === "workflow" && <WorkflowPage turn={turn} />}
        {page === "evidence" && <EvidencePage result={turn.result} />}
        {page === "evaluation" && <EvaluationPage />}
        {page === "memory" && <MemoryPage />}
      </main>
    </div>
  );
}
