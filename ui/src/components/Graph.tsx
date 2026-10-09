import mermaid from "mermaid";
import { useEffect, useId, useRef, useState } from "react";

import type { Phase } from "../api";

mermaid.initialize({
  startOnLoad: false,
  theme: "base",
  securityLevel: "strict",
  themeVariables: {
    fontFamily: '"Hanken Grotesk Variable", sans-serif',
    fontSize: "13px",
    primaryColor: "#f5f3ec",
    primaryBorderColor: "#c9c4b4",
    lineColor: "#a8a396",
    textColor: "#3d3a33",
  },
  flowchart: { curve: "basis", padding: 10, nodeSpacing: 28, rankSpacing: 34 },
});

const CLASS: Record<Phase, string> = {
  start: "is-active",
  end: "is-done",
  degraded: "is-degraded",
  failed: "is-failed",
};

/** A graph drawn by mermaid from the compiled LangGraph, its nodes coloured by this turn's events. */
export function Graph({ definition, phases }: { definition: string; phases: Record<string, Phase> }) {
  const host = useRef<HTMLDivElement>(null);
  const id = useId().replace(/:/g, "");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    mermaid
      .render(`graph-${id}`, definition)
      .then(({ svg }) => {
        if (live && host.current) host.current.innerHTML = svg;
      })
      .catch((e: unknown) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [definition, id]);

  useEffect(() => {
    const root = host.current;
    if (!root) return;
    for (const node of root.querySelectorAll<SVGGElement>("g.node")) {
      node.classList.remove(...Object.values(CLASS));
      const phase = phases[name(node)];
      if (phase) node.classList.add(CLASS[phase]);
    }
  });

  return error ? <div className="error-box">{error}</div> : <div className="graph" ref={host} />;
}

/** "graph-r1-flowchart-understand-12" → "understand" (mermaid's node element ids). */
function name(node: SVGGElement): string {
  return node.dataset.id ?? /flowchart-(.+)-\d+$/.exec(node.id)?.[1] ?? "";
}
