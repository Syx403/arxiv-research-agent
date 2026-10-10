// The server's API (ara/api/server.py). A turn is a POST whose body is a Server-Sent Event stream,
// read here with fetch so the message can be sent in the request body.

export type Role = "user" | "assistant";
export interface Message {
  role: Role;
  text: string;
}

export interface Paper {
  arxiv_id: string;
  version: number;
  title: string;
  abstract: string;
  published: string;
  similarity: number;
  relevance: number | null;
  reason: string;
  violated: string[];
  named: string | null;
  read_before: boolean;
}

export interface Evidence {
  id: string;
  paper_id: string;
  chunk_id: number;
  paragraph: number;
  heading_path: string;
  text: string;
}

export interface Claim {
  index: number;
  text: string;
  citations: string[];
  paragraph: number; // 0 for answers recorded before D37
}

export interface Answer {
  question: string;
  short: string;
  abstained: boolean;
  withheld?: boolean; // a direct answer failed verification; its verified lines stand (D40)
  sentences: Claim[];
  dropped: Claim[];
  checked: number;
  rejected: Claim[];
  evidence: Evidence[];
  context: string;
}

/** One piece of a reply (D39): the UI renders listings and answers from the turn's data. */
export interface Part {
  kind: "text" | "context" | "warning" | "listing" | "answer" | "note";
  text: string;
  source?: "arxiv" | "library";
}

export interface Waiting {
  kind: string;
  question: string;
}

/** One recorded turn of a conversation: what was said, what came back, and how (D36). */
export interface Turn {
  turn_id: string;
  message: string;
  reply: string;
  status: string;
  intent: string | null;
  answer: Answer | null;
  papers: Paper[];
  read: Paper[];
  problems: string[];
  trace: NodeEvent[];
  waiting: Waiting | null;
  parts: Part[]; // empty for turns recorded before D39
  calls: Call[];
  started_at: string;
  finished_at: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  running?: boolean;
}

/** The turn a conversation is running on the server (D38). */
export interface Running {
  turn_id: string;
  message: string;
  kind: "messages" | "resume";
}

export interface Conversation extends Omit<ConversationSummary, "running"> {
  turns: Turn[];
  running: Running | null;
}

export type Phase = "start" | "end" | "failed" | "degraded";
export interface NodeEvent {
  graph: string;
  node: string;
  phase: Phase;
}

export interface Call {
  id: number;
  stage: string;
  model: string;
  status: string;
  latency_ms: number | null;
  input_tokens: number | null;
  cached_tokens: number | null;
  output_tokens: number | null;
  reasoning_tokens: number | null;
  cost_usd: number | null;
}

export interface Stopped extends Running {
  spent_usd: number;
  conversation_removed: boolean;
  error?: string; // set when the turn failed and was undone (D39)
}

export type StreamEvent =
  | { event: "started"; data: Running & { started_at: string } }
  | { event: "node"; data: NodeEvent }
  | { event: "call"; data: Call }
  | { event: "problem"; data: { text: string } }
  | { event: "done"; data: Turn }
  | { event: "stopped"; data: Stopped }
  | { event: "error"; data: Stopped & { error: string } };

export interface Passage {
  chunk_id: number;
  heading_path: string;
  text: string;
  sentences: number[];
}

export interface PaperDocument {
  id: string;
  title: string;
  source: string;
  passages: Passage[];
}

export interface Run {
  id: string;
  suite: string;
  status: string;
  created_at: string;
  finished_at: string | null;
  langsmith_experiment: string | null;
  items: number;
  cost_usd: number;
}

export interface Metric {
  split: string;
  arm: string;
  metric: string;
  n: number;
  mean: number;
  low: number;
  high: number;
}

export interface RunSummary {
  run: Run & { config: Record<string, unknown> & { code?: string } };
  prompts: string[]; // "synthesize@1a2b3c4d": the prompt files its model calls carried
  metrics: Metric[];
  items: {
    item_id: string;
    arm: string;
    split: string;
    metrics: Record<string, number>;
    output: unknown;
    error: string | null;
  }[];
  usage: {
    stage: string;
    model: string;
    requests: number;
    input_tokens: number;
    cached_tokens: number;
    output_tokens: number;
    usd: number;
    p50_ms: number;
    p95_ms: number;
  }[];
}

export interface Fact {
  key: string;
  quote: string;
  statement: string;
  thread: string;
  turn: number;
  day: string;
  conversation: { id: string; title: string } | null;
}

export interface Shelved {
  arxiv_id: string;
  version: number;
  title: string;
  published: string | null;
  first_read_at: string;
}

export interface LibraryPaper extends Shelved {
  abstract: string;
  last_read_at: string;
  read_in: { conversation_id: string; title: string; turn_id: string; message: string; finished_at: string }[];
}

export interface Page<T> {
  items: T[];
  total: number;
}

export interface SuiteInfo {
  id: string;
  name: string;
  covers: string;
  question: string;
  data: string;
  grading: string;
  headline: string;
  arm: string;
  metrics: Record<string, string>;
  items: { dev: number; test: number };
}

export interface Catalog {
  suites: SuiteInfo[];
  protocol: string[];
  prompts: Record<string, string>; // today's version of every prompt file
}

export type Graphs = Record<"main" | "discover" | "read" | "answer", string>;

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return (await response.json()) as T;
}

export const api = {
  graph: () => json<Graphs>("/api/graph"),
  conversations: () => json<ConversationSummary[]>("/api/conversations"),
  newConversation: () => json<{ id: string }>("/api/conversations", { method: "POST" }),
  conversation: (id: string) => json<Conversation>(`/api/conversations/${id}`),
  rename: (id: string, title: string) =>
    json<{ title: string }>(`/api/conversations/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),
  remove: (id: string) => json<{ deleted: string }>(`/api/conversations/${id}`, { method: "DELETE" }),
  document: (paperId: string) =>
    json<PaperDocument>(`/api/papers/${encodeURIComponent(paperId)}/document`),
  stop: (id: string) => json<Stopped>(`/api/conversations/${id}/stop`, { method: "POST" }),
  runs: () => json<Run[]>("/api/evals"),
  run: (id: string) => json<RunSummary>(`/api/evals/${id}`),
  suites: () => json<Catalog>("/api/evals/suites"),
  facts: (offset: number) => json<Page<Fact>>(`/api/memory/facts?offset=${offset}`),
  library: (query: string, offset: number) =>
    json<Page<Shelved>>(`/api/memory/library?q=${encodeURIComponent(query)}&offset=${offset}`),
  libraryPaper: (arxivId: string) =>
    json<LibraryPaper>(`/api/memory/library/${encodeURIComponent(arxivId)}`),
  forget: (key: string) =>
    json<{ forgotten: string }>(`/api/memory/${encodeURIComponent(key)}`, { method: "DELETE" }),
};

/** A turn's events as they arrive: sending a message, answering a question, or rejoining the
 *  turn a conversation is running ("live"). The turn runs on the server either way (D38). */
export async function* stream(
  conversation: string,
  kind: "messages" | "resume" | "live",
  text?: string,
): AsyncGenerator<StreamEvent> {
  const body = kind === "messages" ? { text } : { answer: text };
  const response = await fetch(
    `/api/conversations/${conversation}/${kind}`,
    kind === "live"
      ? undefined
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
  );
  if (!response.ok || !response.body) throw new Error(`${response.status} ${await response.text()}`);
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const event = /^event: (.*)$/m.exec(block)?.[1] ?? "message";
      const data = /^data: (.*)$/m.exec(block)?.[1];
      if (data !== undefined) yield { event, data: JSON.parse(data) } as StreamEvent;
    }
  }
}
