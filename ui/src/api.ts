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
}

export interface Answer {
  question: string;
  short: string;
  abstained: boolean;
  sentences: Claim[];
  dropped: Claim[];
  checked: number;
  rejected: Claim[];
  evidence: Evidence[];
  context: string;
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
  calls: Call[];
  started_at: string;
  finished_at: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Conversation extends ConversationSummary {
  turns: Turn[];
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

export type StreamEvent =
  | { event: "node"; data: NodeEvent }
  | { event: "call"; data: Call }
  | { event: "done"; data: Turn }
  | { event: "error"; data: { error: string } };

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
  run: Run & { config: Record<string, unknown> };
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
}

export interface Memory {
  facts: Fact[];
  library: { arxiv_id: string; version: number; title: string; published: string }[];
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
  runs: () => json<Run[]>("/api/evals"),
  run: (id: string) => json<RunSummary>(`/api/evals/${id}`),
  memory: () => json<Memory>("/api/memory"),
  forget: (key: string) =>
    json<{ forgotten: string }>(`/api/memory/${encodeURIComponent(key)}`, { method: "DELETE" }),
};

/** Send a message (or answer a waiting question) and yield the turn's events as they arrive. */
export async function* stream(
  conversation: string,
  kind: "messages" | "resume",
  body: Record<string, string>,
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`/api/conversations/${conversation}/${kind}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok || !response.body) throw new Error(`${response.status} ${response.statusText}`);
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
