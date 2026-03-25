const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Issue {
  issue_id: string;
  description: string;
  status: string;
  best_approach: string;
  n_positive: number;
  n_negative: number;
  total_classified: number;
  n_flagged: number;
  frequency: number;
  n_affected_users: number;
  metrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
    training_time_ms?: number;
    inference_time_per_trace_ms?: number;
    device?: string;
  };
  created_at: string;
}

export interface IssueFlag {
  trace_id: string;
  confidence: number;
  user_query: string;
  agent_response: string;
  tools_called: string[];
  failure_type: string;
}

export interface TraceDetail {
  trace_id: string;
  user_id: string | null;
  user_query: string;
  agent_response: string;
  total_tokens: number;
  total_duration_ms: number;
  tools_called: string[];
  spans: SpanDetail[];
  metadata: Record<string, unknown>;
  timestamp: string;
}

export interface SpanDetail {
  span_id: string;
  parent_span_id: string | null;
  name: string;
  kind: string;
  status: string;
  start_time: string;
  end_time: string;
  content: string | null;
  tool_name: string | null;
  tool_input: string | null;
  tool_output: string | null;
  model: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
}

export interface TraceStats {
  total_traces: number;
  unique_users: number;
  failure_distribution: Record<string, number>;
  avg_tokens: number;
  avg_duration_ms: number;
}

export interface DiscoveryResult {
  n_traces: number;
  n_clusters: number;
  n_noise: number;
  clusters: {
    cluster_id: number;
    size: number;
    percentage: number;
    suggested_label: string;
    avg_confidence: number;
    representative_texts: string[];
  }[];
}

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error ${res.status}: ${err}`);
  }
  return res.json();
}

export const api = {
  // Issues
  createIssue: (description: string, approach = "all") =>
    request<Issue>("/api/issues/", {
      method: "POST",
      body: JSON.stringify({ description, approach, labeling_sample_size: 80 }),
    }),

  listIssues: () => request<Issue[]>("/api/issues/"),

  getIssue: (id: string) => request<Issue>(`/api/issues/${id}`),

  getIssueFlags: (id: string, limit = 50) =>
    request<IssueFlag[]>(`/api/issues/${id}/flags?limit=${limit}`),

  // Traces
  listTraces: (limit = 50, failureType?: string) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (failureType) params.set("failure_type", failureType);
    return request<TraceDetail[]>(`/api/traces/?${params}`);
  },

  getTrace: (id: string) => request<TraceDetail>(`/api/traces/${id}`),

  getTraceStats: () => request<TraceStats>("/api/traces/stats"),

  // Discovery
  runDiscovery: () => request<DiscoveryResult>("/api/discovery/run", { method: "POST" }),

  // Health
  health: () => request<{ status: string }>("/health"),
};

