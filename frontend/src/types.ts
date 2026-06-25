export type ScoreExplanation = {
  bm25: number;
  dense: number;
  hybrid: number;
  quantum: number;
  context: number;
  final: number;
  formula: string;
};

export type SearchResult = {
  rank: number;
  document_id: string;
  chunk_id: string;
  title: string;
  snippet: string;
  url: string | null;
  metadata: Record<string, unknown>;
  scores: ScoreExplanation;
};

export type SearchResponse = {
  request_id: string;
  query: string;
  total: number;
  latency_ms: number;
  results: SearchResult[];
};

export type Citation = {
  citation_id: string;
  rank: number;
  document_id: string;
  chunk_id: string;
  title: string;
  url: string | null;
  excerpt: string;
  score: number;
};

export type AskResponse = {
  request_id: string;
  answer: string;
  citations: Citation[];
  sources: SearchResult[];
  confidence: number;
  latency_ms: number;
};

export type IndexResponse = {
  job_id: string;
  status: string;
  accepted_count: number;
};

export type IndexJob = {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  requested_count: number;
  indexed_count: number;
  skipped_count: number;
  failed_count: number;
  index_version: string | null;
  error_message: string | null;
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
};

export type ActivityRecord = {
  id: string;
  type: "search" | "chat";
  query: string;
  timestamp: string;
  resultCount: number;
  latencyMs: number;
  confidence?: number;
  scores: ScoreExplanation[];
};
