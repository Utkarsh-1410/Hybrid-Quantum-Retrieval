import type {
  AskResponse,
  IndexJob,
  IndexResponse,
  SearchResponse,
} from "../types";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";
const ROOT_BASE = API_BASE.replace(/\/api\/v1\/?$/, "");

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const detail =
      payload?.detail ?? `Request failed with status ${response.status}`;
    throw new Error(
      typeof detail === "string" ? detail : JSON.stringify(detail),
    );
  }
  return response.json() as Promise<T>;
}

export async function searchDocuments(input: {
  query: string;
  topK: number;
  source?: string;
}): Promise<SearchResponse> {
  return request<SearchResponse>("/search", {
    method: "POST",
    body: JSON.stringify({
      query: input.query,
      top_k: input.topK,
      filters: input.source ? { source: input.source } : {},
    }),
  });
}

export async function askQuestion(input: {
  query: string;
  topK: number;
}): Promise<AskResponse> {
  return request<AskResponse>("/ask", {
    method: "POST",
    body: JSON.stringify({
      query: input.query,
      top_k: input.topK,
      filters: {},
    }),
  });
}

export async function indexDocument(input: {
  title: string;
  content: string;
  source?: string;
  canonicalUrl?: string;
}): Promise<IndexResponse> {
  return request<IndexResponse>("/index", {
    method: "POST",
    body: JSON.stringify({
      documents: [
        {
          title: input.title,
          content: input.content,
          source: input.source || null,
          canonical_url: input.canonicalUrl || null,
          metadata: {},
        },
      ],
    }),
  });
}

export async function getIndexJob(jobId: string): Promise<IndexJob> {
  return request<IndexJob>(`/index/${jobId}`);
}

export async function getHealth(): Promise<boolean> {
  const response = await fetch(`${ROOT_BASE}/health`);
  return response.ok;
}
