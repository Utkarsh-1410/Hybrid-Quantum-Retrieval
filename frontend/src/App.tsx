import { useEffect, useState } from "react";
import { Layout, type Page } from "./components/Layout";
import { loadActivity, saveActivity } from "./lib/activity";
import { getHealth } from "./lib/api";
import { AnalyticsPage } from "./pages/AnalyticsPage";
import { ChatPage } from "./pages/ChatPage";
import { SearchPage } from "./pages/SearchPage";
import type {
  ActivityRecord,
  AskResponse,
  SearchResponse,
} from "./types";

export function App() {
  const [page, setPage] = useState<Page>("search");
  const [activity, setActivity] = useState<ActivityRecord[]>(loadActivity);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  useEffect(() => {
    getHealth()
      .then(setBackendOnline)
      .catch(() => setBackendOnline(false));
  }, []);

  function addActivity(record: ActivityRecord) {
    setActivity((current) => {
      const next = [record, ...current].slice(0, 100);
      saveActivity(next);
      return next;
    });
  }

  function recordSearch(response: SearchResponse) {
    addActivity({
      id: response.request_id,
      type: "search",
      query: response.query,
      timestamp: new Date().toISOString(),
      resultCount: response.total,
      latencyMs: response.latency_ms,
      scores: response.results.map((result) => result.scores),
    });
  }

  function recordChat(query: string, response: AskResponse) {
    addActivity({
      id: response.request_id,
      type: "chat",
      query,
      timestamp: new Date().toISOString(),
      resultCount: response.sources.length,
      latencyMs: response.latency_ms,
      confidence: response.confidence,
      scores: response.sources.map((source) => source.scores),
    });
  }

  return (
    <Layout
      page={page}
      onNavigate={setPage}
      backendOnline={backendOnline}
    >
      {page === "search" && <SearchPage onComplete={recordSearch} />}
      {page === "chat" && <ChatPage onComplete={recordChat} />}
      {page === "analytics" && (
        <AnalyticsPage
          activity={activity}
          backendOnline={backendOnline}
        />
      )}
    </Layout>
  );
}
