import { useState } from "react";
import { searchDocuments } from "../lib/api";
import type { SearchResponse } from "../types";
import { ArrowIcon, SearchIcon } from "../components/Icons";
import { ResultCard } from "../components/ResultCard";

export function SearchPage({
  onComplete,
}: {
  onComplete: (response: SearchResponse) => void;
}) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [source, setSource] = useState("");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError("");
    try {
      const result = await searchDocuments({
        query: query.trim(),
        topK,
        source: source.trim() || undefined,
      });
      setResponse(result);
      onComplete(result);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Search failed",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <section className="grid gap-8 lg:grid-cols-[1fr_340px] lg:items-end">
        <div>
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-cyan-300/15 bg-cyan-300/[0.06] px-3 py-1.5 text-xs font-medium text-cyan-200">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-300" />
            BM25 + Dense + Quantum context
          </div>
          <h1 className="max-w-4xl text-4xl font-semibold tracking-[-0.04em] text-white sm:text-6xl">
            Search beyond keywords.
            <span className="block text-slate-500">
              Rank by meaning and context.
            </span>
          </h1>
          <p className="mt-5 max-w-2xl text-base leading-7 text-slate-400">
            Explore indexed research using lexical precision, semantic
            similarity, and Hilbert-space-inspired re-ranking.
          </p>
        </div>
        <div className="rounded-2xl border border-white/8 bg-white/[0.025] p-5">
          <p className="text-xs font-medium tracking-[0.16em] text-slate-500 uppercase">
            Ranking equation
          </p>
          <p className="mt-3 font-mono text-sm leading-7 text-slate-300">
            0.35 hybrid
            <br />+ 0.45 quantum
            <br />+ 0.20 context
          </p>
        </div>
      </section>

      <form
        onSubmit={submit}
        className="mt-10 rounded-2xl border border-white/10 bg-[#0b1722] p-3 shadow-[0_28px_80px_rgba(0,0,0,.24)]"
      >
        <div className="flex flex-col gap-3 lg:flex-row">
          <label className="flex min-w-0 flex-1 items-center gap-3 rounded-xl bg-white/[0.035] px-4">
            <SearchIcon className="h-5 w-5 text-slate-500" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search papers, methods, findings..."
              className="h-14 w-full bg-transparent text-base text-white outline-none placeholder:text-slate-600"
            />
          </label>
          <input
            value={source}
            onChange={(event) => setSource(event.target.value)}
            placeholder="Source filter"
            className="h-14 rounded-xl border border-white/8 bg-white/[0.035] px-4 text-sm outline-none placeholder:text-slate-600 focus:border-cyan-300/30 lg:w-40"
          />
          <select
            value={topK}
            onChange={(event) => setTopK(Number(event.target.value))}
            className="h-14 rounded-xl border border-white/8 bg-[#0e1b27] px-4 text-sm text-slate-300 outline-none lg:w-28"
          >
            {[5, 10, 20, 50].map((value) => (
              <option key={value} value={value}>
                Top {value}
              </option>
            ))}
          </select>
          <button
            disabled={loading || !query.trim()}
            className="flex h-14 items-center justify-center gap-2 rounded-xl bg-cyan-300 px-6 font-semibold text-[#071019] transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Searching..." : "Search"}
            {!loading && <ArrowIcon />}
          </button>
        </div>
      </form>

      {error && (
        <div className="mt-5 rounded-xl border border-rose-400/20 bg-rose-400/8 p-4 text-sm text-rose-200">
          {error}
        </div>
      )}

      {response && (
        <section className="mt-10">
          <div className="mb-5 flex items-end justify-between">
            <div>
              <p className="text-sm text-slate-500">
                {response.total} results in {response.latency_ms} ms
              </p>
              <h2 className="mt-1 text-2xl font-semibold text-white">
                Ranked evidence
              </h2>
            </div>
            <span className="hidden font-mono text-xs text-slate-600 sm:block">
              {response.request_id.slice(0, 8)}
            </span>
          </div>
          <div className="grid gap-4">
            {response.results.length ? (
              response.results.map((result) => (
                <ResultCard key={result.chunk_id} result={result} />
              ))
            ) : (
              <EmptyState text="No indexed documents matched this query." />
            )}
          </div>
        </section>
      )}
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="rounded-2xl border border-dashed border-white/10 px-6 py-16 text-center text-sm text-slate-500">
      {text}
    </div>
  );
}
