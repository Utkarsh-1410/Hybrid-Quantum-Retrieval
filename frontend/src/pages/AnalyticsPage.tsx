import { useMemo, useState } from "react";
import { getIndexJob, indexDocument } from "../lib/api";
import type { ActivityRecord, IndexJob } from "../types";
import { ChartIcon, DatabaseIcon, SparkIcon } from "../components/Icons";

export function AnalyticsPage({
  activity,
  backendOnline,
}: {
  activity: ActivityRecord[];
  backendOnline: boolean | null;
}) {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [source, setSource] = useState("manual");
  const [canonicalUrl, setCanonicalUrl] = useState("");
  const [job, setJob] = useState<IndexJob | null>(null);
  const [indexing, setIndexing] = useState(false);
  const [error, setError] = useState("");

  const metrics = useMemo(() => {
    const searches = activity.filter((item) => item.type === "search");
    const chats = activity.filter((item) => item.type === "chat");
    const allScores = activity.flatMap((item) => item.scores);
    const average = (values: number[]) =>
      values.length
        ? values.reduce((sum, value) => sum + value, 0) / values.length
        : 0;
    return {
      searches: searches.length,
      chats: chats.length,
      latency: average(activity.map((item) => item.latencyMs)),
      confidence: average(
        chats
          .map((item) => item.confidence)
          .filter((value): value is number => value !== undefined),
      ),
      bm25: average(allScores.map((score) => score.bm25)),
      dense: average(allScores.map((score) => score.dense)),
      quantum: average(allScores.map((score) => score.quantum)),
      final: average(allScores.map((score) => score.final)),
    };
  }, [activity]);

  async function submitDocument(event: React.FormEvent) {
    event.preventDefault();
    setIndexing(true);
    setError("");
    setJob(null);
    try {
      const submitted = await indexDocument({
        title,
        content,
        source,
        canonicalUrl: canonicalUrl || undefined,
      });
      let current = await getIndexJob(submitted.job_id);
      setJob(current);
      while (current.status === "pending" || current.status === "running") {
        await new Promise((resolve) => setTimeout(resolve, 1200));
        current = await getIndexJob(submitted.job_id);
        setJob(current);
      }
      if (current.status === "completed") {
        setTitle("");
        setContent("");
        setCanonicalUrl("");
      }
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Indexing failed",
      );
    } finally {
      setIndexing(false);
    }
  }

  return (
    <div>
      <div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
        <div>
          <p className="text-xs font-medium tracking-[0.18em] text-amber-300 uppercase">
            System telemetry
          </p>
          <h1 className="mt-2 text-4xl font-semibold tracking-tight text-white">
            Retrieval analytics
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">
            Local interaction metrics, ranking-score trends, API status, and
            incremental document ingestion.
          </p>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-white/8 px-3 py-2 text-xs text-slate-400">
          <span
            className={`h-2 w-2 rounded-full ${
              backendOnline ? "bg-emerald-400" : "bg-rose-400"
            }`}
          />
          Backend {backendOnline ? "operational" : "unavailable"}
        </div>
      </div>

      <div className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Metric
          label="Searches"
          value={String(metrics.searches)}
          detail={`${metrics.chats} AI conversations`}
          icon={<ChartIcon />}
        />
        <Metric
          label="Average latency"
          value={`${Math.round(metrics.latency)} ms`}
          detail="Across browser activity"
          icon={<SparkIcon />}
        />
        <Metric
          label="Answer confidence"
          value={`${Math.round(metrics.confidence * 100)}%`}
          detail="Citation-aware heuristic"
          icon={<SparkIcon />}
        />
        <Metric
          label="Final rank score"
          value={metrics.final.toFixed(3)}
          detail="Mean retrieved result"
          icon={<DatabaseIcon />}
        />
      </div>

      <div className="mt-8 grid gap-6 xl:grid-cols-[1.05fr_.95fr]">
        <section className="rounded-2xl border border-white/8 bg-[#0b1722] p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-medium tracking-[0.15em] text-slate-500 uppercase">
                Score composition
              </p>
              <h2 className="mt-2 text-xl font-semibold text-white">
                Average retrieval signals
              </h2>
            </div>
            <span className="font-mono text-xs text-slate-600">
              {activity.length} events
            </span>
          </div>
          <div className="mt-8 space-y-6">
            <AnalyticsBar
              label="BM25 lexical"
              value={metrics.bm25}
              color="bg-amber-300"
            />
            <AnalyticsBar
              label="Dense semantic"
              value={metrics.dense}
              color="bg-cyan-300"
            />
            <AnalyticsBar
              label="Quantum probability"
              value={metrics.quantum}
              color="bg-violet-400"
            />
            <AnalyticsBar
              label="Final score"
              value={metrics.final}
              color="bg-emerald-400"
            />
          </div>

          <div className="mt-9 border-t border-white/8 pt-6">
            <p className="text-xs font-medium tracking-[0.15em] text-slate-500 uppercase">
              Recent activity
            </p>
            <div className="mt-4 space-y-3">
              {activity.slice(0, 5).map((item) => (
                <div
                  key={item.id}
                  className="flex items-center gap-3 rounded-xl bg-white/[0.025] px-4 py-3"
                >
                  <span
                    className={`h-2 w-2 rounded-full ${
                      item.type === "search"
                        ? "bg-cyan-300"
                        : "bg-violet-400"
                    }`}
                  />
                  <p className="min-w-0 flex-1 truncate text-sm text-slate-300">
                    {item.query}
                  </p>
                  <span className="font-mono text-xs text-slate-600">
                    {item.latencyMs}ms
                  </span>
                </div>
              ))}
              {!activity.length && (
                <p className="py-8 text-center text-sm text-slate-600">
                  Search and chat activity will appear here.
                </p>
              )}
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-white/8 bg-[#0b1722] p-6">
          <div className="flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-xl bg-amber-300/10 text-amber-300">
              <DatabaseIcon />
            </span>
            <div>
              <p className="text-xs font-medium tracking-[0.15em] text-amber-300 uppercase">
                Incremental indexing
              </p>
              <h2 className="mt-1 text-xl font-semibold text-white">
                Add a document
              </h2>
            </div>
          </div>

          <form onSubmit={submitDocument} className="mt-6 space-y-4">
            <Field label="Title">
              <input
                required
                value={title}
                onChange={(event) => setTitle(event.target.value)}
                placeholder="Paper or document title"
                className="input"
              />
            </Field>
            <Field label="Content">
              <textarea
                required
                value={content}
                onChange={(event) => setContent(event.target.value)}
                placeholder="Paste the abstract or document text"
                rows={7}
                className="input resize-y py-3"
              />
            </Field>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Source">
                <input
                  value={source}
                  onChange={(event) => setSource(event.target.value)}
                  className="input"
                />
              </Field>
              <Field label="Canonical URL">
                <input
                  type="url"
                  value={canonicalUrl}
                  onChange={(event) => setCanonicalUrl(event.target.value)}
                  placeholder="https://..."
                  className="input"
                />
              </Field>
            </div>
            <button
              disabled={indexing}
              className="h-12 w-full rounded-xl bg-amber-300 font-semibold text-[#071019] transition hover:bg-amber-200 disabled:opacity-50"
            >
              {indexing ? "Indexing document..." : "Submit to index"}
            </button>
          </form>

          {error && (
            <p className="mt-4 rounded-xl border border-rose-400/20 bg-rose-400/8 p-3 text-sm text-rose-200">
              {error}
            </p>
          )}

          {job && (
            <div className="mt-5 rounded-xl border border-white/8 bg-[#071019]/70 p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-white">
                  Job {job.job_id.slice(0, 8)}
                </span>
                <span
                  className={`rounded-full px-2.5 py-1 text-xs ${
                    job.status === "completed"
                      ? "bg-emerald-400/10 text-emerald-300"
                      : job.status === "failed"
                        ? "bg-rose-400/10 text-rose-300"
                        : "bg-cyan-300/10 text-cyan-200"
                  }`}
                >
                  {job.status}
                </span>
              </div>
              <div className="mt-4 grid grid-cols-3 gap-3 text-center">
                <JobStat label="Indexed" value={job.indexed_count} />
                <JobStat label="Skipped" value={job.skipped_count} />
                <JobStat label="Failed" value={job.failed_count} />
              </div>
              {job.error_message && (
                <p className="mt-3 text-xs text-rose-300">
                  {job.error_message}
                </p>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  detail,
  icon,
}: {
  label: string;
  value: string;
  detail: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-white/8 bg-[#0b1722] p-5">
      <div className="flex items-center justify-between text-slate-500">
        <span className="text-xs font-medium tracking-[0.12em] uppercase">
          {label}
        </span>
        {icon}
      </div>
      <p className="mt-5 text-3xl font-semibold tracking-tight text-white">
        {value}
      </p>
      <p className="mt-2 text-xs text-slate-500">{detail}</p>
    </div>
  );
}

function AnalyticsBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div>
      <div className="mb-2 flex justify-between text-sm">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-slate-200">{value.toFixed(3)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-white/6">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${Math.max(2, value * 100)}%` }}
        />
      </div>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-xs font-medium text-slate-400">
        {label}
      </span>
      {children}
    </label>
  );
}

function JobStat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg bg-white/[0.035] p-3">
      <p className="font-mono text-lg text-white">{value}</p>
      <p className="mt-1 text-[10px] tracking-wider text-slate-600 uppercase">
        {label}
      </p>
    </div>
  );
}
