import type { SearchResult } from "../types";
import { ExternalIcon } from "./Icons";
import { ScoreBar } from "./ScoreBar";

export function ResultCard({ result }: { result: SearchResult }) {
  return (
    <article className="group rounded-2xl border border-white/8 bg-[#0b1722] p-5 transition hover:-translate-y-0.5 hover:border-cyan-300/25">
      <div className="flex gap-4">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-white/10 bg-white/5 font-mono text-xs text-cyan-200">
          {String(result.rank).padStart(2, "0")}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="text-lg font-semibold text-white">
                {result.title}
              </h3>
              <p className="mt-2 line-clamp-3 text-sm leading-6 text-slate-400">
                {result.snippet}
              </p>
            </div>
            <div className="rounded-lg bg-cyan-300/10 px-2.5 py-1 font-mono text-xs text-cyan-200">
              {result.scores.final.toFixed(3)}
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <ScoreBar
              label="BM25"
              value={result.scores.bm25}
              color="bg-amber-300"
            />
            <ScoreBar
              label="Dense"
              value={result.scores.dense}
              color="bg-cyan-300"
            />
            <ScoreBar
              label="Quantum"
              value={result.scores.quantum}
              color="bg-violet-400"
            />
            <ScoreBar
              label="Final"
              value={result.scores.final}
              color="bg-emerald-400"
            />
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-white/7 pt-4 text-xs text-slate-500">
            {String(result.metadata.source ?? "") && (
              <span className="rounded-full border border-white/8 px-2.5 py-1">
                {String(result.metadata.source)}
              </span>
            )}
            {String(result.metadata.primary_category ?? "") && (
              <span className="rounded-full border border-white/8 px-2.5 py-1">
                {String(result.metadata.primary_category)}
              </span>
            )}
            {result.url && (
              <a
                href={result.url}
                target="_blank"
                rel="noreferrer"
                className="ml-auto flex items-center gap-1.5 text-cyan-300 hover:text-cyan-200"
              >
                Open source <ExternalIcon />
              </a>
            )}
          </div>
        </div>
      </div>
    </article>
  );
}
