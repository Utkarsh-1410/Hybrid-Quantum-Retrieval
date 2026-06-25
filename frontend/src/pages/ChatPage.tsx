import { useState } from "react";
import { askQuestion } from "../lib/api";
import type { AskResponse } from "../types";
import { ArrowIcon, ChatIcon, SparkIcon } from "../components/Icons";

type Turn = {
  id: string;
  query: string;
  response?: AskResponse;
  error?: string;
};

export function ChatPage({
  onComplete,
}: {
  onComplete: (query: string, response: AskResponse) => void;
}) {
  const [query, setQuery] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const question = query.trim();
    if (!question || loading) return;
    const id = crypto.randomUUID();
    setTurns((current) => [...current, { id, query: question }]);
    setQuery("");
    setLoading(true);
    try {
      const response = await askQuestion({ query: question, topK: 6 });
      setTurns((current) =>
        current.map((turn) =>
          turn.id === id ? { ...turn, response } : turn,
        ),
      );
      onComplete(question, response);
    } catch (error) {
      setTurns((current) =>
        current.map((turn) =>
          turn.id === id
            ? {
                ...turn,
                error: error instanceof Error ? error.message : "Chat failed",
              }
            : turn,
        ),
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-5xl">
      <div className="mb-8 flex items-start gap-4">
        <span className="grid h-12 w-12 place-items-center rounded-xl bg-violet-400/12 text-violet-300">
          <ChatIcon />
        </span>
        <div>
          <p className="text-xs font-medium tracking-[0.18em] text-violet-300 uppercase">
            Grounded research assistant
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">
            Ask the indexed evidence
          </h1>
          <p className="mt-3 text-sm leading-6 text-slate-400">
            Answers are generated only from quantum-ranked sources. Validated
            citations and confidence are shown for every response.
          </p>
        </div>
      </div>

      <div className="min-h-[480px] space-y-6 rounded-3xl border border-white/8 bg-[#0a1520] p-5 sm:p-8">
        {!turns.length && (
          <div className="grid min-h-[390px] place-items-center text-center">
            <div>
              <SparkIcon className="mx-auto h-8 w-8 text-cyan-300" />
              <p className="mt-4 text-lg font-medium text-slate-200">
                Start with a research question
              </p>
              <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">
                Try asking for a comparison, explanation, or synthesis across
                the indexed papers.
              </p>
            </div>
          </div>
        )}

        {turns.map((turn) => (
          <div key={turn.id} className="space-y-4">
            <div className="ml-auto max-w-2xl rounded-2xl rounded-tr-sm bg-cyan-300 px-5 py-4 text-sm leading-6 text-[#071019]">
              {turn.query}
            </div>

            {turn.error && (
              <div className="max-w-3xl rounded-2xl border border-rose-400/20 bg-rose-400/8 p-5 text-sm text-rose-200">
                {turn.error}
              </div>
            )}

            {turn.response ? (
              <div className="max-w-4xl rounded-2xl rounded-tl-sm border border-white/8 bg-white/[0.035] p-5 sm:p-6">
                <div className="mb-4 flex items-center justify-between gap-4">
                  <span className="flex items-center gap-2 text-xs font-medium tracking-wider text-cyan-200 uppercase">
                    <SparkIcon className="h-4 w-4" />
                    Evidence-grounded answer
                  </span>
                  <Confidence value={turn.response.confidence} />
                </div>
                <p className="whitespace-pre-wrap text-sm leading-7 text-slate-200">
                  {turn.response.answer}
                </p>

                <div className="mt-6 border-t border-white/8 pt-5">
                  <p className="mb-3 text-xs font-medium tracking-[0.14em] text-slate-500 uppercase">
                    Validated citations
                  </p>
                  {turn.response.citations.length ? (
                    <div className="grid gap-3 sm:grid-cols-2">
                      {turn.response.citations.map((citation) => (
                        <a
                          key={citation.citation_id}
                          href={citation.url ?? undefined}
                          target={citation.url ? "_blank" : undefined}
                          rel="noreferrer"
                          className="rounded-xl border border-white/8 bg-[#071019]/60 p-4 transition hover:border-cyan-300/25"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-mono text-xs text-cyan-300">
                              [{citation.citation_id}]
                            </span>
                            <span className="font-mono text-xs text-slate-500">
                              {citation.score.toFixed(3)}
                            </span>
                          </div>
                          <p className="mt-2 text-sm font-medium text-white">
                            {citation.title}
                          </p>
                          <p className="mt-2 line-clamp-2 text-xs leading-5 text-slate-500">
                            {citation.excerpt}
                          </p>
                        </a>
                      ))}
                    </div>
                  ) : (
                    <p className="text-sm text-slate-500">
                      No citations were validated for this answer.
                    </p>
                  )}
                </div>
              </div>
            ) : (
              !turn.error && (
                <div className="flex items-center gap-3 text-sm text-slate-500">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-cyan-300" />
                  Retrieving and synthesizing evidence...
                </div>
              )
            )}
          </div>
        ))}
      </div>

      <form
        onSubmit={submit}
        className="sticky bottom-4 mt-5 flex gap-3 rounded-2xl border border-white/10 bg-[#0d1a26]/95 p-3 shadow-2xl backdrop-blur-xl"
      >
        <textarea
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
          rows={1}
          placeholder="Ask a question about the indexed research..."
          className="max-h-32 min-h-12 flex-1 resize-none bg-transparent px-3 py-3 text-sm leading-6 outline-none placeholder:text-slate-600"
        />
        <button
          disabled={loading || !query.trim()}
          className="grid h-12 w-12 shrink-0 place-items-center rounded-xl bg-cyan-300 text-[#071019] disabled:opacity-40"
          aria-label="Send question"
        >
          <ArrowIcon />
        </button>
      </form>
    </div>
  );
}

function Confidence({ value }: { value: number }) {
  const percent = Math.round(value * 100);
  return (
    <span className="rounded-full border border-emerald-400/20 bg-emerald-400/8 px-3 py-1 font-mono text-xs text-emerald-300">
      {percent}% confidence
    </span>
  );
}
