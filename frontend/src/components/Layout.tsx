import type { ReactNode } from "react";
import {
  ChartIcon,
  ChatIcon,
  DatabaseIcon,
  SearchIcon,
  SparkIcon,
} from "./Icons";

export type Page = "search" | "chat" | "analytics";

const navigation = [
  { id: "search" as const, label: "Search", icon: SearchIcon },
  { id: "chat" as const, label: "AI Chat", icon: ChatIcon },
  { id: "analytics" as const, label: "Analytics", icon: ChartIcon },
];

export function Layout({
  page,
  onNavigate,
  backendOnline,
  children,
}: {
  page: Page;
  onNavigate: (page: Page) => void;
  backendOnline: boolean | null;
  children: ReactNode;
}) {
  return (
    <div className="min-h-screen bg-[#071019] text-slate-100">
      <header className="sticky top-0 z-40 border-b border-white/8 bg-[#071019]/92 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between px-5 py-4 lg:px-8">
          <button
            className="flex items-center gap-3 text-left"
            onClick={() => onNavigate("search")}
          >
            <span className="grid h-10 w-10 place-items-center rounded-xl bg-cyan-300 text-[#071019] shadow-[0_0_32px_rgba(103,232,249,.18)]">
              <SparkIcon />
            </span>
            <span>
              <span className="block text-sm font-semibold tracking-[0.18em] text-cyan-200 uppercase">
                Q-Find
              </span>
              <span className="hidden text-xs text-slate-500 sm:block">
                Hybrid research intelligence
              </span>
            </span>
          </button>

          <nav className="flex items-center rounded-xl border border-white/8 bg-white/[0.035] p-1">
            {navigation.map((item) => {
              const Icon = item.icon;
              const active = page === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition sm:px-4 ${
                    active
                      ? "bg-white/10 text-white"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{item.label}</span>
                </button>
              );
            })}
          </nav>

          <div className="hidden items-center gap-2 text-xs text-slate-400 md:flex">
            <DatabaseIcon className="h-4 w-4" />
            <span
              className={`h-2 w-2 rounded-full ${
                backendOnline
                  ? "bg-emerald-400"
                  : backendOnline === false
                    ? "bg-rose-400"
                    : "bg-amber-300"
              }`}
            />
            {backendOnline
              ? "API connected"
              : backendOnline === false
                ? "API offline"
                : "Checking API"}
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-[1500px] px-5 py-8 lg:px-8 lg:py-12">
        {children}
      </main>
    </div>
  );
}
