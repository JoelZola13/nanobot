"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Search, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { formatDistanceToNow } from "date-fns";
import MarkdownContent from "./MarkdownContent";

interface SearchResult {
  id: string;
  channelId: string;
  content: string;
  createdAt: string;
  author: { id: string; displayName: string; isAgent: boolean };
  channel: { id: string; name: string | null; slug: string | null; type: string };
}

interface SearchPanelProps {
  onClose: () => void;
}

export default function SearchPanel({ onClose }: SearchPanelProps) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    inputRef.current?.focus();
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  const doSearch = useCallback(async (q: string) => {
    if (q.length < 2) { setResults([]); return; }
    setLoading(true);
    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      setResults(data.results || []);
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleChange = (value: string) => {
    setQuery(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => doSearch(value), 300);
  };

  const handleClickResult = (result: SearchResult) => {
    const path = result.channel.type === "DM"
      ? `/dm/${result.channelId}`
      : `/channels/${result.channel.slug || result.channelId}`;
    router.push(path);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]" onClick={onClose}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative w-full max-w-xl bg-bg-surface border border-border rounded-xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
          <Search size={18} className="text-text-muted shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => handleChange(e.target.value)}
            placeholder="Search messages..."
            className="flex-1 bg-transparent text-sm text-text-primary placeholder-text-muted focus:outline-none"
          />
          <button onClick={onClose} className="p-1 rounded hover:bg-bg-hover text-text-muted">
            <X size={16} />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-[50vh] overflow-y-auto">
          {loading && (
            <div className="px-4 py-6 text-center text-sm text-text-muted">Searching...</div>
          )}
          {!loading && query.length >= 2 && results.length === 0 && (
            <div className="px-4 py-6 text-center text-sm text-text-muted">No results found</div>
          )}
          {!loading && results.map((result) => (
            <button
              key={result.id}
              onClick={() => handleClickResult(result)}
              className="w-full text-left px-4 py-3 hover:bg-bg-hover transition-colors border-b border-border/50 last:border-0"
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-2xs font-medium text-accent">
                  {result.channel.type === "DM" ? "DM" : `#${result.channel.name || result.channel.slug}`}
                </span>
                <span className="text-2xs text-text-muted">·</span>
                <span className={`text-2xs font-medium ${result.author.isAgent ? "text-teal" : "text-text-secondary"}`}>
                  {result.author.displayName}
                </span>
                <span className="text-2xs text-text-muted ml-auto">
                  {formatDistanceToNow(new Date(result.createdAt), { addSuffix: true })}
                </span>
              </div>
              <div className="text-sm text-text-primary/80 line-clamp-2">
                <MarkdownContent content={result.content} />
              </div>
            </button>
          ))}
        </div>

        {/* Footer hint */}
        <div className="px-4 py-2 border-t border-border text-2xs text-text-muted">
          Press <kbd className="px-1.5 py-0.5 rounded bg-bg-elevated border border-border font-mono">Esc</kbd> to close
        </div>
      </div>
    </div>
  );
}
