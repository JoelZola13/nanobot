"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import { Paperclip, Search, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { formatDistanceToNow } from "date-fns";
import MarkdownContent from "./MarkdownContent";
import { apiUrl } from "@/lib/apiUrl";

type FilterOption = {
  id: string;
  label: string;
  username?: string;
  isAgent?: boolean;
};

interface SearchResult {
  id: string;
  channelId: string;
  content: string;
  createdAt: string;
  author: { id: string; displayName: string; isAgent: boolean };
  channel: { id: string; name: string | null; slug: string | null; type: string };
  attachments: {
    id: string;
    fileName: string;
    mimeType: string;
    url: string;
  }[];
}

interface SearchPanelProps {
  onClose: () => void;
}

const EMPTY_FILTERS: {
  channels: FilterOption[];
  authors: FilterOption[];
} = {
  channels: [],
  authors: [],
};

export default function SearchPanel({ onClose }: SearchPanelProps) {
  const [query, setQuery] = useState("");
  const [channelId, setChannelId] = useState("");
  const [authorId, setAuthorId] = useState("");
  const [dateRange, setDateRange] = useState("");
  const [attachmentType, setAttachmentType] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  const trimmedQuery = query.trim();
  const hasActiveFilters = Boolean(
    channelId || authorId || dateRange || attachmentType,
  );
  const hasSearchCriteria = trimmedQuery.length >= 2 || hasActiveFilters;

  useEffect(() => {
    inputRef.current?.focus();
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  useEffect(() => {
    fetch(apiUrl("/api/search"))
      .then((response) => response.json())
      .then((data: { filters?: typeof EMPTY_FILTERS }) => {
        setFilters(data.filters || EMPTY_FILTERS);
      })
      .catch(() => setFilters(EMPTY_FILTERS));
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);

    if (!hasSearchCriteria) {
      setResults([]);
      setLoading(false);
      return;
    }

    const params = new URLSearchParams();
    if (trimmedQuery.length >= 2) params.set("q", trimmedQuery);
    if (channelId) params.set("channelId", channelId);
    if (authorId) params.set("authorId", authorId);
    if (dateRange) params.set("date", dateRange);
    if (attachmentType) params.set("attachment", attachmentType);

    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await fetch(apiUrl(`/api/search?${params.toString()}`));
        const data = await res.json();
        setResults(data.results || []);
        if (data.filters) setFilters(data.filters);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [attachmentType, authorId, channelId, dateRange, hasSearchCriteria, trimmedQuery]);

  const filteredResultCount = useMemo(() => results.length, [results.length]);

  const clearFilters = () => {
    setChannelId("");
    setAuthorId("");
    setDateRange("");
    setAttachmentType("");
  };

  const handleChange = (value: string) => {
    setQuery(value);
  };

  const handleClickResult = (result: SearchResult) => {
    const basePath = result.channel.type === "DM" ? "dm" : "channels";
    router.push(`/${basePath}/${result.channelId}?message=${result.id}`);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[12vh]" onClick={onClose}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative w-full max-w-3xl overflow-hidden rounded-xl border border-border bg-bg-surface shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 border-b border-border px-4 py-3">
          <Search size={18} className="shrink-0 text-text-muted" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => handleChange(e.target.value)}
            placeholder="Search messages..."
            className="flex-1 bg-transparent text-sm text-text-primary placeholder-text-muted focus:outline-none"
          />
          {hasSearchCriteria && (
            <span className="rounded-full bg-bg-elevated px-2 py-1 text-2xs font-medium text-text-muted">
              {filteredResultCount}
            </span>
          )}
          <button onClick={onClose} className="rounded p-1 text-text-muted hover:bg-bg-hover">
            <X size={16} />
          </button>
        </div>

        <div className="border-b border-border px-4 py-3">
          <div className="grid grid-cols-2 gap-2 lg:grid-cols-[1.3fr_1fr_0.9fr_1fr_auto]">
            <select
              value={channelId}
              onChange={(e) => setChannelId(e.target.value)}
              aria-label="Filter by channel"
              className="h-9 min-w-0 rounded-md border border-border bg-bg-elevated px-2 text-xs text-text-secondary focus:border-accent focus:outline-none"
            >
              <option value="">All channels</option>
              {filters.channels.map((channel) => (
                <option key={channel.id} value={channel.id}>
                  {channel.label}
                </option>
              ))}
            </select>

            <select
              value={authorId}
              onChange={(e) => setAuthorId(e.target.value)}
              aria-label="Filter by sender"
              className="h-9 min-w-0 rounded-md border border-border bg-bg-elevated px-2 text-xs text-text-secondary focus:border-accent focus:outline-none"
            >
              <option value="">Any sender</option>
              {filters.authors.map((author) => (
                <option key={author.id} value={author.id}>
                  {author.label || author.username}
                </option>
              ))}
            </select>

            <select
              value={dateRange}
              onChange={(e) => setDateRange(e.target.value)}
              aria-label="Filter by date"
              className="h-9 min-w-0 rounded-md border border-border bg-bg-elevated px-2 text-xs text-text-secondary focus:border-accent focus:outline-none"
            >
              <option value="">Any time</option>
              <option value="today">Today</option>
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
            </select>

            <select
              value={attachmentType}
              onChange={(e) => setAttachmentType(e.target.value)}
              aria-label="Filter by attachment type"
              className="h-9 min-w-0 rounded-md border border-border bg-bg-elevated px-2 text-xs text-text-secondary focus:border-accent focus:outline-none"
            >
              <option value="">All message types</option>
              <option value="file">Any attachment</option>
              <option value="image">Images</option>
              <option value="document">Documents</option>
              <option value="audio">Audio</option>
              <option value="video">Video</option>
            </select>

            <button
              type="button"
              onClick={clearFilters}
              disabled={!hasActiveFilters}
              className="h-9 rounded-md border border-border px-3 text-xs font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-50 lg:w-auto"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="max-h-[52vh] overflow-y-auto">
          {loading && (
            <div className="px-4 py-6 text-center text-sm text-text-muted">Searching...</div>
          )}
          {!loading && hasSearchCriteria && results.length === 0 && (
            <div className="px-4 py-6 text-center text-sm text-text-muted">No results found</div>
          )}
          {!loading && results.map((result) => (
            <button
              key={result.id}
              onClick={() => handleClickResult(result)}
              className="w-full border-b border-border/50 px-4 py-3 text-left transition-colors last:border-0 hover:bg-bg-hover"
            >
              <div className="mb-1 flex min-w-0 items-center gap-2">
                <span className="shrink-0 text-2xs font-medium text-accent">
                  {result.channel.type === "DM" ? "DM" : `#${result.channel.name || result.channel.slug}`}
                </span>
                <span className="text-2xs text-text-muted">/</span>
                <span className={`truncate text-2xs font-medium ${result.author.isAgent ? "text-teal" : "text-text-secondary"}`}>
                  {result.author.displayName}
                </span>
                <span className="ml-auto shrink-0 text-2xs text-text-muted">
                  {formatDistanceToNow(new Date(result.createdAt), { addSuffix: true })}
                </span>
              </div>
              <div className="line-clamp-2 text-sm text-text-primary/80">
                <MarkdownContent content={result.content} />
              </div>
              {result.attachments.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {result.attachments.slice(0, 3).map((attachment) => (
                    <span
                      key={attachment.id}
                      className="inline-flex max-w-[12rem] items-center gap-1 rounded-md border border-border bg-bg-elevated px-2 py-1 text-2xs text-text-muted"
                    >
                      <Paperclip size={10} className="shrink-0" />
                      <span className="truncate">{attachment.fileName}</span>
                    </span>
                  ))}
                  {result.attachments.length > 3 && (
                    <span className="rounded-md border border-border px-2 py-1 text-2xs text-text-muted">
                      +{result.attachments.length - 3}
                    </span>
                  )}
                </div>
              )}
            </button>
          ))}
        </div>

        <div className="border-t border-border px-4 py-2 text-2xs text-text-muted">
          Press <kbd className="rounded border border-border bg-bg-elevated px-1.5 py-0.5 font-mono">Esc</kbd> to close
        </div>
      </div>
    </div>
  );
}
