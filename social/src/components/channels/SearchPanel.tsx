"use client";

import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import {
  AlertCircle,
  Loader2,
  Paperclip,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { formatDistanceToNow } from "date-fns";
import MarkdownContent from "./MarkdownContent";
import { apiUrl } from "@/lib/apiUrl";
import { useEmbeddedNavigation } from "@/lib/useEmbeddedNavigation";

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
  channel: {
    id: string;
    name: string | null;
    slug: string | null;
    type: string;
  };
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

async function apiErrorMessage(res: Response, fallback: string) {
  const data = (await res.json().catch(() => null)) as {
    error?: string;
  } | null;
  return data?.error || fallback;
}

export default function SearchPanel({ onClose }: SearchPanelProps) {
  const [query, setQuery] = useState("");
  const [channelId, setChannelId] = useState("");
  const [authorId, setAuthorId] = useState("");
  const [dateRange, setDateRange] = useState("");
  const [attachmentType, setAttachmentType] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [loading, setLoading] = useState(false);
  const [filtersLoading, setFiltersLoading] = useState(false);
  const [filterError, setFilterError] = useState<string | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  const { withEmbed } = useEmbeddedNavigation();
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

  const loadFilters = useCallback(async () => {
    setFiltersLoading(true);
    setFilterError(null);
    try {
      const response = await fetch(apiUrl("/api/search"), {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(
          await apiErrorMessage(response, "Search filters could not load."),
        );
      }
      const data = (await response.json()) as {
        filters?: typeof EMPTY_FILTERS;
      };
      setFilters(data.filters || EMPTY_FILTERS);
    } catch (error) {
      setFilters(EMPTY_FILTERS);
      setFilterError(
        error instanceof Error
          ? error.message
          : "Search filters could not load.",
      );
    } finally {
      setFiltersLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadFilters();
  }, [loadFilters]);

  const buildSearchParams = useCallback(() => {
    const params = new URLSearchParams();
    if (trimmedQuery.length >= 2) params.set("q", trimmedQuery);
    if (channelId) params.set("channelId", channelId);
    if (authorId) params.set("authorId", authorId);
    if (dateRange) params.set("date", dateRange);
    if (attachmentType) params.set("attachment", attachmentType);
    return params;
  }, [attachmentType, authorId, channelId, dateRange, trimmedQuery]);

  const executeSearch = useCallback(async () => {
    const params = buildSearchParams();
    setLoading(true);
    setSearchError(null);
    try {
      const res = await fetch(apiUrl(`/api/search?${params.toString()}`), {
        cache: "no-store",
      });
      if (!res.ok) {
        throw new Error(
          await apiErrorMessage(res, "Messages search could not load."),
        );
      }
      const data = (await res.json()) as {
        results?: SearchResult[];
        filters?: typeof EMPTY_FILTERS;
      };
      setResults(data.results || []);
      if (data.filters) setFilters(data.filters);
    } catch (error) {
      setResults([]);
      setSearchError(
        error instanceof Error
          ? error.message
          : "Messages search could not load.",
      );
    } finally {
      setLoading(false);
    }
  }, [buildSearchParams]);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);

    if (!hasSearchCriteria) {
      setResults([]);
      setLoading(false);
      setSearchError(null);
      return;
    }

    debounceRef.current = setTimeout(() => {
      void executeSearch();
    }, 300);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [executeSearch, hasSearchCriteria]);

  const filteredResultCount = useMemo(() => results.length, [results.length]);

  const clearSearch = () => {
    setQuery("");
    setResults([]);
    setSearchError(null);
    clearFilters();
    window.requestAnimationFrame(() => inputRef.current?.focus());
  };

  const clearFilters = () => {
    setChannelId("");
    setAuthorId("");
    setDateRange("");
    setAttachmentType("");
    setFilterError(null);
  };

  const handleChange = (value: string) => {
    setQuery(value);
    setSearchError(null);
  };

  const handleClickResult = (result: SearchResult) => {
    const basePath = result.channel.type === "DM" ? "dm" : "channels";
    router.push(
      withEmbed(`/${basePath}/${result.channelId}?message=${result.id}`),
    );
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[12vh]"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative w-full max-w-3xl overflow-hidden rounded-xl border border-border bg-bg-surface shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label="Search messages"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 border-b border-border px-4 py-3">
          <Search size={18} className="shrink-0 text-text-muted" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => handleChange(e.target.value)}
            placeholder="Search messages..."
            aria-label="Search message text"
            className="flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-muted focus:outline-none"
          />
          {hasSearchCriteria && (
            <span className="rounded-full bg-bg-elevated px-2 py-1 text-2xs font-medium text-text-muted">
              {filteredResultCount}
            </span>
          )}
          {hasSearchCriteria && (
            <button
              type="button"
              onClick={clearSearch}
              className="rounded-md px-2 py-1 text-xs font-medium text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
              aria-label="Clear message search"
            >
              Clear
            </button>
          )}
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-text-muted hover:bg-bg-hover"
            title="Close search"
            aria-label="Close search"
          >
            <X size={16} />
          </button>
        </div>

        <div className="border-b border-border px-4 py-3">
          {filterError && (
            <div
              data-testid="message-search-filter-error"
              className="mb-3 flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
            >
              <AlertCircle size={14} className="mt-0.5 shrink-0" />
              <span className="min-w-0 flex-1">{filterError}</span>
              <button
                type="button"
                onClick={() => void loadFilters()}
                className="inline-flex shrink-0 items-center gap-1 rounded border border-red-300 bg-white px-1.5 py-0.5 text-2xs font-medium hover:bg-red-50 dark:border-red-900/70 dark:bg-red-950/40 dark:hover:bg-red-900/30"
                aria-label="Retry search filters"
              >
                <RefreshCw size={10} />
                Retry
              </button>
            </div>
          )}
          <div className="grid grid-cols-2 gap-2 lg:grid-cols-[1.3fr_1fr_0.9fr_1fr_auto]">
            <select
              value={channelId}
              onChange={(e) => setChannelId(e.target.value)}
              disabled={filtersLoading}
              aria-label="Filter by channel"
              className="h-9 min-w-0 rounded-md border border-border bg-bg-elevated px-2 text-xs text-text-secondary focus:border-accent focus:outline-none disabled:cursor-wait disabled:opacity-60"
            >
              <option value="">
                {filtersLoading ? "Loading channels..." : "All channels"}
              </option>
              {filters.channels.map((channel) => (
                <option key={channel.id} value={channel.id}>
                  {channel.label}
                </option>
              ))}
            </select>

            <select
              value={authorId}
              onChange={(e) => setAuthorId(e.target.value)}
              disabled={filtersLoading}
              aria-label="Filter by sender"
              className="h-9 min-w-0 rounded-md border border-border bg-bg-elevated px-2 text-xs text-text-secondary focus:border-accent focus:outline-none disabled:cursor-wait disabled:opacity-60"
            >
              <option value="">
                {filtersLoading ? "Loading senders..." : "Any sender"}
              </option>
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
              aria-label="Clear search filters"
              className="h-9 rounded-md border border-border px-3 text-xs font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-50 lg:w-auto"
            >
              Clear
            </button>
          </div>
        </div>

        <div className="max-h-[52vh] overflow-y-auto">
          {loading && (
            <div
              data-testid="message-search-loading"
              className="flex items-center justify-center gap-2 px-4 py-6 text-center text-sm text-text-muted"
              role="status"
            >
              <Loader2 size={15} className="animate-spin" />
              Searching...
            </div>
          )}
          {!loading && searchError && (
            <div
              data-testid="message-search-error"
              className="m-4 rounded-lg border border-red-300 bg-red-50 px-3 py-3 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
            >
              <div className="flex items-start gap-2">
                <AlertCircle size={15} className="mt-0.5 shrink-0" />
                <div className="min-w-0 flex-1">
                  <div className="font-medium">Search could not load</div>
                  <div className="mt-0.5 text-xs opacity-90">{searchError}</div>
                </div>
              </div>
              <button
                type="button"
                onClick={() => void executeSearch()}
                className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-red-300 bg-white px-2 py-1 text-xs font-medium text-red-700 hover:bg-red-50 dark:border-red-900/70 dark:bg-red-950/40 dark:text-red-100 dark:hover:bg-red-900/30"
                aria-label="Retry message search"
              >
                <RefreshCw size={12} />
                Retry search
              </button>
            </div>
          )}
          {!loading &&
            !searchError &&
            hasSearchCriteria &&
            results.length === 0 && (
              <div
                className="px-4 py-6 text-center text-sm text-text-muted"
                data-testid="message-search-empty"
              >
                <div>No results found</div>
                <button
                  type="button"
                  onClick={clearSearch}
                  className="btn-secondary mt-3 text-xs"
                  aria-label="Clear message search from empty state"
                  data-testid="message-search-empty-clear"
                >
                  Clear search
                </button>
              </div>
            )}
          {!loading &&
            !searchError &&
            results.map((result) => (
              <button
                type="button"
                key={result.id}
                onClick={() => handleClickResult(result)}
                aria-label={`Open search result from ${result.author.displayName}`}
                className="w-full border-b border-border/50 px-4 py-3 text-left transition-colors last:border-0 hover:bg-bg-hover"
              >
                <div className="mb-1 flex min-w-0 items-center gap-2">
                  <span className="shrink-0 text-2xs font-medium text-accent">
                    {result.channel.type === "DM"
                      ? "DM"
                      : `#${result.channel.name || result.channel.slug}`}
                  </span>
                  <span className="text-2xs text-text-muted">/</span>
                  <span
                    className={`truncate text-2xs font-medium ${result.author.isAgent ? "text-teal" : "text-text-secondary"}`}
                  >
                    {result.author.displayName}
                  </span>
                  <span className="ml-auto shrink-0 text-2xs text-text-muted">
                    {formatDistanceToNow(new Date(result.createdAt), {
                      addSuffix: true,
                    })}
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
          {!loading && !searchError && !hasSearchCriteria && (
            <div className="px-4 py-6 text-center text-sm text-text-muted">
              Type at least two characters or choose a filter to search.
            </div>
          )}
        </div>

        <div className="border-t border-border px-4 py-2 text-2xs text-text-muted">
          Press{" "}
          <kbd className="rounded border border-border bg-bg-elevated px-1.5 py-0.5 font-mono">
            Esc
          </kbd>{" "}
          to close
        </div>
      </div>
    </div>
  );
}
