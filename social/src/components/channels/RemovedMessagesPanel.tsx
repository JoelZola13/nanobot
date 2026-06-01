"use client";

import { useCallback, useEffect, useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { AlertCircle, Loader2, RefreshCw, ShieldAlert, X } from "lucide-react";
import MarkdownContent from "./MarkdownContent";
import { apiUrl } from "@/lib/apiUrl";

type RemovedMessage = {
  id: string;
  content: string;
  createdAt: string;
  deletedAt: string | null;
  author: {
    id: string;
    displayName: string;
    isAgent: boolean;
  };
  removedBy: {
    id: string;
    displayName: string;
  } | null;
  removalMode: "author" | "moderator" | null;
  reason: string | null;
};

type RemovedMessagesResponse = {
  deletedMessages: RemovedMessage[];
};

async function getApiErrorMessage(response: Response, fallback: string) {
  const payload = (await response.json().catch(() => null)) as {
    error?: unknown;
    message?: unknown;
  } | null;

  if (typeof payload?.error === "string" && payload.error.trim()) {
    return payload.error;
  }

  if (typeof payload?.message === "string" && payload.message.trim()) {
    return payload.message;
  }

  return fallback;
}

export default function RemovedMessagesPanel({
  channelId,
  onClose,
}: {
  channelId: string;
  onClose: () => void;
}) {
  const [messages, setMessages] = useState<RemovedMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadMessages = useCallback(
    async (signal?: AbortSignal) => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(
          apiUrl(`/api/channels/${channelId}/messages/deleted`),
          {
            cache: "no-store",
            signal,
          },
        );
        if (!response.ok) {
          throw new Error(
            await getApiErrorMessage(
              response,
              "Removed messages could not load.",
            ),
          );
        }

        const data = (await response.json()) as RemovedMessagesResponse;
        if (signal?.aborted) return;
        setMessages(data.deletedMessages || []);
      } catch (error) {
        if (signal?.aborted) return;
        setMessages([]);
        setError(
          error instanceof Error
            ? error.message
            : "Removed messages could not load.",
        );
      } finally {
        if (!signal?.aborted) setLoading(false);
      }
    },
    [channelId],
  );

  useEffect(() => {
    const controller = new AbortController();
    void loadMessages(controller.signal);

    return () => {
      controller.abort();
    };
  }, [loadMessages]);

  return (
    <div
      data-testid="removed-messages-panel"
      role="dialog"
      aria-label="Removed messages"
      className="absolute right-4 top-14 z-40 w-[28rem] max-w-[calc(100vw-2rem)] overflow-hidden rounded-xl border border-border bg-bg-surface shadow-xl"
    >
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex min-w-0 items-center gap-2">
          <ShieldAlert size={14} className="text-accent" />
          <span className="font-heading text-sm font-semibold text-text-primary">
            Removed messages
          </span>
          <span
            className="rounded-full bg-bg-elevated px-1.5 py-0.5 text-2xs font-medium text-text-muted"
            aria-label={`${messages.length} removed messages`}
          >
            {messages.length}
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-text-muted hover:bg-bg-hover"
          aria-label="Close removed messages"
        >
          <X size={14} />
        </button>
      </div>

      <div className="max-h-[28rem] overflow-y-auto">
        {loading && (
          <div
            className="flex items-center justify-center gap-2 px-4 py-6 text-sm text-text-muted"
            role="status"
            aria-label="Loading removed messages"
          >
            <Loader2 size={14} className="animate-spin" />
            Loading
          </div>
        )}
        {!loading && error && (
          <div
            className="mx-4 my-4 rounded-lg border border-red-300 bg-red-50 px-3 py-3 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
            data-testid="removed-messages-load-error"
            role="alert"
          >
            <div className="flex items-start gap-2">
              <AlertCircle size={15} className="mt-0.5 shrink-0" />
              <span className="min-w-0 flex-1">{error}</span>
            </div>
            <button
              type="button"
              onClick={() => void loadMessages()}
              className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-red-300 bg-white px-2.5 py-1.5 text-xs font-medium text-red-700 hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-red-400 dark:border-red-800 dark:bg-red-950/20 dark:text-red-100 dark:hover:bg-red-950/50"
              aria-label="Retry removed messages"
            >
              <RefreshCw size={12} />
              Retry
            </button>
          </div>
        )}
        {!loading && !error && messages.length === 0 && (
          <div className="px-4 py-6 text-center text-sm text-text-muted">
            No removed messages
          </div>
        )}
        {!loading &&
          !error &&
          messages.map((message) => (
            <div
              key={message.id}
              data-testid="removed-message-row"
              className="border-b border-border/50 px-4 py-3 last:border-0"
            >
              <div className="mb-1 flex min-w-0 items-center justify-between gap-2">
                <span
                  className={`truncate text-xs font-medium ${
                    message.author.isAgent ? "text-teal" : "text-text-primary"
                  }`}
                >
                  {message.author.displayName}
                </span>
                {message.deletedAt && (
                  <span className="shrink-0 text-2xs text-text-muted">
                    {formatDistanceToNow(new Date(message.deletedAt), {
                      addSuffix: true,
                    })}
                  </span>
                )}
              </div>
              <div className="line-clamp-3 rounded-lg border border-border bg-bg-base px-3 py-2 text-sm text-text-primary/80">
                <MarkdownContent content={message.content} />
              </div>
              <div className="mt-2 space-y-1 text-2xs text-text-muted">
                <div>
                  Removed by{" "}
                  <span className="font-medium text-text-secondary">
                    {message.removedBy?.displayName || "Unknown user"}
                  </span>
                  {message.removalMode === "moderator" && " as moderator"}
                </div>
                {message.reason && (
                  <div>
                    Reason:{" "}
                    <span className="text-text-secondary">
                      {message.reason}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}
