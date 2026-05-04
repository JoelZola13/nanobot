"use client";

import { useEffect, useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { ShieldAlert, X } from "lucide-react";
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

  useEffect(() => {
    let cancelled = false;

    setLoading(true);
    setError(null);

    fetch(apiUrl(`/api/channels/${channelId}/messages/deleted`), {
      cache: "no-store",
    })
      .then((response) => {
        if (!response.ok) throw new Error("Failed to load removed messages");
        return response.json() as Promise<RemovedMessagesResponse>;
      })
      .then((data) => {
        if (!cancelled) setMessages(data.deletedMessages || []);
      })
      .catch(() => {
        if (!cancelled) setError("Removed messages could not load.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [channelId]);

  return (
    <div className="absolute right-4 top-14 z-40 w-[28rem] max-w-[calc(100vw-2rem)] overflow-hidden rounded-xl border border-border bg-bg-surface shadow-xl">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex min-w-0 items-center gap-2">
          <ShieldAlert size={14} className="text-accent" />
          <span className="font-heading text-sm font-semibold text-text-primary">
            Removed messages
          </span>
          <span className="rounded-full bg-bg-elevated px-1.5 py-0.5 text-2xs font-medium text-text-muted">
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
          <div className="px-4 py-6 text-center text-sm text-text-muted">
            Loading...
          </div>
        )}
        {!loading && error && (
          <div className="px-4 py-6 text-center text-sm text-danger">
            {error}
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
