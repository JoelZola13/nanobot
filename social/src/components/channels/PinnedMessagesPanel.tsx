"use client";

import { useState, useEffect, useCallback } from "react";
import { AlertCircle, Loader2, Pin, RefreshCw, X } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import MarkdownContent from "./MarkdownContent";
import { apiUrl } from "@/lib/apiUrl";

interface PinnedMessage {
  id: string;
  channelId: string;
  content: string;
  createdAt: string;
  author: { id: string; displayName: string; isAgent: boolean };
}

interface PinnedMessagesPanelProps {
  channelId: string;
  onClose: () => void;
}

async function apiErrorMessage(res: Response, fallback: string) {
  const data = (await res.json().catch(() => null)) as {
    error?: string;
  } | null;
  return data?.error || fallback;
}

export default function PinnedMessagesPanel({
  channelId,
  onClose,
}: PinnedMessagesPanelProps) {
  const [pins, setPins] = useState<PinnedMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<{
    messageId: string;
    text: string;
  } | null>(null);
  const [unpinningId, setUnpinningId] = useState<string | null>(null);

  const loadPins = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    setActionError(null);
    try {
      const response = await fetch(apiUrl(`/api/channels/${channelId}/pins`), {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(
          await apiErrorMessage(response, "Pinned messages could not load."),
        );
      }
      const data = (await response.json()) as { pins?: PinnedMessage[] };
      setPins(data.pins || []);
    } catch (error) {
      setPins([]);
      setLoadError(
        error instanceof Error
          ? error.message
          : "Pinned messages could not load.",
      );
    } finally {
      setLoading(false);
    }
  }, [channelId]);

  useEffect(() => {
    void loadPins();
  }, [loadPins]);

  const handleUnpin = async (messageId: string) => {
    if (unpinningId) return;
    setActionError(null);
    setUnpinningId(messageId);
    try {
      const response = await fetch(apiUrl(`/api/channels/${channelId}/pins`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messageId }),
      });
      if (!response.ok) {
        throw new Error(
          await apiErrorMessage(
            response,
            "Pinned message could not be removed.",
          ),
        );
      }
      const data = (await response.json().catch(() => null)) as {
        isPinned?: boolean;
      } | null;
      if (data?.isPinned === true) {
        throw new Error("Pinned message is still pinned.");
      }
      setPins((prev) => prev.filter((p) => p.id !== messageId));
    } catch (error) {
      setActionError({
        messageId,
        text:
          error instanceof Error
            ? error.message
            : "Pinned message could not be removed.",
      });
    } finally {
      setUnpinningId(null);
    }
  };

  return (
    <div
      data-testid="pinned-messages-panel"
      role="dialog"
      aria-label="Pinned messages"
      className="absolute right-4 top-14 z-40 w-80 bg-bg-surface border border-border rounded-xl shadow-xl overflow-hidden"
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <Pin size={14} className="text-accent" />
          <span className="font-heading font-semibold text-sm text-text-primary">
            Pinned Messages
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="p-1 rounded hover:bg-bg-hover text-text-muted"
          title="Close pinned messages"
          aria-label="Close pinned messages"
        >
          <X size={14} />
        </button>
      </div>

      <div className="max-h-80 overflow-y-auto">
        {loading && (
          <div
            className="flex items-center justify-center gap-2 px-4 py-6 text-center text-sm text-text-muted"
            role="status"
          >
            <Loader2 size={15} className="animate-spin" />
            Loading pinned messages...
          </div>
        )}
        {!loading && !loadError && pins.length === 0 && (
          <div className="px-4 py-6 text-center text-sm text-text-muted">
            No pinned messages
          </div>
        )}
        {!loading && loadError && (
          <div
            data-testid="pinned-messages-load-error"
            className="m-3 rounded-lg border border-red-300 bg-red-50 px-3 py-3 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
          >
            <div className="flex items-start gap-2">
              <AlertCircle size={15} className="mt-0.5 shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="font-medium">
                  Pinned messages could not load
                </div>
                <div className="mt-0.5 text-xs opacity-90">{loadError}</div>
              </div>
            </div>
            <button
              type="button"
              onClick={() => void loadPins()}
              className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-red-300 bg-white px-2 py-1 text-xs font-medium text-red-700 hover:bg-red-50 dark:border-red-900/70 dark:bg-red-950/40 dark:text-red-100 dark:hover:bg-red-900/30"
              aria-label="Retry pinned messages"
            >
              <RefreshCw size={12} />
              Retry pinned messages
            </button>
          </div>
        )}
        {!loading &&
          !loadError &&
          pins.map((pin) => {
            const unpinning = unpinningId === pin.id;
            const rowError =
              actionError?.messageId === pin.id ? actionError.text : null;

            return (
              <div
                key={pin.id}
                data-testid="pinned-message-row"
                className="px-4 py-3 border-b border-border/50 last:border-0 hover:bg-bg-hover/50 transition-colors group focus-within:bg-bg-hover/50"
              >
                <div className="flex items-center justify-between mb-1">
                  <span
                    className={`text-xs font-medium ${pin.author.isAgent ? "text-teal" : "text-text-primary"}`}
                  >
                    {pin.author.displayName}
                  </span>
                  <div className="flex items-center gap-1">
                    <span className="text-2xs text-text-muted">
                      {formatDistanceToNow(new Date(pin.createdAt), {
                        addSuffix: true,
                      })}
                    </span>
                    <button
                      type="button"
                      onClick={() => void handleUnpin(pin.id)}
                      disabled={Boolean(unpinningId)}
                      className="opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus:opacity-100 p-0.5 rounded hover:bg-bg-elevated text-text-muted hover:text-danger transition-all disabled:cursor-wait disabled:opacity-60"
                      title="Unpin"
                      aria-label={`Unpin pinned message from ${pin.author.displayName}`}
                    >
                      {unpinning ? (
                        <Loader2 size={10} className="animate-spin" />
                      ) : (
                        <X size={10} />
                      )}
                    </button>
                  </div>
                </div>
                <div className="text-sm text-text-primary/80 line-clamp-3">
                  <MarkdownContent content={pin.content} />
                </div>
                {rowError && (
                  <div
                    data-testid="pinned-message-action-error"
                    className="mt-2 flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 px-2.5 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
                  >
                    <AlertCircle size={13} className="mt-0.5 shrink-0" />
                    <span className="min-w-0 flex-1">{rowError}</span>
                    <button
                      type="button"
                      onClick={() => setActionError(null)}
                      className="rounded p-0.5 hover:bg-red-100 dark:hover:bg-red-900/40"
                      aria-label="Dismiss pinned message error"
                    >
                      <X size={12} />
                    </button>
                  </div>
                )}
              </div>
            );
          })}
      </div>
    </div>
  );
}
