"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Bookmark, Bot, Hash, MessageSquare, X } from "lucide-react";
import MarkdownContent from "@/components/channels/MarkdownContent";
import JumpToMessageLink from "@/components/messages/JumpToMessageLink";
import RelativeTime from "@/components/messages/RelativeTime";
import ProfilePopover from "@/components/users/ProfilePopover";
import { apiUrl } from "@/lib/apiUrl";
import { getJumpToMessageLabel } from "@/lib/messageLinks";
import { useEmbeddedNavigation } from "@/lib/useEmbeddedNavigation";
import type { SavedItemResult } from "@/lib/savedItems";

export default function SavedItemsView({
  initialSavedItems,
}: {
  initialSavedItems: SavedItemResult[];
}) {
  const [savedItems, setSavedItems] = useState(initialSavedItems);
  const [pendingMessageId, setPendingMessageId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<{
    messageId: string;
    message: string;
  } | null>(null);
  const [isInteractive, setIsInteractive] = useState(false);
  const { withEmbed } = useEmbeddedNavigation();

  useEffect(() => {
    const interactiveTimer = window.setTimeout(
      () => setIsInteractive(true),
      120,
    );
    return () => window.clearTimeout(interactiveTimer);
  }, []);

  const handleUnsave = async (messageId: string) => {
    if (pendingMessageId) return;
    setPendingMessageId(messageId);
    setActionError(null);
    try {
      const res = await fetch(apiUrl("/api/saved"), {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messageId }),
      });

      if (!res.ok) {
        const payload = (await res.json().catch(() => null)) as {
          error?: unknown;
          message?: unknown;
        } | null;
        const errorMessage =
          typeof payload?.error === "string" && payload.error.trim()
            ? payload.error
            : typeof payload?.message === "string" && payload.message.trim()
              ? payload.message
              : "Saved message could not be removed.";
        throw new Error(errorMessage);
      }

      setSavedItems((items) =>
        items.filter((item) => item.messageId !== messageId),
      );
    } catch (error) {
      setActionError({
        messageId,
        message:
          error instanceof Error
            ? error.message
            : "Saved message could not be removed.",
      });
    } finally {
      setPendingMessageId(null);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl px-6 py-5">
        <div className="mb-4 flex items-center justify-between gap-4">
          <div className="min-w-0">
            <h2 className="font-heading text-xl font-semibold text-text-primary">
              Later
            </h2>
            <p className="text-sm text-text-muted">
              {savedItems.length} saved message
              {savedItems.length === 1 ? "" : "s"}
            </p>
          </div>
        </div>

        {savedItems.length === 0 ? (
          <div
            className="flex min-h-[22rem] items-center justify-center rounded-lg border border-border bg-bg-surface px-6 text-center"
            data-testid="saved-items-empty"
          >
            <div className="max-w-sm">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-lg border border-border bg-bg-elevated text-accent">
                <Bookmark size={22} />
              </div>
              <h3 className="font-heading text-lg font-semibold text-text-primary">
                Nothing saved yet
              </h3>
              <p className="mt-2 text-sm leading-6 text-text-muted">
                Save important messages from channels or DMs and they will stay
                here until you remove them.
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {savedItems.map((savedItem) => (
              <SavedItemCard
                key={savedItem.id}
                savedItem={savedItem}
                removing={pendingMessageId === savedItem.messageId}
                interactive={isInteractive}
                error={
                  actionError?.messageId === savedItem.messageId
                    ? actionError.message
                    : null
                }
                onUnsave={handleUnsave}
                onDismissError={() => setActionError(null)}
                withEmbed={withEmbed}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function SavedItemCard({
  savedItem,
  removing,
  interactive,
  error,
  onUnsave,
  onDismissError,
  withEmbed,
}: {
  savedItem: SavedItemResult;
  removing: boolean;
  interactive: boolean;
  error: string | null;
  onUnsave: (messageId: string) => void;
  onDismissError: () => void;
  withEmbed: (href: string) => string;
}) {
  const isDm = savedItem.channelType === "DM";
  const itemHref = withEmbed(savedItem.href);

  return (
    <article
      className="rounded-lg border border-border bg-bg-surface px-4 py-3"
      data-testid="saved-item-card"
    >
      <div className="mb-2 flex min-w-0 flex-wrap items-center justify-between gap-2 text-xs text-text-muted">
        <div className="flex min-w-0 items-center gap-2">
          <Link
            href={itemHref}
            className="flex min-w-0 items-center gap-2 rounded-md hover:text-accent"
            data-testid="saved-item-channel-link"
            aria-label={`Open saved conversation ${savedItem.channelLabel}`}
          >
            <span className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md border border-border bg-bg-elevated text-text-secondary">
              {isDm ? <MessageSquare size={13} /> : <Hash size={13} />}
            </span>
            <span className="truncate font-medium text-text-secondary">
              {savedItem.channelLabel}
            </span>
          </Link>
          <span>/</span>
          <span className="shrink-0">
            saved <RelativeTime value={savedItem.savedAt} />
          </span>
        </div>
        <JumpToMessageLink
          href={itemHref}
          label={getJumpToMessageLabel("saved")}
          channelLabel={savedItem.channelLabel}
        />
      </div>

      <div className="flex gap-3">
        <ProfilePopover
          user={savedItem.author}
          className="mt-0.5 shrink-0"
          triggerClassName="rounded-full focus:outline-none focus:ring-2 focus:ring-accent/40"
        >
          <span
            className={`avatar h-9 w-9 text-sm ${
              savedItem.author.isAgent
                ? "bg-teal-muted text-teal"
                : "bg-accent-muted text-accent"
            }`}
          >
            {savedItem.author.avatarUrl ? (
              <img
                src={savedItem.author.avatarUrl}
                alt=""
                className="h-full w-full rounded-full object-cover"
              />
            ) : savedItem.author.isAgent ? (
              <Bot size={16} />
            ) : (
              savedItem.author.displayName[0]?.toUpperCase()
            )}
          </span>
        </ProfilePopover>

        <div className="min-w-0 flex-1">
          <div className="mb-1 flex min-w-0 items-baseline gap-2">
            <ProfilePopover
              user={savedItem.author}
              triggerClassName={`rounded-sm text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-accent/40 ${
                savedItem.author.isAgent ? "text-teal" : "text-text-primary"
              }`}
            >
              {savedItem.author.displayName}
            </ProfilePopover>
            {savedItem.author.isAgent && (
              <span className="badge-teal">agent</span>
            )}
            <RelativeTime
              value={savedItem.createdAt}
              className="truncate text-2xs text-text-muted"
            />
          </div>
          <div className="text-sm leading-6 text-text-primary/90">
            <MarkdownContent content={savedItem.content} />
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => onUnsave(savedItem.messageId)}
              disabled={!interactive || removing}
              className="inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-danger hover:text-danger disabled:opacity-60"
              aria-label={`Remove saved message from ${savedItem.channelLabel}`}
            >
              <X size={12} />
              {removing ? "Removing..." : "Remove"}
            </button>
          </div>
          {error && (
            <div
              className="mt-3 flex items-start justify-between gap-3 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
              role="status"
              data-testid="saved-item-action-error"
            >
              <span className="min-w-0 flex-1">{error}</span>
              <button
                type="button"
                onClick={onDismissError}
                className="rounded-md p-0.5 text-red-700 hover:bg-red-100 dark:text-red-200 dark:hover:bg-red-900/30"
                aria-label={`Dismiss saved item error for ${savedItem.channelLabel}`}
              >
                <X size={13} />
              </button>
            </div>
          )}
        </div>
      </div>
    </article>
  );
}
