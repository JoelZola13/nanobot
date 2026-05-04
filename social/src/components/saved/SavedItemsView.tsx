"use client";

import Link from "next/link";
import { useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { Bookmark, Bot, Hash, MessageSquare, X } from "lucide-react";
import MarkdownContent from "@/components/channels/MarkdownContent";
import JumpToMessageLink from "@/components/messages/JumpToMessageLink";
import ProfilePopover from "@/components/users/ProfilePopover";
import { apiUrl } from "@/lib/apiUrl";
import { getJumpToMessageLabel } from "@/lib/messageLinks";
import type { SavedItemResult } from "@/lib/savedItems";

export default function SavedItemsView({
  initialSavedItems,
}: {
  initialSavedItems: SavedItemResult[];
}) {
  const [savedItems, setSavedItems] = useState(initialSavedItems);
  const [pendingMessageId, setPendingMessageId] = useState<string | null>(null);

  const handleUnsave = async (messageId: string) => {
    setPendingMessageId(messageId);
    try {
      const res = await fetch(apiUrl("/api/saved"), {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messageId }),
      });
      if (res.ok) {
        setSavedItems((items) => items.filter((item) => item.messageId !== messageId));
      }
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
              {savedItems.length} saved message{savedItems.length === 1 ? "" : "s"}
            </p>
          </div>
        </div>

        {savedItems.length === 0 ? (
          <div className="flex min-h-[22rem] items-center justify-center rounded-lg border border-border bg-bg-surface px-6 text-center">
            <div className="max-w-sm">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-lg border border-border bg-bg-elevated text-accent">
                <Bookmark size={22} />
              </div>
              <h3 className="font-heading text-lg font-semibold text-text-primary">
                Nothing saved yet
              </h3>
              <p className="mt-2 text-sm leading-6 text-text-muted">
                Save important messages from channels or DMs and they will stay here until you remove them.
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
                onUnsave={handleUnsave}
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
  onUnsave,
}: {
  savedItem: SavedItemResult;
  removing: boolean;
  onUnsave: (messageId: string) => void;
}) {
  const isDm = savedItem.channelType === "DM";

  return (
    <article className="rounded-lg border border-border bg-bg-surface px-4 py-3">
      <div className="mb-2 flex min-w-0 flex-wrap items-center justify-between gap-2 text-xs text-text-muted">
        <div className="flex min-w-0 items-center gap-2">
          <Link
            href={savedItem.href}
            className="flex min-w-0 items-center gap-2 rounded-md hover:text-accent"
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
            saved {formatDistanceToNow(new Date(savedItem.savedAt), { addSuffix: true })}
          </span>
        </div>
        <JumpToMessageLink
          href={savedItem.href}
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
            {savedItem.author.isAgent && <span className="badge-teal">agent</span>}
            <span className="truncate text-2xs text-text-muted">
              {formatDistanceToNow(new Date(savedItem.createdAt), { addSuffix: true })}
            </span>
          </div>
          <div className="text-sm leading-6 text-text-primary/90">
            <MarkdownContent content={savedItem.content} />
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => onUnsave(savedItem.messageId)}
              disabled={removing}
              className="inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-danger hover:text-danger disabled:opacity-60"
            >
              <X size={12} />
              Remove
            </button>
          </div>
        </div>
      </div>
    </article>
  );
}
