"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { formatDistanceToNow } from "date-fns";
import { AtSign, Bell, Bookmark, Bot, Hash, MessageSquare } from "lucide-react";
import MarkdownContent from "@/components/channels/MarkdownContent";
import JumpToMessageLink from "@/components/messages/JumpToMessageLink";
import ProfilePopover from "@/components/users/ProfilePopover";
import { getJumpToMessageLabel } from "@/lib/messageLinks";
import type { ActivityItem } from "@/lib/activity";
import type { MentionResult } from "@/lib/mentions";
import type { SavedItemResult } from "@/lib/savedItems";

export default function ActivityView({
  username,
  items,
  mentionCount,
  savedCount,
}: {
  username: string | null;
  items: ActivityItem[];
  mentionCount: number;
  savedCount: number;
}) {
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl px-6 py-5">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-4">
          <div className="min-w-0">
            <h2 className="font-heading text-xl font-semibold text-text-primary">
              Activity
            </h2>
            <p className="text-sm text-text-muted">
              {items.length} recent item{items.length === 1 ? "" : "s"}
              {username ? ` for @${username}` : ""}
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-text-secondary">
            <ActivityCount icon={<AtSign size={13} />} label="Mentions" count={mentionCount} />
            <ActivityCount icon={<Bookmark size={13} />} label="Later" count={savedCount} />
          </div>
        </div>

        {items.length === 0 ? (
          <div className="flex min-h-[22rem] items-center justify-center rounded-lg border border-border bg-bg-surface px-6 text-center">
            <div className="max-w-sm">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-lg border border-border bg-bg-elevated text-accent">
                <Bell size={22} />
              </div>
              <h3 className="font-heading text-lg font-semibold text-text-primary">
                No activity yet
              </h3>
              <p className="mt-2 text-sm leading-6 text-text-muted">
                Mentions and saved follow-ups will collect here as your workspace gets moving.
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {items.map((item) => (
              <ActivityCard key={item.id} item={item} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ActivityCount({
  icon,
  label,
  count,
}: {
  icon: ReactNode;
  label: string;
  count: number;
}) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-bg-surface px-2.5 py-1">
      {icon}
      <span>{label}</span>
      <span className="font-semibold text-text-primary">{count}</span>
    </span>
  );
}

function ActivityCard({ item }: { item: ActivityItem }) {
  const source = item.kind === "mention" ? item.mention : item.savedItem;
  const label = item.kind === "mention" ? "Mention" : "Saved for later";
  const labelIcon = item.kind === "mention" ? <AtSign size={12} /> : <Bookmark size={12} />;
  const jumpLabel = getJumpToMessageLabel(item.kind === "mention" ? "mention" : "saved");

  return (
    <article data-testid="activity-card" className="rounded-lg border border-border bg-bg-surface px-4 py-3">
      <div className="mb-2 flex min-w-0 flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs text-text-muted">
          <span className="inline-flex items-center gap-1 rounded-full border border-border bg-bg-elevated px-2 py-0.5 font-semibold text-text-secondary">
            {labelIcon}
            {label}
          </span>
          <ConversationLink item={source} />
          <span>/</span>
          <span className="shrink-0">
            {formatDistanceToNow(new Date(item.occurredAt), { addSuffix: true })}
          </span>
        </div>
        <JumpToMessageLink
          href={source.href}
          label={jumpLabel}
          channelLabel={source.channelLabel}
        />
      </div>

      <MessagePreview item={source} />
    </article>
  );
}

function ConversationLink({ item }: { item: MentionResult | SavedItemResult }) {
  const isDm = item.channelType === "DM";

  return (
    <Link
      href={item.href}
      className="flex min-w-0 items-center gap-2 rounded-md hover:text-accent"
    >
      <span className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md border border-border bg-bg-elevated text-text-secondary">
        {isDm ? <MessageSquare size={13} /> : <Hash size={13} />}
      </span>
      <span className="truncate font-medium text-text-secondary">
        {item.channelLabel}
      </span>
    </Link>
  );
}

function MessagePreview({ item }: { item: MentionResult | SavedItemResult }) {
  return (
    <div className="flex gap-3">
      <ProfilePopover
        user={item.author}
        className="mt-0.5 shrink-0"
        triggerClassName="rounded-full focus:outline-none focus:ring-2 focus:ring-accent/40"
      >
        <span
          className={`avatar h-9 w-9 text-sm ${
            item.author.isAgent
              ? "bg-teal-muted text-teal"
              : "bg-accent-muted text-accent"
          }`}
        >
          {item.author.avatarUrl ? (
            <img
              src={item.author.avatarUrl}
              alt=""
              className="h-full w-full rounded-full object-cover"
            />
          ) : item.author.isAgent ? (
            <Bot size={16} />
          ) : (
            item.author.displayName[0]?.toUpperCase()
          )}
        </span>
      </ProfilePopover>

      <div className="min-w-0 flex-1">
        <div className="mb-1 flex min-w-0 items-baseline gap-2">
          <ProfilePopover
            user={item.author}
            triggerClassName={`rounded-sm text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-accent/40 ${
              item.author.isAgent ? "text-teal" : "text-text-primary"
            }`}
          >
            {item.author.displayName}
          </ProfilePopover>
          {item.author.isAgent && <span className="badge-teal">agent</span>}
          <span className="truncate text-2xs text-text-muted">
            {formatDistanceToNow(new Date(item.createdAt), { addSuffix: true })}
          </span>
        </div>
        <div className="text-sm leading-6 text-text-primary/90">
          <MarkdownContent content={item.content} />
        </div>
      </div>
    </div>
  );
}
