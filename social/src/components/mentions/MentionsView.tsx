"use client";

import Link from "next/link";
import { formatDistanceToNow } from "date-fns";
import { AtSign, Bot, Hash, MessageSquare } from "lucide-react";
import MarkdownContent from "@/components/channels/MarkdownContent";
import JumpToMessageLink from "@/components/messages/JumpToMessageLink";
import ProfilePopover from "@/components/users/ProfilePopover";
import { getJumpToMessageLabel } from "@/lib/messageLinks";
import type { MentionResult } from "@/lib/mentions";

export default function MentionsView({
  username,
  mentions,
}: {
  username: string | null;
  mentions: MentionResult[];
}) {
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl px-6 py-5">
        <div className="mb-4 flex items-center justify-between gap-4">
          <div className="min-w-0">
            <h2 className="font-heading text-xl font-semibold text-text-primary">
              Mentions
            </h2>
            <p className="text-sm text-text-muted">
              {mentions.length} message{mentions.length === 1 ? "" : "s"}
              {username ? ` for @${username}` : ""}
            </p>
          </div>
        </div>

        {mentions.length === 0 ? (
          <div className="flex min-h-[22rem] items-center justify-center rounded-lg border border-border bg-bg-surface px-6 text-center">
            <div className="max-w-sm">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-lg border border-border bg-bg-elevated text-accent">
                <AtSign size={22} />
              </div>
              <h3 className="font-heading text-lg font-semibold text-text-primary">
                No mentions yet
              </h3>
              <p className="mt-2 text-sm leading-6 text-text-muted">
                Messages that include your username will collect here so you can
                jump back into the right channel or DM quickly.
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {mentions.map((mention) => (
              <MentionCard key={mention.id} mention={mention} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function MentionCard({ mention }: { mention: MentionResult }) {
  const isDm = mention.channelType === "DM";

  return (
    <article className="rounded-lg border border-border bg-bg-surface px-4 py-3">
      <div className="mb-2 flex min-w-0 flex-wrap items-center justify-between gap-2">
        <Link
          href={mention.href}
          className="flex min-w-0 items-center gap-2 rounded-md text-xs text-text-muted hover:text-accent"
        >
          <span className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md border border-border bg-bg-elevated text-text-secondary">
            {isDm ? <MessageSquare size={13} /> : <Hash size={13} />}
          </span>
          <span className="truncate font-medium text-text-secondary">
            {mention.channelLabel}
          </span>
          <span>/</span>
          <span className="shrink-0">
            {formatDistanceToNow(new Date(mention.createdAt), { addSuffix: true })}
          </span>
        </Link>
        <JumpToMessageLink
          href={mention.href}
          label={getJumpToMessageLabel("mention")}
          channelLabel={mention.channelLabel}
        />
      </div>

      <div className="flex gap-3">
        <ProfilePopover
          user={mention.author}
          className="mt-0.5 shrink-0"
          triggerClassName="rounded-full focus:outline-none focus:ring-2 focus:ring-accent/40"
        >
          <span
            className={`avatar h-9 w-9 text-sm ${
              mention.author.isAgent
                ? "bg-teal-muted text-teal"
                : "bg-accent-muted text-accent"
            }`}
          >
            {mention.author.avatarUrl ? (
              <img
                src={mention.author.avatarUrl}
                alt=""
                className="h-full w-full rounded-full object-cover"
              />
            ) : mention.author.isAgent ? (
              <Bot size={16} />
            ) : (
              mention.author.displayName[0]?.toUpperCase()
            )}
          </span>
        </ProfilePopover>

        <div className="min-w-0 flex-1">
          <div className="mb-1 flex min-w-0 items-baseline gap-2">
            <ProfilePopover
              user={mention.author}
              triggerClassName={`rounded-sm text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-accent/40 ${
                mention.author.isAgent ? "text-teal" : "text-text-primary"
              }`}
            >
              {mention.author.displayName}
            </ProfilePopover>
            {mention.author.isAgent && <span className="badge-teal">agent</span>}
            <span className="truncate text-2xs text-text-muted">
              @{mention.author.username}
            </span>
          </div>
          <div className="text-sm leading-6 text-text-primary/90">
            <MarkdownContent content={mention.content} />
          </div>
        </div>
      </div>
    </article>
  );
}
