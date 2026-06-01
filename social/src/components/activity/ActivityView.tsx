"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { formatDistanceToNow } from "date-fns";
import {
  AtSign,
  Bell,
  BellOff,
  Bookmark,
  Bot,
  Hash,
  MessageSquare,
  Reply,
  SmilePlus,
} from "lucide-react";
import MarkdownContent from "@/components/channels/MarkdownContent";
import NotificationPreferencesPanel from "@/components/channels/NotificationPreferencesPanel";
import JumpToMessageLink from "@/components/messages/JumpToMessageLink";
import { useNotificationPreferences } from "@/components/providers/SocketProvider";
import ProfilePopover from "@/components/users/ProfilePopover";
import {
  filterActivityItems,
  getActivityCounts,
  getUnreadActivityCounts,
  groupActivityItemsByDate,
  type ActivityDateGroup,
  type ActivityFilter,
  type ActivityItem,
  type ReactionActivityResult,
  type ThreadActivityResult,
} from "@/lib/activityItems";
import { apiUrl } from "@/lib/apiUrl";
import { useEmbeddedNavigation } from "@/lib/useEmbeddedNavigation";
import { getJumpToMessageLabel } from "@/lib/messageLinks";
import type { NotificationLevel } from "@/lib/notificationPreferences";
import type { MentionResult } from "@/lib/mentions";
import type { SavedItemResult } from "@/lib/savedItems";

type ActivityCounts = ReturnType<typeof getActivityCounts>;
type UnreadActivityCounts = ReturnType<typeof getUnreadActivityCounts>;
type ActivitySource =
  | MentionResult
  | SavedItemResult
  | ThreadActivityResult
  | ReactionActivityResult;

const ACTIVITY_FILTERS: {
  id: ActivityFilter;
  label: string;
  icon: ReactNode;
  countKey: keyof ActivityCounts;
}[] = [
  { id: "all", label: "All", icon: <Bell size={13} />, countKey: "all" },
  {
    id: "mentions",
    label: "Mentions",
    icon: <AtSign size={13} />,
    countKey: "mentions",
  },
  {
    id: "saved",
    label: "Later",
    icon: <Bookmark size={13} />,
    countKey: "saved",
  },
  {
    id: "threads",
    label: "Threads",
    icon: <Reply size={13} />,
    countKey: "threads",
  },
  {
    id: "reactions",
    label: "Reactions",
    icon: <SmilePlus size={13} />,
    countKey: "reactions",
  },
];

export default function ActivityView({
  username,
  items,
  counts,
  unreadCounts,
}: {
  username: string | null;
  items: ActivityItem[];
  counts: ActivityCounts;
  unreadCounts: UnreadActivityCounts;
}) {
  const router = useRouter();
  const [activeFilter, setActiveFilter] = useState<ActivityFilter>("all");
  const { withEmbed } = useEmbeddedNavigation();
  const filteredItems = useMemo(
    () => filterActivityItems(items, activeFilter),
    [activeFilter, items],
  );
  const groupedItems = useMemo(
    () => groupActivityItemsByDate(filteredItems),
    [filteredItems],
  );
  const activeFilterLabel = getFilterLabel(activeFilter);
  const emptyTitle =
    activeFilter === "all"
      ? "No activity yet"
      : `No ${activeFilterLabel.toLowerCase()} yet`;

  useEffect(() => {
    if (unreadCounts.all === 0) return;

    let cancelled = false;
    fetch(apiUrl("/api/activity/read"), { method: "POST" })
      .then((response) => {
        if (!cancelled && response.ok) router.refresh();
      })
      .catch(() => {});

    return () => {
      cancelled = true;
    };
  }, [router, unreadCounts.all]);

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl px-6 py-5">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-4">
          <div className="min-w-0">
            <h2 className="font-heading text-xl font-semibold text-text-primary">
              Activity
            </h2>
            <p className="text-sm text-text-muted">
              {filteredItems.length} recent item
              {filteredItems.length === 1 ? "" : "s"}
              {username ? ` for @${username}` : ""}
              {unreadCounts.all > 0 && (
                <span className="ml-2 rounded-full bg-accent-muted px-2 py-0.5 text-2xs font-semibold text-accent">
                  {unreadCounts.all} new
                </span>
              )}
            </p>
          </div>
          <ActivityFilterBar
            activeFilter={activeFilter}
            counts={counts}
            unreadCounts={unreadCounts}
            onFilterChange={setActiveFilter}
          />
        </div>

        {filteredItems.length === 0 ? (
          <div
            id="activity-results-panel"
            role="tabpanel"
            aria-labelledby={`activity-filter-tab-${activeFilter}`}
            data-testid="activity-empty-state"
            className="flex min-h-[22rem] items-center justify-center rounded-lg border border-border bg-bg-surface px-6 text-center"
          >
            <div className="max-w-sm">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-lg border border-border bg-bg-elevated text-accent">
                {getEmptyIcon(activeFilter)}
              </div>
              <h3 className="font-heading text-lg font-semibold text-text-primary">
                {emptyTitle}
              </h3>
              <p className="mt-2 text-sm leading-6 text-text-muted">
                {getEmptyCopy(activeFilter)}
              </p>
            </div>
          </div>
        ) : (
          <div
            id="activity-results-panel"
            role="tabpanel"
            aria-labelledby={`activity-filter-tab-${activeFilter}`}
            data-testid="activity-results"
            className="space-y-5"
          >
            {groupedItems.map((group) => (
              <ActivityDateSection
                key={group.id}
                group={group}
                withEmbed={withEmbed}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ActivityFilterBar({
  activeFilter,
  counts,
  unreadCounts,
  onFilterChange,
}: {
  activeFilter: ActivityFilter;
  counts: ActivityCounts;
  unreadCounts: UnreadActivityCounts;
  onFilterChange: (filter: ActivityFilter) => void;
}) {
  return (
    <div
      role="tablist"
      aria-label="Activity filters"
      className="flex max-w-full items-center gap-1 overflow-x-auto rounded-lg border border-border bg-bg-surface p-1 text-xs text-text-secondary"
    >
      {ACTIVITY_FILTERS.map((filter) => {
        const active = filter.id === activeFilter;
        const unread = unreadCounts[filter.countKey];
        return (
          <button
            key={filter.id}
            id={`activity-filter-tab-${filter.id}`}
            type="button"
            role="tab"
            aria-selected={active}
            aria-controls="activity-results-panel"
            aria-label={formatActivityFilterAriaLabel(
              filter.label,
              counts[filter.countKey],
              unread,
            )}
            data-testid={`activity-filter-${filter.id}`}
            data-sv-yellow-surface={active ? "true" : undefined}
            onClick={() => onFilterChange(filter.id)}
            className={`group inline-flex h-8 shrink-0 items-center gap-1.5 rounded-md px-2.5 font-medium transition-colors ${
              active
                ? "bg-accent text-black shadow-sm"
                : "text-white hover:bg-accent hover:text-black"
            }`}
          >
            {filter.icon}
            <span>{filter.label}</span>
            <span
              className={`rounded-full bg-bg-hover px-1.5 py-0.5 text-2xs transition-colors ${
                active ? "text-black" : "text-white group-hover:text-black"
              }`}
              aria-hidden="true"
            >
              {counts[filter.countKey]}
            </span>
            {unread > 0 && (
              <span
                className="rounded-full bg-accent px-1.5 py-0.5 text-2xs font-semibold text-black group-hover:text-black"
                aria-hidden="true"
              >
                {unread}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

function formatActivityFilterAriaLabel(
  label: string,
  count: number,
  unread: number,
) {
  const itemLabel = `${count} item${count === 1 ? "" : "s"}`;
  if (unread <= 0) return `${label} activity, ${itemLabel}`;
  return `${label} activity, ${itemLabel}, ${unread} unread`;
}

function ActivityDateSection({
  group,
  withEmbed,
}: {
  group: ActivityDateGroup;
  withEmbed: (href: string) => string;
}) {
  return (
    <section
      data-testid="activity-date-group"
      aria-labelledby={`activity-date-group-${group.id}`}
      className="space-y-3"
    >
      <div className="flex items-center gap-3">
        <h3
          id={`activity-date-group-${group.id}`}
          className="shrink-0 font-heading text-xs font-semibold uppercase text-text-muted"
        >
          {group.label}
        </h3>
        <span className="h-px flex-1 bg-border" />
        <span className="shrink-0 text-2xs font-medium text-text-muted">
          {group.items.length} item{group.items.length === 1 ? "" : "s"}
        </span>
      </div>
      <div className="space-y-3">
        {group.items.map((item) => (
          <ActivityCard key={item.id} item={item} withEmbed={withEmbed} />
        ))}
      </div>
    </section>
  );
}

function ActivityCard({
  item,
  withEmbed,
}: {
  item: ActivityItem;
  withEmbed: (href: string) => string;
}) {
  const [showNotificationPreferences, setShowNotificationPreferences] =
    useState(false);
  const source = getActivitySource(item);
  const { preferences, setPreference } = useNotificationPreferences();
  const [localNotificationLevel, setLocalNotificationLevel] =
    useState<NotificationLevel | null>(null);
  const notificationLevel =
    localNotificationLevel ?? preferences[source.channelId] ?? null;
  const label = getActivityLabel(item);
  const labelIcon = getActivityIcon(
    item.kind,
    "emoji" in source ? source.emoji : undefined,
  );
  const jumpLabel = getJumpToMessageLabel(item.kind);
  const NotificationShortcutIcon =
    getNotificationShortcutIcon(notificationLevel);
  const notificationLabel = getNotificationLevelLabel(notificationLevel);

  useEffect(() => {
    setLocalNotificationLevel(null);
  }, [source.channelId]);

  return (
    <article
      data-testid="activity-card"
      className={`relative rounded-lg border px-4 py-3 ${
        item.isUnread
          ? "border-accent/50 bg-accent-muted/20"
          : "border-border bg-bg-surface"
      }`}
    >
      <div className="mb-2 flex min-w-0 flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs text-text-muted">
          {item.isUnread && (
            <span
              className="h-2 w-2 rounded-full bg-accent"
              aria-label="Unread activity"
            />
          )}
          <span className="inline-flex items-center gap-1 rounded-full border border-border bg-bg-elevated px-2 py-0.5 font-semibold text-text-secondary">
            {labelIcon}
            {label}
          </span>
          <ConversationLink item={source} withEmbed={withEmbed} />
          <span>/</span>
          <span className="shrink-0">
            {formatDistanceToNow(new Date(item.occurredAt), {
              addSuffix: true,
            })}
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            data-testid="activity-notification-shortcut"
            onClick={() =>
              setShowNotificationPreferences((current) => !current)
            }
            aria-label={`Notification preferences for ${source.channelLabel}: ${notificationLabel}`}
            aria-expanded={showNotificationPreferences}
            title="Notification preferences"
            className={`inline-flex h-7 shrink-0 items-center gap-1 rounded-full border px-2.5 text-2xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-accent/30 ${
              showNotificationPreferences
                ? "border-accent bg-accent-muted text-accent"
                : "border-border bg-bg-elevated text-text-secondary hover:border-accent hover:bg-accent-muted hover:text-accent"
            }`}
          >
            <NotificationShortcutIcon size={12} />
            <span>{notificationLabel}</span>
          </button>
          <JumpToMessageLink
            href={withEmbed(source.href)}
            label={jumpLabel}
            channelLabel={source.channelLabel}
          />
        </div>
      </div>

      {showNotificationPreferences && (
        <NotificationPreferencesPanel
          channelId={source.channelId}
          onClose={() => setShowNotificationPreferences(false)}
          onLevelChange={(level) => {
            setLocalNotificationLevel(level);
            setPreference(source.channelId, level);
          }}
          className="absolute right-3 top-12 z-50 w-80 max-w-[calc(100vw-2rem)] overflow-hidden rounded-xl border border-border bg-bg-surface shadow-xl"
        />
      )}

      <MessagePreview item={source} />
    </article>
  );
}

function ConversationLink({
  item,
  withEmbed,
}: {
  item: ActivitySource;
  withEmbed: (href: string) => string;
}) {
  const isDm = item.channelType === "DM";

  return (
    <Link
      href={withEmbed(item.href)}
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

function MessagePreview({ item }: { item: ActivitySource }) {
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

function getActivitySource(item: ActivityItem): ActivitySource {
  if (item.kind === "mention") return item.mention;
  if (item.kind === "saved") return item.savedItem;
  if (item.kind === "thread") return item.thread;
  return item.reaction;
}

function getActivityLabel(item: ActivityItem) {
  if (item.kind === "mention") return "Mention";
  if (item.kind === "saved") return "Saved for later";
  if (item.kind === "thread") return "Thread reply";
  return `${item.reaction.emoji} reaction`;
}

function getActivityIcon(kind: ActivityItem["kind"], emoji?: string) {
  if (kind === "mention") return <AtSign size={12} />;
  if (kind === "saved") return <Bookmark size={12} />;
  if (kind === "thread") return <Reply size={12} />;
  return emoji ? (
    <span className="text-xs leading-none">{emoji}</span>
  ) : (
    <SmilePlus size={12} />
  );
}

function getNotificationShortcutIcon(level: NotificationLevel | null) {
  if (level === "MUTED") return BellOff;
  if (level === "MENTIONS") return AtSign;
  return Bell;
}

function getNotificationLevelLabel(level: NotificationLevel | null) {
  if (level === "ALL") return "All activity";
  if (level === "MENTIONS") return "Mentions";
  if (level === "MUTED") return "Muted";
  return "Notifications";
}

function getFilterLabel(filter: ActivityFilter) {
  return (
    ACTIVITY_FILTERS.find((item) => item.id === filter)?.label || "Activity"
  );
}

function getEmptyIcon(filter: ActivityFilter) {
  if (filter === "mentions") return <AtSign size={22} />;
  if (filter === "saved") return <Bookmark size={22} />;
  if (filter === "threads") return <Reply size={22} />;
  if (filter === "reactions") return <SmilePlus size={22} />;
  return <Bell size={22} />;
}

function getEmptyCopy(filter: ActivityFilter) {
  if (filter === "mentions") {
    return "Messages that mention your username will show up here.";
  }
  if (filter === "saved") {
    return "Messages you save for follow-up will show up here.";
  }
  if (filter === "threads") {
    return "Replies to your messages will show up here.";
  }
  if (filter === "reactions") {
    return "Reactions to your messages will show up here.";
  }
  return "Mentions, saved follow-ups, thread replies, and reactions will collect here.";
}
