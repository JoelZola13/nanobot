import type { MentionResult } from "@/lib/mentions";
import type { SavedItemResult } from "@/lib/savedItems";

type ChannelType = "PUBLIC" | "PRIVATE" | "DM" | "GROUP_DM";

type ActivityAuthor = {
  id: string;
  username: string;
  displayName: string;
  avatarUrl: string | null;
  isAgent: boolean;
};

export type ActivityFilter = "all" | "mentions" | "saved" | "threads" | "reactions";
export type ActivityDateGroup = {
  id: string;
  label: string;
  items: ActivityItem[];
};

export type ThreadActivityResult = {
  id: string;
  replyId: string;
  parentMessageId: string;
  channelId: string;
  content: string;
  createdAt: string;
  href: string;
  channelLabel: string;
  channelType: ChannelType;
  author: ActivityAuthor;
};

export type ReactionActivityResult = {
  id: string;
  messageId: string;
  channelId: string;
  content: string;
  createdAt: string;
  reactedAt: string;
  href: string;
  channelLabel: string;
  channelType: ChannelType;
  emoji: string;
  author: ActivityAuthor;
};

export type ActivityItem =
  | {
      id: string;
      kind: "mention";
      occurredAt: string;
      isUnread?: boolean;
      mention: MentionResult;
    }
  | {
      id: string;
      kind: "saved";
      occurredAt: string;
      isUnread?: boolean;
      savedItem: SavedItemResult;
    }
  | {
      id: string;
      kind: "thread";
      occurredAt: string;
      isUnread?: boolean;
      thread: ThreadActivityResult;
    }
  | {
      id: string;
      kind: "reaction";
      occurredAt: string;
      isUnread?: boolean;
      reaction: ReactionActivityResult;
    };

export function mergeActivityItems({
  mentions,
  savedItems,
  threads = [],
  reactions = [],
  limit = 50,
}: {
  mentions: MentionResult[];
  savedItems: SavedItemResult[];
  threads?: ThreadActivityResult[];
  reactions?: ReactionActivityResult[];
  limit?: number;
}) {
  return [
    ...mentions.map<ActivityItem>((mention) => ({
      id: `mention:${mention.id}`,
      kind: "mention",
      occurredAt: mention.createdAt,
      mention,
    })),
    ...savedItems.map<ActivityItem>((savedItem) => ({
      id: `saved:${savedItem.id}`,
      kind: "saved",
      occurredAt: savedItem.savedAt,
      savedItem,
    })),
    ...threads.map<ActivityItem>((thread) => ({
      id: `thread:${thread.id}`,
      kind: "thread",
      occurredAt: thread.createdAt,
      thread,
    })),
    ...reactions.map<ActivityItem>((reaction) => ({
      id: `reaction:${reaction.id}`,
      kind: "reaction",
      occurredAt: reaction.reactedAt,
      reaction,
    })),
  ]
    .sort(
      (a, b) =>
        new Date(b.occurredAt).getTime() - new Date(a.occurredAt).getTime(),
    )
    .slice(0, limit);
}

export function getActivityCounts(items: ActivityItem[]) {
  return {
    all: items.length,
    mentions: items.filter((item) => item.kind === "mention").length,
    saved: items.filter((item) => item.kind === "saved").length,
    threads: items.filter((item) => item.kind === "thread").length,
    reactions: items.filter((item) => item.kind === "reaction").length,
  };
}

export function getUnreadActivityCounts(items: ActivityItem[]) {
  return getActivityCounts(items.filter((item) => item.isUnread));
}

export function filterActivityItems(items: ActivityItem[], filter: ActivityFilter) {
  if (filter === "all") return items;
  const kind =
    filter === "mentions"
      ? "mention"
      : filter === "threads"
        ? "thread"
        : filter === "reactions"
          ? "reaction"
          : "saved";
  return items.filter((item) => item.kind === kind);
}

const DAY_IN_MS = 24 * 60 * 60 * 1000;
const dateGroupFormatter = new Intl.DateTimeFormat("en-US", {
  month: "long",
  day: "numeric",
  year: "numeric",
});

function dayNumber(date: Date) {
  return Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()) / DAY_IN_MS;
}

function dateGroupKey(date: Date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export function getActivityDateGroupLabel(
  occurredAt: string,
  now: Date = new Date(),
) {
  const occurredDate = new Date(occurredAt);
  if (Number.isNaN(occurredDate.getTime())) return "Unknown date";

  const daysAgo = dayNumber(now) - dayNumber(occurredDate);
  if (daysAgo === 0) return "Today";
  if (daysAgo === 1) return "Yesterday";
  return dateGroupFormatter.format(occurredDate);
}

export function groupActivityItemsByDate(
  items: ActivityItem[],
  now: Date = new Date(),
) {
  const groups = new Map<string, ActivityDateGroup>();

  items.forEach((item) => {
    const occurredDate = new Date(item.occurredAt);
    const id = Number.isNaN(occurredDate.getTime())
      ? "unknown"
      : dateGroupKey(occurredDate);
    const existingGroup = groups.get(id);

    if (existingGroup) {
      existingGroup.items.push(item);
      return;
    }

    groups.set(id, {
      id,
      label: getActivityDateGroupLabel(item.occurredAt, now),
      items: [item],
    });
  });

  return Array.from(groups.values());
}
