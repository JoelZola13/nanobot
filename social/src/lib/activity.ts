import { getMentionMessagesForUser, type MentionResult } from "@/lib/mentions";
import { getSavedItemsForUser, type SavedItemResult } from "@/lib/savedItems";

export type ActivityItem =
  | {
      id: string;
      kind: "mention";
      occurredAt: string;
      mention: MentionResult;
    }
  | {
      id: string;
      kind: "saved";
      occurredAt: string;
      savedItem: SavedItemResult;
    };

export function mergeActivityItems({
  mentions,
  savedItems,
  limit = 50,
}: {
  mentions: MentionResult[];
  savedItems: SavedItemResult[];
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
  ]
    .sort(
      (a, b) =>
        new Date(b.occurredAt).getTime() - new Date(a.occurredAt).getTime(),
    )
    .slice(0, limit);
}

export async function getActivityForUser(userId: string, limit = 50) {
  const [{ username, mentions }, savedItems] = await Promise.all([
    getMentionMessagesForUser(userId, limit),
    getSavedItemsForUser(userId, limit),
  ]);

  return {
    username,
    items: mergeActivityItems({ mentions, savedItems, limit }),
    mentionCount: mentions.length,
    savedCount: savedItems.length,
  };
}
