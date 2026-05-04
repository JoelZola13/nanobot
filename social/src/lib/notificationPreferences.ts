export const NOTIFICATION_LEVELS = ["ALL", "MENTIONS", "MUTED"] as const;

export type NotificationLevel = (typeof NOTIFICATION_LEVELS)[number];

export const isNotificationLevel = (level: unknown): level is NotificationLevel =>
  typeof level === "string" &&
  NOTIFICATION_LEVELS.includes(level as NotificationLevel);

export function normalizeNotificationLevel(
  level: string | null | undefined,
  mutedAt?: Date | string | null,
): NotificationLevel {
  if (isNotificationLevel(level)) return level;
  return mutedAt ? "MUTED" : "ALL";
}

const escapeRegExp = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export function messageMentionsUsername(content: string, username?: string) {
  if (!username) return false;

  const mentionPattern = new RegExp(
    `(^|[^A-Za-z0-9_])@${escapeRegExp(username)}(?=$|[^A-Za-z0-9_-])`,
    "i",
  );

  return mentionPattern.test(content);
}
