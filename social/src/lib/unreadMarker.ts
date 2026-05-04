type UnreadBasis = Date | string | null | undefined;

type UnreadMarkerMessage = {
  id: string;
  createdAt: string;
  author: {
    id: string;
  };
};

function toTimestamp(value: UnreadBasis) {
  if (!value) return null;
  const timestamp = value instanceof Date ? value.getTime() : Date.parse(value);
  return Number.isFinite(timestamp) ? timestamp : null;
}

export function resolveUnreadAfter(readAt: UnreadBasis, joinedAt: UnreadBasis) {
  const readTimestamp = toTimestamp(readAt);
  const joinedTimestamp = toTimestamp(joinedAt);
  const unreadAfterTimestamp =
    readTimestamp !== null && joinedTimestamp !== null
      ? Math.max(readTimestamp, joinedTimestamp)
      : readTimestamp ?? joinedTimestamp;

  return unreadAfterTimestamp === null
    ? null
    : new Date(unreadAfterTimestamp).toISOString();
}

export function findFirstUnreadMessageId(
  messages: UnreadMarkerMessage[],
  currentUserId: string,
  unreadAfter: string | null | undefined,
) {
  const unreadAfterTimestamp = toTimestamp(unreadAfter);
  if (unreadAfterTimestamp === null) return null;

  const firstUnreadMessage = messages.find((message) => {
    if (message.author.id === currentUserId) return false;
    const messageTimestamp = toTimestamp(message.createdAt);
    return messageTimestamp !== null && messageTimestamp > unreadAfterTimestamp;
  });

  return firstUnreadMessage?.id ?? null;
}
