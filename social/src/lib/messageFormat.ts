import type { MessageData } from "@/types";

type RawAuthor = {
  id: string;
  username: string;
  displayName: string;
  avatarUrl: string | null;
  isAgent: boolean;
};

type RawReaction = {
  emoji: string;
  userId: string;
};

type RawAttachment = {
  id: string;
  fileName: string;
  mimeType: string;
  url: string;
  width: number | null;
  height: number | null;
};

type RawReply = {
  id: string;
  content: string;
  createdAt: Date;
  author: RawAuthor;
};

type RawMessage = {
  id: string;
  channelId: string;
  content: string;
  createdAt: Date;
  isEdited: boolean;
  isPinned: boolean;
  parentId: string | null;
  metadata?: unknown;
  author: RawAuthor;
  reactions?: RawReaction[];
  attachments?: RawAttachment[];
  replies?: RawReply[];
  _count?: { replies: number };
};

const toThreadParticipant = (author: RawAuthor) => ({
  id: author.id,
  displayName: author.displayName,
  avatarUrl: author.avatarUrl,
  isAgent: author.isAgent,
});

const buildThreadPreview = (replies: RawReply[] | undefined): MessageData["threadPreview"] => {
  if (!replies?.length) return undefined;

  const latestReply = replies[0];
  const participants = Array.from(
    replies
      .reduce((map, reply) => {
        map.set(reply.author.id, toThreadParticipant(reply.author));
        return map;
      }, new Map<string, ReturnType<typeof toThreadParticipant>>())
      .values(),
  ).slice(0, 3);

  return {
    participants,
    latestReply: {
      id: latestReply.id,
      content: latestReply.content,
      createdAt: latestReply.createdAt.toISOString(),
      author: toThreadParticipant(latestReply.author),
    },
  };
};

export function formatMessageForClient(msg: RawMessage, currentUserId: string): MessageData {
  const reactionMap = new Map<string, { count: number; userReacted: boolean }>();

  for (const reaction of msg.reactions || []) {
    const existing = reactionMap.get(reaction.emoji) || {
      count: 0,
      userReacted: false,
    };
    existing.count++;
    if (reaction.userId === currentUserId) existing.userReacted = true;
    reactionMap.set(reaction.emoji, existing);
  }

  return {
    id: msg.id,
    channelId: msg.channelId,
    content: msg.content,
    createdAt: msg.createdAt.toISOString(),
    isEdited: msg.isEdited,
    isPinned: msg.isPinned,
    parentId: msg.parentId,
    replyCount: msg._count?.replies ?? 0,
    threadPreview: buildThreadPreview(msg.replies),
    author: msg.author,
    metadata: (msg.metadata as MessageData["metadata"]) || undefined,
    reactions: Array.from(reactionMap.entries()).map(([emoji, { count, userReacted }]) => ({
      emoji,
      count,
      userReacted,
    })),
    attachments: (msg.attachments || []).map((attachment) => ({
      id: attachment.id,
      fileName: attachment.fileName,
      mimeType: attachment.mimeType,
      url: attachment.url,
      width: attachment.width,
      height: attachment.height,
    })),
  };
}
