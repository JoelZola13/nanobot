import { prisma } from "@/lib/prisma";
import { getMessageHref } from "@/lib/messageLinks";

const MENTION_FETCH_LIMIT = 100;

const escapeRegExp = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export type MentionResult = {
  id: string;
  channelId: string;
  content: string;
  createdAt: string;
  href: string;
  channelLabel: string;
  channelType: "PUBLIC" | "PRIVATE" | "DM" | "GROUP_DM";
  author: {
    id: string;
    username: string;
    displayName: string;
    avatarUrl: string | null;
    isAgent: boolean;
  };
};

export async function getMentionMessagesForUser(userId: string, limit = 50) {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { username: true },
  });

  if (!user?.username) {
    return { username: null, mentions: [] as MentionResult[] };
  }

  const memberships = await prisma.channelMember.findMany({
    where: { userId },
    select: { channelId: true },
  });

  const channelIds = memberships.map((membership) => membership.channelId);
  if (channelIds.length === 0) {
    return { username: user.username, mentions: [] as MentionResult[] };
  }

  const mentionPattern = new RegExp(
    `(^|[^A-Za-z0-9_])@${escapeRegExp(user.username)}(?=$|[^A-Za-z0-9_-])`,
    "i",
  );

  const candidates = await prisma.message.findMany({
    where: {
      channelId: { in: channelIds },
      deletedAt: null,
      content: { contains: `@${user.username}`, mode: "insensitive" },
    },
    include: {
      author: {
        select: {
          id: true,
          username: true,
          displayName: true,
          avatarUrl: true,
          isAgent: true,
        },
      },
      channel: {
        include: {
          members: {
            include: {
              user: {
                select: {
                  id: true,
                  displayName: true,
                },
              },
            },
          },
        },
      },
    },
    orderBy: { createdAt: "desc" },
    take: Math.max(limit * 3, MENTION_FETCH_LIMIT),
  });

  const mentions = candidates
    .filter((message) => mentionPattern.test(message.content))
    .slice(0, limit)
    .map<MentionResult>((message) => {
      const isDm = message.channel.type === "DM";
      const otherDmMember = isDm
        ? message.channel.members.find((member) => member.userId !== userId)
        : null;
      const channelLabel = isDm
        ? otherDmMember?.user.displayName || "Direct message"
        : `#${message.channel.name || "channel"}`;

      return {
        id: message.id,
        channelId: message.channelId,
        content: message.content,
        createdAt: message.createdAt.toISOString(),
        href: getMessageHref({
          channelId: message.channelId,
          messageId: message.id,
          channelType: message.channel.type,
        }),
        channelLabel,
        channelType: message.channel.type,
        author: message.author,
      };
    });

  return { username: user.username, mentions };
}
