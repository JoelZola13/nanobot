import { prisma } from "@/lib/prisma";
import { getMentionMessagesForUser } from "@/lib/mentions";
import { getMessageHref } from "@/lib/messageLinks";
import { getSavedItemsForUser } from "@/lib/savedItems";
import {
  getActivityCounts,
  getUnreadActivityCounts,
  mergeActivityItems,
  type ReactionActivityResult,
  type ThreadActivityResult,
} from "@/lib/activityItems";

type ChannelType = "PUBLIC" | "PRIVATE" | "DM" | "GROUP_DM";

export {
  filterActivityItems,
  getActivityCounts,
  getUnreadActivityCounts,
  mergeActivityItems,
} from "@/lib/activityItems";
export type {
  ActivityFilter,
  ActivityItem,
  ReactionActivityResult,
  ThreadActivityResult,
} from "@/lib/activityItems";

function getChannelLabel({
  channel,
  userId,
}: {
  channel: {
    name: string | null;
    type: ChannelType;
    members: { userId: string; user: { displayName: string } }[];
  };
  userId: string;
}) {
  if (channel.type !== "DM") return `#${channel.name || "channel"}`;
  const otherDmMember = channel.members.find((member) => member.userId !== userId);
  return otherDmMember?.user.displayName || "Direct message";
}

async function getThreadActivityForUser(userId: string, limit: number) {
  const replies = await prisma.message.findMany({
    where: {
      deletedAt: null,
      parentId: { not: null },
      authorId: { not: userId },
      channel: {
        members: {
          some: { userId },
        },
      },
      parent: {
        is: {
          authorId: userId,
          deletedAt: null,
        },
      },
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
                  displayName: true,
                },
              },
            },
          },
        },
      },
    },
    orderBy: { createdAt: "desc" },
    take: limit,
  });

  return replies.flatMap<ThreadActivityResult>((reply) => {
    if (!reply.parentId) return [];
    return {
      id: reply.id,
      replyId: reply.id,
      parentMessageId: reply.parentId,
      channelId: reply.channelId,
      content: reply.content,
      createdAt: reply.createdAt.toISOString(),
      href: getMessageHref({
        channelId: reply.channelId,
        messageId: reply.parentId,
        channelType: reply.channel.type,
      }),
      channelLabel: getChannelLabel({ channel: reply.channel, userId }),
      channelType: reply.channel.type,
      author: reply.author,
    };
  });
}

async function getReactionActivityForUser(userId: string, limit: number) {
  const messages = await prisma.message.findMany({
    where: {
      authorId: userId,
      deletedAt: null,
      channel: {
        members: {
          some: { userId },
        },
      },
      reactions: {
        some: {
          userId: { not: userId },
        },
      },
    },
    include: {
      channel: {
        include: {
          members: {
            include: {
              user: {
                select: {
                  displayName: true,
                },
              },
            },
          },
        },
      },
      reactions: {
        where: {
          userId: { not: userId },
        },
        include: {
          user: {
            select: {
              id: true,
              username: true,
              displayName: true,
              avatarUrl: true,
              isAgent: true,
            },
          },
        },
      },
    },
    orderBy: { updatedAt: "desc" },
    take: limit,
  });

  return messages.flatMap<ReactionActivityResult>((message) =>
    message.reactions.map((reaction) => ({
      id: reaction.id,
      messageId: message.id,
      channelId: message.channelId,
      content: message.content,
      createdAt: message.createdAt.toISOString(),
      reactedAt: message.updatedAt.toISOString(),
      href: getMessageHref({
        channelId: message.channelId,
        messageId: message.id,
        channelType: message.channel.type,
      }),
      channelLabel: getChannelLabel({ channel: message.channel, userId }),
      channelType: message.channel.type,
      emoji: reaction.emoji,
      author: reaction.user,
    })),
  );
}

export async function getActivityForUser(userId: string, limit = 50) {
  const [readState, { username, mentions }, savedItems, threads, reactions] = await Promise.all([
    prisma.activityReadState.findUnique({
      where: { userId },
      select: { readAt: true },
    }),
    getMentionMessagesForUser(userId, limit),
    getSavedItemsForUser(userId, limit),
    getThreadActivityForUser(userId, limit),
    getReactionActivityForUser(userId, limit),
  ]);
  const readAt = readState?.readAt ?? null;
  const items = mergeActivityItems({ mentions, savedItems, threads, reactions, limit })
    .map((item) => ({
      ...item,
      isUnread: readAt
        ? new Date(item.occurredAt).getTime() > readAt.getTime()
        : true,
    }));
  const counts = getActivityCounts(items);
  const unreadCounts = getUnreadActivityCounts(items);

  return {
    username,
    items,
    counts,
    unreadCounts,
    readAt: readAt?.toISOString() ?? null,
  };
}

export async function getActivityUnreadCountForUser(userId: string) {
  const activity = await getActivityForUser(userId);
  return activity.unreadCounts.all;
}

export async function markActivityReadForUser(userId: string, readAt = new Date()) {
  const state = await prisma.activityReadState.upsert({
    where: { userId },
    update: { readAt },
    create: { userId, readAt },
    select: { readAt: true },
  });

  return state.readAt;
}
