import { prisma } from "@/lib/prisma";

export type SavedItemResult = {
  id: string;
  messageId: string;
  channelId: string;
  content: string;
  createdAt: string;
  savedAt: string;
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

export async function getSavedItemsForUser(userId: string, limit = 50) {
  const savedItems = await prisma.savedItem.findMany({
    where: {
      userId,
      message: {
        deletedAt: null,
        channel: {
          members: {
            some: { userId },
          },
        },
      },
    },
    include: {
      message: {
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
      },
    },
    orderBy: { createdAt: "desc" },
    take: limit,
  });

  return savedItems.map<SavedItemResult>((savedItem) => {
    const { message } = savedItem;
    const isDm = message.channel.type === "DM";
    const otherDmMember = isDm
      ? message.channel.members.find((member) => member.userId !== userId)
      : null;
    const channelLabel = isDm
      ? otherDmMember?.user.displayName || "Direct message"
      : `#${message.channel.name || "channel"}`;
    const basePath = isDm ? "dm" : "channels";

    return {
      id: savedItem.id,
      messageId: message.id,
      channelId: message.channelId,
      content: message.content,
      createdAt: message.createdAt.toISOString(),
      savedAt: savedItem.createdAt.toISOString(),
      href: `/${basePath}/${message.channelId}?message=${message.id}`,
      channelLabel,
      channelType: message.channel.type,
      author: message.author,
    };
  });
}
