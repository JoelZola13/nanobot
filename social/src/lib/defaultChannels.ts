import { prisma } from "@/lib/prisma";
import { getDefaultMembershipPreferences } from "@/lib/workspacePolicies";

export const DEFAULT_CHANNELS = [
  {
    slug: "announcements",
    name: "announcements",
    description: "Official team updates and workspace announcements.",
  },
  {
    slug: "general",
    name: "general",
    description: "Team-wide discussion for everyday coordination.",
  },
  {
    slug: "help",
    name: "help",
    description: "Questions, blockers, and teammate support.",
  },
] as const;

type SortableChannelMembership = {
  channel: {
    isDefault: boolean;
    slug: string | null;
    updatedAt: Date;
  };
};

const DEFAULT_CHANNEL_ORDER = new Map<string, number>(
  DEFAULT_CHANNELS.map((channel, index) => [channel.slug, index]),
);

function isUniqueConstraintError(error: unknown) {
  return (error as { code?: string }).code === "P2002";
}

export function sortDefaultChannelMemberships<T extends SortableChannelMembership>(
  memberships: T[],
) {
  return [...memberships].sort((a, b) => {
    if (a.channel.isDefault !== b.channel.isDefault) {
      return a.channel.isDefault ? -1 : 1;
    }

    if (a.channel.isDefault && b.channel.isDefault) {
      const aRank =
        DEFAULT_CHANNEL_ORDER.get(a.channel.slug || "") ??
        Number.MAX_SAFE_INTEGER;
      const bRank =
        DEFAULT_CHANNEL_ORDER.get(b.channel.slug || "") ??
        Number.MAX_SAFE_INTEGER;
      if (aRank !== bRank) return aRank - bRank;
    }

    return b.channel.updatedAt.getTime() - a.channel.updatedAt.getTime();
  });
}

export async function ensureDefaultChannelsForUser(userId: string) {
  const channels = [];

  for (const defaultChannel of DEFAULT_CHANNELS) {
    const channel = await prisma.channel.upsert({
      where: { slug: defaultChannel.slug },
      update: {
        name: defaultChannel.name,
        description: defaultChannel.description,
        type: "PUBLIC",
        isArchived: false,
        isDefault: true,
      },
      create: {
        name: defaultChannel.name,
        slug: defaultChannel.slug,
        description: defaultChannel.description,
        type: "PUBLIC",
        isDefault: true,
        createdById: userId,
      },
    });

    const existingMember = await prisma.channelMember.findUnique({
      where: {
        channelId_userId: {
          channelId: channel.id,
          userId,
        },
      },
    });

    if (!existingMember) {
      try {
        await prisma.channelMember.create({
          data: {
            channelId: channel.id,
            userId,
            role: "member",
            ...getDefaultMembershipPreferences(),
          },
        });
      } catch (error) {
        if (!isUniqueConstraintError(error)) {
          throw error;
        }

        await prisma.channelMember.findUnique({
          where: {
            channelId_userId: {
              channelId: channel.id,
              userId,
            },
          },
        });
      }
    }

    channels.push(channel);
  }

  return channels;
}
