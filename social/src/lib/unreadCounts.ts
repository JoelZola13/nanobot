import { prisma } from "./prisma";

type MembershipReadBasis = {
  channelId: string;
  joinedAt: Date;
};

export async function getInitialUnreadCountsForUser(
  userId: string,
  memberships: MembershipReadBasis[],
) {
  const uniqueMemberships = Array.from(
    new Map(memberships.map((membership) => [membership.channelId, membership])).values(),
  );

  if (!userId || uniqueMemberships.length === 0) return {};

  const channelIds = uniqueMemberships.map((membership) => membership.channelId);
  const readReceipts = await prisma.readReceipt.findMany({
    where: {
      userId,
      channelId: { in: channelIds },
    },
    select: {
      channelId: true,
      readAt: true,
    },
  });

  const readAtByChannel = new Map(
    readReceipts.map((receipt) => [receipt.channelId, receipt.readAt]),
  );

  const entries = await Promise.all(
    uniqueMemberships.map(async (membership) => {
      const lastReadAt = readAtByChannel.get(membership.channelId);
      const unreadAfter =
        lastReadAt && lastReadAt > membership.joinedAt ? lastReadAt : membership.joinedAt;

      const count = await prisma.message.count({
        where: {
          channelId: membership.channelId,
          deletedAt: null,
          authorId: { not: userId },
          createdAt: { gt: unreadAfter },
        },
      });

      return [membership.channelId, count] as const;
    }),
  );

  return Object.fromEntries(entries.filter(([, count]) => count > 0));
}
