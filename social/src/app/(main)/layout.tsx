import { redirect } from "next/navigation";
import { auth, isLibreChatBridgeUnavailableError } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import {
  ensureDefaultChannelsForUser,
  sortDefaultChannelMemberships,
} from "@/lib/defaultChannels";
import ResponsiveMessagesShell from "@/components/layout/ResponsiveMessagesShell";
import SocketProvider from "@/components/providers/SocketProvider";
import CallOverlay from "@/components/calls/CallOverlay";
import IncomingCallModal from "@/components/calls/IncomingCallModal";
import { normalizeNotificationLevel } from "@/lib/notificationPreferences";

export default async function MainLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  let session;
  try {
    session = await auth({ bridgeUnavailable: "throw" });
  } catch (error) {
    if (isLibreChatBridgeUnavailableError(error)) {
      redirect("/bridge-unavailable");
    }
    throw error;
  }

  if (!session?.user) redirect("/login");

  const userId = session.user.id;
  await ensureDefaultChannelsForUser(userId);

  // Fetch user's channels and DMs
  const memberships = await prisma.channelMember.findMany({
    where: { userId },
    include: {
      channel: {
        include: {
          _count: { select: { members: true } },
          members: {
            include: {
              user: {
                select: { id: true, displayName: true, avatarUrl: true, isAgent: true, status: true },
              },
            },
          },
        },
      },
    },
    orderBy: { channel: { updatedAt: "desc" } },
  });

  const sortedMemberships = sortDefaultChannelMemberships(memberships);

  const channels = sortedMemberships
    .filter(
      (m) =>
        m.channel.type === "PUBLIC" || m.channel.type === "PRIVATE",
    )
    .map((m) => ({
      id: m.channel.id,
      name: m.channel.name,
      slug: m.channel.slug,
      description: m.channel.description,
      type: m.channel.type as "PUBLIC" | "PRIVATE",
      iconEmoji: m.channel.iconEmoji,
      isDefault: m.channel.isDefault,
      memberCount: m.channel._count.members,
      role: m.role,
    }));

  const dms = sortedMemberships
    .filter(
      (m) => m.channel.type === "DM" || m.channel.type === "GROUP_DM",
    )
    .map((m) => {
      // For DMs, show the other person's name
      const otherMember = m.channel.members.find((member) => member.userId !== userId);
      return {
        id: m.channel.id,
        name: otherMember?.user.displayName || m.channel.name || "Unknown",
        slug: m.channel.slug,
        description: m.channel.description,
        type: m.channel.type as "DM" | "GROUP_DM",
        iconEmoji: m.channel.iconEmoji,
        otherUser: otherMember?.user || null,
      };
    });

  const username =
    (session.user as Record<string, unknown>).username as string ||
    session.user.name ||
    "User";

  const allChannelIds = memberships.map((m) => m.channel.id);
  const notificationPreferences = Object.fromEntries(
    memberships.map((membership) => [
      membership.channel.id,
      normalizeNotificationLevel(membership.notificationLevel, membership.mutedAt),
    ]),
  );

  return (
    <SocketProvider
      userId={userId}
      channelIds={allChannelIds}
      userName={username}
      notificationPreferences={notificationPreferences}
    >
      <ResponsiveMessagesShell channels={channels} dms={dms} userId={userId}>
        {children}
      </ResponsiveMessagesShell>
      <CallOverlay />
      <IncomingCallModal />
    </SocketProvider>
  );
}
