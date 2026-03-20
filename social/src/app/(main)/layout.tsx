import { redirect } from "next/navigation";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import Sidebar from "@/components/layout/Sidebar";
import SocketProvider from "@/components/providers/SocketProvider";
import CallOverlay from "@/components/calls/CallOverlay";
import IncomingCallModal from "@/components/calls/IncomingCallModal";

export default async function MainLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await auth();
  if (!session?.user) redirect("/login");


  const userId = session.user.id;

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

  const channels = memberships
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
      memberCount: m.channel._count.members,
    }));

  const dms = memberships
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

  return (
    <SocketProvider userId={userId} channelIds={allChannelIds} userName={username}>
      <div className="flex h-screen overflow-hidden">
        <Sidebar channels={channels} dms={dms} username={username} userId={userId} />
        <main className="flex-1 flex flex-col overflow-hidden">{children}</main>
      </div>
      <CallOverlay />
      <IncomingCallModal />
    </SocketProvider>
  );
}
