import { notFound, redirect } from "next/navigation";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import TopBar from "@/components/layout/TopBar";
import ChannelView from "@/components/channels/ChannelView";
import { formatMessageForClient } from "@/lib/messageFormat";

export default async function ChannelPage({
  params,
}: {
  params: Promise<{ channelId: string }>;
}) {
  const session = await auth();
  if (!session?.user?.id) redirect("/login");

  const { channelId } = await params;

  const channel = await prisma.channel.findUnique({
    where: { id: channelId },
    include: { _count: { select: { members: true } } },
  });
  if (!channel) notFound();

  // Verify membership
  const membership = await prisma.channelMember.findUnique({
    where: {
      channelId_userId: { channelId, userId: session.user.id },
    },
  });
  if (!membership) notFound();

  // Fetch messages
  const messages = await prisma.message.findMany({
    where: { channelId, deletedAt: null, parentId: null },
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
      reactions: { select: { emoji: true, userId: true } },
      savedItems: {
        where: { userId: session.user.id },
        select: { userId: true },
      },
      attachments: true,
      replies: {
        where: { deletedAt: null },
        orderBy: { createdAt: "desc" },
        take: 3,
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
        },
      },
      _count: { select: { replies: true } },
    },
    orderBy: { createdAt: "asc" },
    take: 100,
  });

  const formatted = messages.map((msg) => formatMessageForClient(msg, session.user!.id));

  return (
    <>
      <TopBar
        title={channel.name || "Channel"}
        description={channel.description || undefined}
        type="channel"
        memberCount={channel._count.members}
        channelId={channelId}
      />
      <ChannelView
        channelId={channelId}
        channelName={channel.name || "channel"}
        initialMessages={formatted}
        currentUserId={session.user.id}
        placeholder={`Message #${channel.name || "channel"}`}
        emptyState={{
          kind: "channel",
          name: channel.name || "channel",
          description: channel.description || undefined,
          isPrivate: channel.type === "PRIVATE",
          memberCount: channel._count.members,
        }}
      />
    </>
  );
}
