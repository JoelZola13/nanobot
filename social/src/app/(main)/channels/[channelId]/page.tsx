import { notFound, redirect } from "next/navigation";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import TopBar from "@/components/layout/TopBar";
import ChannelView from "@/components/channels/ChannelView";

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
      attachments: true,
      _count: { select: { replies: true } },
    },
    orderBy: { createdAt: "asc" },
    take: 100,
  });

  const formatted = messages.map((msg) => {
    const reactionMap = new Map<
      string,
      { count: number; userReacted: boolean }
    >();
    for (const r of msg.reactions) {
      const existing = reactionMap.get(r.emoji) || {
        count: 0,
        userReacted: false,
      };
      existing.count++;
      if (r.userId === session.user!.id) existing.userReacted = true;
      reactionMap.set(r.emoji, existing);
    }

    return {
      id: msg.id,
      channelId: msg.channelId,
      content: msg.content,
      createdAt: msg.createdAt.toISOString(),
      isEdited: msg.isEdited,
      isPinned: msg.isPinned,
      parentId: msg.parentId,
      replyCount: msg._count.replies,
      author: msg.author,
      reactions: Array.from(reactionMap.entries()).map(
        ([emoji, { count, userReacted }]) => ({ emoji, count, userReacted }),
      ),
      attachments: msg.attachments.map((a) => ({
        id: a.id,
        fileName: a.fileName,
        mimeType: a.mimeType,
        url: a.url,
        width: a.width,
        height: a.height,
      })),
    };
  });

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
      />
    </>
  );
}
