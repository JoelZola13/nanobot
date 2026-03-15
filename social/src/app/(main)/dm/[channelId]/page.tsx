import { notFound } from "next/navigation";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import TopBar from "@/components/layout/TopBar";
import ChannelView from "@/components/channels/ChannelView";

export default async function DMConversationPage({
  params,
}: {
  params: Promise<{ channelId: string }>;
}) {
  const session = await auth();
  if (!session?.user?.id) notFound();

  const { channelId } = await params;

  // Fetch channel with membership check
  const channel = await prisma.channel.findUnique({
    where: { id: channelId },
    include: {
      members: {
        include: {
          user: {
            select: { id: true, displayName: true, avatarUrl: true, isAgent: true, status: true },
          },
        },
      },
    },
  });

  if (!channel || channel.type !== "DM") notFound();

  const isMember = channel.members.some((m) => m.userId === session.user!.id);
  if (!isMember) notFound();

  // Find the other user in this DM
  const otherMember = channel.members.find((m) => m.userId !== session.user!.id);
  const otherName = otherMember?.user.displayName || "Unknown";

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
    take: 50,
  });

  const formatted = messages.map((msg) => {
    const reactionMap = new Map<string, { count: number; userReacted: boolean }>();
    for (const r of msg.reactions) {
      const existing = reactionMap.get(r.emoji) || { count: 0, userReacted: false };
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
      reactions: Array.from(reactionMap.entries()).map(([emoji, { count, userReacted }]) => ({
        emoji,
        count,
        userReacted,
      })),
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
        title={otherName}
        type="dm"
        description={otherMember?.user.isAgent ? "AI Agent" : otherMember?.user.status === "online" ? "Online" : "Offline"}
        channelId={channelId}
        otherUserId={otherMember?.user.id}
        otherUserName={otherMember?.user.displayName}
      />
      <ChannelView
        channelId={channelId}
        channelName={otherName}
        initialMessages={formatted}
        currentUserId={session.user.id}
      />
    </>
  );
}
