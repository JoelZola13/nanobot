import { notFound } from "next/navigation";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import TopBar from "@/components/layout/TopBar";
import ChannelView from "@/components/channels/ChannelView";
import { formatMessageForClient } from "@/lib/messageFormat";
import { resolveUnreadAfter } from "@/lib/unreadMarker";

const INITIAL_MESSAGE_LIMIT = 50;

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

  const currentMember = channel.members.find((m) => m.userId === session.user!.id);
  const readReceipt = await prisma.readReceipt.findUnique({
    where: {
      userId_channelId: {
        userId: session.user.id,
        channelId,
      },
    },
    select: { readAt: true },
  });
  const initialUnreadAfter = resolveUnreadAfter(
    readReceipt?.readAt,
    currentMember?.joinedAt,
  );

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
    orderBy: { createdAt: "desc" },
    take: INITIAL_MESSAGE_LIMIT + 1,
  });

  const hasOlderMessages = messages.length > INITIAL_MESSAGE_LIMIT;
  const visibleMessages = messages.slice(0, INITIAL_MESSAGE_LIMIT).reverse();
  const formatted = visibleMessages.map((msg) =>
    formatMessageForClient(msg, session.user!.id),
  );

  return (
    <>
      <TopBar
        title={otherName}
        type="dm"
        description={otherMember?.user.isAgent ? "AI Agent" : otherMember?.user.status === "online" ? "Online" : "Offline"}
        channelId={channelId}
        otherUserId={otherMember?.user.id}
        otherUserName={otherMember?.user.displayName}
        detailsMemberCount={channel.members.length}
      />
      <ChannelView
        channelId={channelId}
        channelName={otherName}
        initialMessages={formatted}
        initialOldestMessageCursor={
          hasOlderMessages ? visibleMessages[0]?.id || null : null
        }
        initialUnreadAfter={initialUnreadAfter}
        currentUserId={session.user.id}
        placeholder={`Message ${otherName}`}
        emptyState={{
          kind: "dm",
          name: otherName,
          isAgent: Boolean(otherMember?.user.isAgent),
          status: otherMember?.user.status,
        }}
      />
    </>
  );
}
