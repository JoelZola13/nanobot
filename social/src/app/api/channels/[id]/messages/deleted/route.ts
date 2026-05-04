import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import { canManageChannel } from "@/lib/channelManagement";
import { readMessageDeletionAudit } from "@/lib/messageModeration";

// GET /api/channels/[id]/messages/deleted — manager-only removal audit
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { id: channelId } = await params;
  const limit = Math.min(
    Number.parseInt(req.nextUrl.searchParams.get("limit") || "50", 10) || 50,
    100,
  );

  const channel = await prisma.channel.findUnique({
    where: { id: channelId },
    select: {
      id: true,
      members: {
        where: { userId: session.user.id },
        select: { role: true },
      },
    },
  });
  if (!channel)
    return NextResponse.json({ error: "Not found" }, { status: 404 });

  const membershipRole = channel.members[0]?.role;
  if (!canManageChannel(session.user, membershipRole)) {
    return NextResponse.json(
      { error: "Channel admin access required" },
      { status: 403 },
    );
  }

  const messages = await prisma.message.findMany({
    where: { channelId, deletedAt: { not: null } },
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
    orderBy: { deletedAt: "desc" },
    take: limit,
  });

  return NextResponse.json({
    deletedMessages: messages.map((message) => {
      const audit = readMessageDeletionAudit(message.metadata);

      return {
        id: message.id,
        channelId: message.channelId,
        content: message.content,
        createdAt: message.createdAt.toISOString(),
        deletedAt: message.deletedAt?.toISOString() || audit?.deletedAt || null,
        author: message.author,
        removedBy: audit
          ? { id: audit.actorId, displayName: audit.actorName }
          : null,
        removalMode: audit?.mode || null,
        reason: audit?.reason || null,
      };
    }),
  });
}
