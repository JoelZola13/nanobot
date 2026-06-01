import { NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import { getIO } from "@/lib/socketServer";

async function requireChannelAccess(channelId: string, userId: string) {
  const membership = await prisma.channelMember.findUnique({
    where: { channelId_userId: { channelId, userId } },
    include: { channel: { select: { isArchived: true } } },
  });

  if (!membership) {
    return NextResponse.json({ error: "Not a member" }, { status: 403 });
  }

  if (membership.channel.isArchived) {
    return NextResponse.json({ error: "Channel is archived" }, { status: 403 });
  }

  return null;
}

// POST /api/channels/[id]/read — mark channel as read up to a given messageId
export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: channelId } = await params;
  const body = (await request.json().catch(() => ({}))) as {
    messageId?: unknown;
  };
  const messageId =
    typeof body.messageId === "string" ? body.messageId.trim() : "";

  if (!messageId) {
    return NextResponse.json({ error: "messageId required" }, { status: 400 });
  }

  const accessError = await requireChannelAccess(channelId, session.user.id);
  if (accessError) {
    return accessError;
  }

  const message = await prisma.message.findUnique({
    where: { id: messageId },
    select: { id: true, channelId: true, deletedAt: true },
  });

  if (!message || message.channelId !== channelId || message.deletedAt) {
    return NextResponse.json(
      { ok: false, ignored: true, reason: "message-not-found" },
      { status: 202 },
    );
  }

  const receipt = await prisma.readReceipt.upsert({
    where: {
      userId_channelId: {
        userId: session.user.id,
        channelId,
      },
    },
    update: {
      messageId,
      readAt: new Date(),
    },
    create: {
      channelId,
      messageId,
      userId: session.user.id,
    },
  });

  // Broadcast via socket if available
  const io = getIO();

  if (io) {
    io.to(`channel:${channelId}`).emit("read:update", {
      channelId,
      userId: session.user.id,
      messageId,
      readAt: receipt.readAt.toISOString(),
    });
  }

  return NextResponse.json({ ok: true, messageId, readAt: receipt.readAt });
}

// GET /api/channels/[id]/read — get all read receipts for a channel
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: channelId } = await params;

  const accessError = await requireChannelAccess(channelId, session.user.id);
  if (accessError) {
    return accessError;
  }

  const receipts = await prisma.readReceipt.findMany({
    where: { channelId, message: { deletedAt: null } },
    include: {
      user: {
        select: { id: true, displayName: true, avatarUrl: true },
      },
    },
  });

  return NextResponse.json(
    receipts.map((r) => ({
      userId: r.userId,
      messageId: r.messageId,
      readAt: r.readAt.toISOString(),
      user: r.user,
    })),
  );
}
