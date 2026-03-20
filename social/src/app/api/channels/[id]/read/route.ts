import { NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import { getIO } from "@/lib/socketServer";

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
  const { messageId } = await request.json();

  if (!messageId) {
    return NextResponse.json({ error: "messageId required" }, { status: 400 });
  }

  // Upsert read receipt — unique on [userId, channelId]
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

  const receipts = await prisma.readReceipt.findMany({
    where: { channelId },
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
