import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";

function getIO() {
  return (globalThis as Record<string, unknown>).__socketio as
    | { to: (room: string) => { emit: (event: string, data: unknown) => void } }
    | undefined;
}

// POST /api/channels/[id]/messages/[messageId]/reactions — toggle reaction
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ id: string; messageId: string }> },
) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { id: channelId, messageId } = await params;
  const { emoji } = await req.json();

  if (!emoji)
    return NextResponse.json({ error: "Emoji required" }, { status: 400 });

  const message = await prisma.message.findUnique({ where: { id: messageId } });
  if (!message || message.channelId !== channelId)
    return NextResponse.json({ error: "Not found" }, { status: 404 });

  // Toggle: remove if exists, create if not
  const existing = await prisma.reaction.findUnique({
    where: { messageId_userId_emoji: { messageId, userId: session.user.id, emoji } },
  });

  if (existing) {
    await prisma.reaction.delete({ where: { id: existing.id } });
  } else {
    await prisma.reaction.create({
      data: { messageId, userId: session.user.id, emoji },
    });
  }

  // Fetch updated reactions for this message
  const reactions = await prisma.reaction.findMany({ where: { messageId } });
  const reactionMap = new Map<string, { count: number; users: string[] }>();
  for (const r of reactions) {
    const entry = reactionMap.get(r.emoji) || { count: 0, users: [] };
    entry.count++;
    entry.users.push(r.userId);
    reactionMap.set(r.emoji, entry);
  }

  const reactionData = Array.from(reactionMap.entries()).map(([e, { count, users }]) => ({
    emoji: e,
    count,
    users,
  }));

  // Broadcast to channel
  const io = getIO();
  io?.to(`channel:${channelId}`).emit("reaction:update", {
    messageId,
    channelId,
    reactions: reactionData,
  });

  return NextResponse.json({ reactions: reactionData });
}
