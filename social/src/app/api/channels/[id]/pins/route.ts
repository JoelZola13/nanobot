import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";

function getIO() {
  return (globalThis as Record<string, unknown>).__socketio as
    | { to: (room: string) => { emit: (event: string, data: unknown) => void } }
    | undefined;
}

// GET /api/channels/[id]/pins — list pinned messages
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { id: channelId } = await params;

  const pins = await prisma.message.findMany({
    where: { channelId, isPinned: true, deletedAt: null },
    include: {
      author: { select: { id: true, username: true, displayName: true, avatarUrl: true, isAgent: true } },
    },
    orderBy: { updatedAt: "desc" },
  });

  return NextResponse.json({
    pins: pins.map((p) => ({
      id: p.id, channelId: p.channelId, content: p.content,
      createdAt: p.createdAt.toISOString(),
      author: p.author,
    })),
  });
}

// POST /api/channels/[id]/pins — toggle pin
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { id: channelId } = await params;
  const { messageId } = await req.json();

  const message = await prisma.message.findUnique({ where: { id: messageId } });
  if (!message || message.channelId !== channelId)
    return NextResponse.json({ error: "Not found" }, { status: 404 });

  const newPinned = !message.isPinned;
  await prisma.message.update({
    where: { id: messageId },
    data: { isPinned: newPinned },
  });

  const io = getIO();
  io?.to(`channel:${channelId}`).emit("message:pin", {
    messageId,
    channelId,
    isPinned: newPinned,
  });

  return NextResponse.json({ isPinned: newPinned });
}
