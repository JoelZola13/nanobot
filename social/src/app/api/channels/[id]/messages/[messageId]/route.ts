import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import { getIO } from "@/lib/socketServer";

// PATCH /api/channels/[id]/messages/[messageId] — edit message
export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ id: string; messageId: string }> },
) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { id: channelId, messageId } = await params;
  const { content } = await req.json();

  if (!content?.trim())
    return NextResponse.json({ error: "Content required" }, { status: 400 });

  const message = await prisma.message.findUnique({ where: { id: messageId } });
  if (!message || message.channelId !== channelId)
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  if (message.authorId !== session.user.id)
    return NextResponse.json({ error: "Not your message" }, { status: 403 });

  const updated = await prisma.message.update({
    where: { id: messageId },
    data: { content: content.trim(), isEdited: true },
    include: {
      author: { select: { id: true, username: true, displayName: true, avatarUrl: true, isAgent: true } },
    },
  });

  const io = getIO();
  io?.to(`channel:${channelId}`).emit("message:edit", {
    id: updated.id,
    channelId,
    content: updated.content,
    isEdited: true,
  });

  return NextResponse.json({
    id: updated.id, channelId, content: updated.content,
    isEdited: true, isPinned: updated.isPinned,
    createdAt: updated.createdAt.toISOString(),
    author: updated.author,
  });
}

// DELETE /api/channels/[id]/messages/[messageId] — soft delete
export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string; messageId: string }> },
) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { id: channelId, messageId } = await params;

  const message = await prisma.message.findUnique({ where: { id: messageId } });
  if (!message || message.channelId !== channelId)
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  if (message.authorId !== session.user.id)
    return NextResponse.json({ error: "Not your message" }, { status: 403 });

  await prisma.message.update({
    where: { id: messageId },
    data: { deletedAt: new Date() },
  });

  const io = getIO();
  io?.to(`channel:${channelId}`).emit("message:delete", { id: messageId, channelId });

  return NextResponse.json({ ok: true });
}
