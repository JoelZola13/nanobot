import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import { getIO } from "@/lib/socketServer";
import {
  buildMessageDeletionAudit,
  getMessageRemovalMode,
  normalizeDeletionReason,
  withMessageDeletionAudit,
} from "@/lib/messageModeration";

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
  req: NextRequest,
  { params }: { params: Promise<{ id: string; messageId: string }> },
) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { id: channelId, messageId } = await params;

  const message = await prisma.message.findUnique({
    where: { id: messageId },
    include: {
      channel: {
        select: {
          isArchived: true,
          members: {
            where: { userId: session.user.id },
            select: { role: true },
          },
        },
      },
    },
  });
  if (!message || message.channelId !== channelId)
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  if (message.channel.isArchived)
    return NextResponse.json({ error: "Channel is archived" }, { status: 403 });
  if (message.deletedAt)
    return NextResponse.json({ error: "Message already removed" }, { status: 409 });

  const channelRole = message.channel.members[0]?.role;
  const removalMode = getMessageRemovalMode(
    session.user,
    message.authorId,
    channelRole,
  );
  if (!removalMode)
    return NextResponse.json(
      { error: "Channel admin access required" },
      { status: 403 },
    );

  const body = await readOptionalJson(req);
  const deletedAt = new Date();
  const audit = buildMessageDeletionAudit({
    user: session.user,
    mode: removalMode,
    reason: normalizeDeletionReason(body.reason),
    deletedAt,
  });

  await prisma.message.update({
    where: { id: messageId },
    data: {
      deletedAt,
      metadata: withMessageDeletionAudit(message.metadata, audit),
    },
  });

  const io = getIO();
  io?.to(`channel:${channelId}`).emit("message:delete", {
    id: messageId,
    channelId,
    deletedById: audit.actorId,
    mode: audit.mode,
  });

  return NextResponse.json({ ok: true, audit });
}

async function readOptionalJson(req: NextRequest) {
  const text = await req.text().catch(() => "");
  if (!text.trim()) return {} as Record<string, unknown>;

  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return {};
  }
}
