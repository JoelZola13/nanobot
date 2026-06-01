import { NextRequest, NextResponse } from "next/server";
import { ZodError } from "zod";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import { formatMessageForClient } from "@/lib/messageFormat";
import { getIO } from "@/lib/socketServer";
import {
  EmailReplyConfigError,
  EmailReplyRequest,
  EmailReplyRequestSchema,
  getImportedEmailReplyTarget,
  sendImportedEmailReply,
} from "@/lib/emailReply";

export const runtime = "nodejs";

// POST /api/email-import/[messageId]/reply - send an email reply from Messages.
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ messageId: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { messageId } = await params;
  let parsed: EmailReplyRequest;
  try {
    parsed = EmailReplyRequestSchema.parse(await req.json());
  } catch (error) {
    const details = error instanceof ZodError ? error.issues : undefined;
    return NextResponse.json(
      { error: "Invalid email reply payload", details },
      { status: 400 },
    );
  }

  const importedMessage = await prisma.message.findUnique({
    where: { id: messageId },
    select: {
      id: true,
      channelId: true,
      authorId: true,
      metadata: true,
      deletedAt: true,
      channel: {
        select: {
          isArchived: true,
        },
      },
    },
  });

  if (!importedMessage || importedMessage.deletedAt) {
    return NextResponse.json({ error: "Imported email not found" }, { status: 404 });
  }

  const membership = await prisma.channelMember.findUnique({
    where: {
      channelId_userId: {
        channelId: importedMessage.channelId,
        userId: session.user.id,
      },
    },
  });
  if (!membership) {
    return NextResponse.json({ error: "Not a member" }, { status: 403 });
  }
  if (importedMessage.channel.isArchived) {
    return NextResponse.json({ error: "Channel is archived" }, { status: 403 });
  }
  if (importedMessage.authorId !== session.user.id) {
    return NextResponse.json(
      { error: "Only the person who imported this email can reply from it" },
      { status: 403 },
    );
  }

  const target = getImportedEmailReplyTarget(importedMessage.metadata);
  if (!target) {
    return NextResponse.json(
      { error: "Imported email does not have a replyable sender" },
      { status: 400 },
    );
  }

  let sent: Awaited<ReturnType<typeof sendImportedEmailReply>>;
  try {
    sent = await sendImportedEmailReply(target, parsed.content);
  } catch (error) {
    if (error instanceof EmailReplyConfigError) {
      return NextResponse.json({ error: error.message }, { status: 503 });
    }
    console.error("[EMAIL_REPLY] SMTP send failed:", error);
    return NextResponse.json(
      { error: "Email reply could not be sent" },
      { status: 502 },
    );
  }

  const metadata = {
    type: "email_reply",
    emailReply: {
      sourceMessageId: importedMessage.id,
      provider: target.provider,
      sourceUrl: target.sourceUrl,
      to: sent.to,
      subject: sent.subject,
      sentAt: sent.sentAt,
      inReplyTo: sent.inReplyTo,
      messageId: sent.messageId,
    },
  };

  const reply = await prisma.message.create({
    data: {
      channelId: importedMessage.channelId,
      authorId: session.user.id,
      parentId: importedMessage.id,
      content: parsed.content,
      metadata,
    },
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
      attachments: true,
    },
  });

  await prisma.channel.update({
    where: { id: importedMessage.channelId },
    data: { updatedAt: new Date() },
  });

  const formatted = formatMessageForClient(reply, session.user.id);
  getIO()?.to(`channel:${importedMessage.channelId}`).emit("message:new", formatted);

  return NextResponse.json(
    {
      message: formatted,
      delivery: {
        to: sent.to,
        subject: sent.subject,
        sentAt: sent.sentAt,
      },
    },
    { status: 201 },
  );
}
