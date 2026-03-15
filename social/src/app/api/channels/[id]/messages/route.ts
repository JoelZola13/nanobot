import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import { invokeAgentStreaming } from "@/lib/nanobot";

// GET /api/channels/[id]/messages — fetch messages
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { id: channelId } = await params;
  const { searchParams } = new URL(req.url);
  const cursor = searchParams.get("cursor");
  const parentId = searchParams.get("parentId");
  const limit = Math.min(parseInt(searchParams.get("limit") || "50"), 100);

  const membership = await prisma.channelMember.findUnique({
    where: { channelId_userId: { channelId, userId: session.user.id } },
  });
  if (!membership)
    return NextResponse.json({ error: "Not a member" }, { status: 403 });

  const messages = await prisma.message.findMany({
    where: { channelId, deletedAt: null, parentId: parentId || null },
    include: {
      author: {
        select: { id: true, username: true, displayName: true, avatarUrl: true, isAgent: true },
      },
      reactions: { select: { emoji: true, userId: true } },
      attachments: true,
      _count: { select: { replies: true } },
    },
    orderBy: { createdAt: "asc" },
    take: limit,
    ...(cursor ? { cursor: { id: cursor }, skip: 1 } : {}),
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
      id: msg.id, channelId: msg.channelId, content: msg.content,
      createdAt: msg.createdAt.toISOString(), isEdited: msg.isEdited,
      isPinned: msg.isPinned, parentId: msg.parentId,
      replyCount: msg._count.replies, author: msg.author,
      metadata: msg.metadata || undefined,
      reactions: Array.from(reactionMap.entries()).map(([emoji, { count, userReacted }]) => ({ emoji, count, userReacted })),
      attachments: msg.attachments.map((a) => ({ id: a.id, fileName: a.fileName, mimeType: a.mimeType, url: a.url, width: a.width, height: a.height })),
    };
  });

  return NextResponse.json({
    messages: formatted,
    nextCursor: messages.length === limit ? messages[messages.length - 1].id : null,
  });
}

function formatMessage(msg: {
  id: string; channelId: string; content: string; createdAt: Date;
  isEdited: boolean; isPinned: boolean; parentId: string | null;
  metadata?: unknown;
  author: { id: string; username: string; displayName: string; avatarUrl: string | null; isAgent: boolean };
  attachments?: { id: string; fileName: string; mimeType: string; url: string; s3Key: string; fileSize: number; width: number | null; height: number | null }[];
}) {
  return {
    id: msg.id, channelId: msg.channelId, content: msg.content,
    createdAt: msg.createdAt.toISOString(), isEdited: msg.isEdited,
    isPinned: msg.isPinned, parentId: msg.parentId,
    metadata: msg.metadata || undefined,
    replyCount: 0, author: msg.author, reactions: [],
    attachments: (msg.attachments || []).map((a) => ({
      id: a.id, fileName: a.fileName, mimeType: a.mimeType, url: a.url,
      width: a.width, height: a.height,
    })),
  };
}

function getIO() {
  return (globalThis as Record<string, unknown>).__socketio as
    | { to: (room: string) => { emit: (event: string, data: unknown) => void } }
    | undefined;
}

// POST /api/channels/[id]/messages — send a message
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { id: channelId } = await params;
  const body = await req.json();
  const { content, parentId, attachments: attachmentData, metadata } = body;

  if (!content?.trim())
    return NextResponse.json({ error: "Message content required" }, { status: 400 });

  const membership = await prisma.channelMember.findUnique({
    where: { channelId_userId: { channelId, userId: session.user.id } },
  });
  if (!membership)
    return NextResponse.json({ error: "Not a member" }, { status: 403 });

  const message = await prisma.message.create({
    data: {
      channelId,
      authorId: session.user.id,
      content: content.trim(),
      parentId: parentId || null,
      metadata: metadata || undefined,
      ...(attachmentData?.length ? {
        attachments: {
          create: attachmentData.map((a: { s3Key: string; url: string; fileName: string; fileSize: number; mimeType: string }) => ({
            s3Key: a.s3Key,
            url: a.url,
            fileName: a.fileName,
            fileSize: a.fileSize,
            mimeType: a.mimeType,
          })),
        },
      } : {}),
    },
    include: {
      author: { select: { id: true, username: true, displayName: true, avatarUrl: true, isAgent: true } },
      attachments: true,
    },
  });

  await prisma.channel.update({ where: { id: channelId }, data: { updatedAt: new Date() } });

  const userMessage = formatMessage(message);

  // --- Agent interaction ---
  const channel = await prisma.channel.findUnique({
    where: { id: channelId },
    include: {
      members: {
        include: {
          user: { select: { id: true, username: true, displayName: true, avatarUrl: true, isAgent: true, agentModel: true } },
        },
      },
    },
  });

  const agentsToRespond: { id: string; username: string; displayName: string; avatarUrl: string | null; isAgent: boolean; agentModel: string | null }[] = [];

  if (channel?.type === "DM") {
    const agentMember = channel.members.find((m) => m.user.isAgent && m.userId !== session.user!.id);
    if (agentMember) agentsToRespond.push(agentMember.user);
  } else {
    const mentionPattern = /@([\w-]+)/g;
    let match;
    while ((match = mentionPattern.exec(content)) !== null) {
      const agentMember = channel?.members.find((m) => m.user.isAgent && m.user.username === match![1]);
      if (agentMember && !agentsToRespond.some((a) => a.id === agentMember.user.id)) {
        agentsToRespond.push(agentMember.user);
      }
    }
  }

  // Skip agent triggering for voice messages — agent will be triggered after transcription
  const isVoiceMessage = metadata?.type === "voice";

  if (agentsToRespond.length > 0 && !isVoiceMessage) {
    const recentMessages = await prisma.message.findMany({
      where: { channelId, deletedAt: null, parentId: null },
      include: {
        author: { select: { displayName: true, isAgent: true } },
        attachments: { select: { fileName: true, mimeType: true, url: true } },
      },
      orderBy: { createdAt: "desc" },
      take: 20,
    });

    const conversationHistory = recentMessages.reverse().map((m) => {
      let text = m.content;
      const meta = m.metadata as Record<string, unknown> | null;

      // Include voice transcription in the message content for agents
      if (meta?.type === "voice" && meta?.transcription) {
        text = `🎙️ Voice message (transcription): "${meta.transcription}"`;
      } else if (meta?.transcription) {
        text += `\n[Transcription: "${meta.transcription}"]`;
      }

      // Include file attachment info
      if (m.attachments.length > 0 && meta?.type !== "voice") {
        const fileList = m.attachments.map((a) => `${a.fileName} (${a.mimeType})`).join(", ");
        text += `\n[Attached files: ${fileList}]`;
      }

      return {
        role: (m.author.isAgent ? "assistant" : "user") as "user" | "assistant",
        content: m.author.isAgent ? text : `[${m.author.displayName}]: ${text}`,
      };
    });

    for (const agent of agentsToRespond) {
      triggerAgentResponseStreaming(channelId, agent, conversationHistory)
        .catch((err) => console.error(`Agent ${agent.username} failed:`, err));
    }
  }

  return NextResponse.json(userMessage, { status: 201 });
}

async function triggerAgentResponseStreaming(
  channelId: string,
  agent: { id: string; username: string; displayName: string; avatarUrl: string | null; isAgent: boolean; agentModel: string | null },
  conversationHistory: { role: "user" | "assistant"; content: string }[],
) {
  const model = agent.agentModel || "agent/auto";
  const io = getIO();

  // Emit "agent is thinking" event
  io?.to(`channel:${channelId}`).emit("agent:activity", {
    channelId,
    agent: { id: agent.id, displayName: agent.displayName, username: agent.username },
    type: "thinking",
    text: `${agent.displayName} is thinking...`,
  });

  try {
    const { content: reply, toolsUsed } = await invokeAgentStreaming(
      model,
      conversationHistory,
      (progressText: string) => {
        // Parse progress hints and broadcast as activity
        // Progress format: "⏳ Calling tool: tool_name" or "⏳ Delegating to agent/xxx"
        let activityType = "working";
        const text = progressText.replace(/⏳\s*/g, "").trim();
        let delegatedTo: string | undefined;

        if (text.toLowerCase().includes("delegat")) {
          activityType = "delegating";
          // Extract target agent name
          const delegateMatch = text.match(/agent[\/](\w+)/i);
          delegatedTo = delegateMatch?.[1]?.replace(/_/g, " ");
        } else if (text.toLowerCase().includes("calling tool")) {
          activityType = "tool_call";
        }

        io?.to(`channel:${channelId}`).emit("agent:activity", {
          channelId,
          agent: { id: agent.id, displayName: agent.displayName, username: agent.username },
          type: activityType,
          text,
          delegatedTo,
          timestamp: new Date().toISOString(),
        });
      },
    );

    if (!reply?.trim()) {
      io?.to(`channel:${channelId}`).emit("agent:activity", {
        channelId,
        agent: { id: agent.id, displayName: agent.displayName, username: agent.username },
        type: "done",
        text: "Finished (no response)",
        toolsUsed,
      });
      return;
    }

    // Save agent's reply
    const agentMessage = await prisma.message.create({
      data: { channelId, authorId: agent.id, content: reply.trim() },
      include: {
        author: { select: { id: true, username: true, displayName: true, avatarUrl: true, isAgent: true } },
      },
    });

    await prisma.channel.update({ where: { id: channelId }, data: { updatedAt: new Date() } });

    // Broadcast final message
    io?.to(`channel:${channelId}`).emit("message:new", formatMessage(agentMessage));

    // Broadcast completion with tools used
    io?.to(`channel:${channelId}`).emit("agent:activity", {
      channelId,
      agent: { id: agent.id, displayName: agent.displayName, username: agent.username },
      type: "done",
      text: "Response delivered",
      toolsUsed,
    });
  } catch (err) {
    console.error(`Agent ${agent.username} (${model}) error:`, err);
    io?.to(`channel:${channelId}`).emit("agent:activity", {
      channelId,
      agent: { id: agent.id, displayName: agent.displayName, username: agent.username },
      type: "error",
      text: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
    });
  }
}
