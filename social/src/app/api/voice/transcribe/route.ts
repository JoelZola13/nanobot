import { NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import { invokeAgentStreaming } from "@/lib/nanobot";
import { getFromS3, S3_BUCKET } from "@/lib/s3";
import { getIO } from "@/lib/socketServer";

// POST /api/voice/transcribe — transcribe a voice message via Groq Whisper
export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { messageId, audioUrl } = await request.json();
  console.log("[TRANSCRIBE] Request received:", { messageId, audioUrl: audioUrl?.substring(0, 80) });
  if (!messageId || !audioUrl) {
    console.log("[TRANSCRIBE] Missing params:", { messageId, audioUrl });
    return NextResponse.json({ error: "messageId and audioUrl required" }, { status: 400 });
  }

  const groqKey = process.env.GROQ_API_KEY;
  if (!groqKey) {
    console.log("[TRANSCRIBE] GROQ_API_KEY not configured!");
    return NextResponse.json({ error: "GROQ_API_KEY not configured" }, { status: 500 });
  }

  try {
    // Download the audio file from S3 using authenticated S3 client
    console.log("[TRANSCRIBE] Fetching audio from S3...");
    // Extract S3 key from the URL: http://localhost:8333/social/uploads/... -> uploads/...
    const s3Key = audioUrl.replace(new RegExp(`^https?://[^/]+/${S3_BUCKET}/`), "");
    console.log("[TRANSCRIBE] S3 key:", s3Key);
    let audioBuffer: Uint8Array;
    try {
      const buf = await getFromS3(s3Key);
      audioBuffer = new Uint8Array(buf);
      console.log("[TRANSCRIBE] Audio downloaded from S3, size:", audioBuffer.byteLength);
    } catch (s3Err) {
      console.error("[TRANSCRIBE] S3 download failed:", s3Err);
      return NextResponse.json({ error: "Failed to fetch audio from S3" }, { status: 500 });
    }

    // Send to Groq Whisper API
    const formData = new FormData();
    formData.append("file", new Blob([audioBuffer] as BlobPart[], { type: "audio/webm" }), "voice.webm");
    formData.append("model", "whisper-large-v3");
    formData.append("language", "en");
    formData.append("response_format", "json");

    const groqRes = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${groqKey}`,
      },
      body: formData,
    });

    if (!groqRes.ok) {
      const errText = await groqRes.text();
      console.error("Groq transcription error:", errText);
      return NextResponse.json({ error: "Transcription failed" }, { status: 500 });
    }

    const { text } = await groqRes.json();

    if (!text || text.trim().length === 0) {
      return NextResponse.json({ transcription: "" });
    }

    // Update message metadata with transcription
    const message = await prisma.message.findUnique({ where: { id: messageId } });
    if (!message) {
      return NextResponse.json({ error: "Message not found" }, { status: 404 });
    }

    const existingMetadata = (message.metadata as Record<string, unknown>) || {};
    await prisma.message.update({
      where: { id: messageId },
      data: {
        metadata: { ...existingMetadata, transcription: text.trim() },
      },
    });

    // Broadcast transcription update via socket
    const io = getIO();

    if (io) {
      io.to(`channel:${message.channelId}`).emit("message:transcription", {
        messageId,
        channelId: message.channelId,
        transcription: text.trim(),
      });
    }

    // After transcription, trigger agent responses if applicable
    triggerAgentsForVoice(message.channelId, message.authorId, text.trim()).catch((err) =>
      console.error("Agent trigger after transcription failed:", err)
    );

    return NextResponse.json({ transcription: text.trim() });
  } catch (err) {
    console.error("Transcription error:", err);
    return NextResponse.json({ error: "Transcription failed" }, { status: 500 });
  }
}


async function triggerAgentsForVoice(channelId: string, authorId: string, transcription: string) {
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
  if (!channel) return;

  const agentsToRespond: { id: string; username: string; displayName: string; avatarUrl: string | null; isAgent: boolean; agentModel: string | null }[] = [];

  if (channel.type === "DM") {
    const agentMember = channel.members.find((m) => m.user.isAgent && m.userId !== authorId);
    if (agentMember) agentsToRespond.push(agentMember.user);
  } else {
    // Check for @mentions in the transcription
    const mentionPattern = /@([\w-]+)/g;
    let match;
    while ((match = mentionPattern.exec(transcription)) !== null) {
      const agentMember = channel.members.find((m) => m.user.isAgent && m.user.username === match![1]);
      if (agentMember && !agentsToRespond.some((a) => a.id === agentMember.user.id)) {
        agentsToRespond.push(agentMember.user);
      }
    }
  }

  if (agentsToRespond.length === 0) return;

  // Build conversation history including the transcription
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
    let msgText = m.content;
    const meta = m.metadata as Record<string, unknown> | null;

    if (meta?.type === "voice" && meta?.transcription) {
      msgText = `🎙️ Voice message (transcription): "${meta.transcription}"`;
    } else if (meta?.transcription) {
      msgText += `\n[Transcription: "${meta.transcription}"]`;
    }

    if (m.attachments.length > 0 && meta?.type !== "voice") {
      const fileList = m.attachments.map((a) => `${a.fileName} (${a.mimeType})`).join(", ");
      msgText += `\n[Attached files: ${fileList}]`;
    }

    return {
      role: (m.author.isAgent ? "assistant" : "user") as "user" | "assistant",
      content: m.author.isAgent ? msgText : `[${m.author.displayName}]: ${msgText}`,
    };
  });

  const io = getIO();

  for (const agent of agentsToRespond) {
    const model = agent.agentModel || "agent/auto";

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
          let activityType = "working";
          const pText = progressText.replace(/⏳\s*/g, "").trim();
          let delegatedTo: string | undefined;

          if (pText.toLowerCase().includes("delegat")) {
            activityType = "delegating";
            const delegateMatch = pText.match(/agent[/](\w+)/i);
            delegatedTo = delegateMatch?.[1]?.replace(/_/g, " ");
          } else if (pText.toLowerCase().includes("calling tool")) {
            activityType = "tool_call";
          }

          io?.to(`channel:${channelId}`).emit("agent:activity", {
            channelId,
            agent: { id: agent.id, displayName: agent.displayName, username: agent.username },
            type: activityType,
            text: pText,
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

      const agentMessage = await prisma.message.create({
        data: { channelId, authorId: agent.id, content: reply.trim() },
        include: {
          author: { select: { id: true, username: true, displayName: true, avatarUrl: true, isAgent: true } },
        },
      });

      await prisma.channel.update({ where: { id: channelId }, data: { updatedAt: new Date() } });

      io?.to(`channel:${channelId}`).emit("message:new", {
        id: agentMessage.id, channelId: agentMessage.channelId, content: agentMessage.content,
        createdAt: agentMessage.createdAt.toISOString(), isEdited: agentMessage.isEdited,
        isPinned: agentMessage.isPinned, parentId: agentMessage.parentId,
        metadata: agentMessage.metadata || undefined,
        replyCount: 0, author: agentMessage.author, reactions: [], attachments: [],
      });

      io?.to(`channel:${channelId}`).emit("agent:activity", {
        channelId,
        agent: { id: agent.id, displayName: agent.displayName, username: agent.username },
        type: "done",
        text: "Response delivered",
        toolsUsed,
      });
    } catch (err) {
      console.error(`Agent ${agent.username} error after voice transcription:`, err);
      io?.to(`channel:${channelId}`).emit("agent:activity", {
        channelId,
        agent: { id: agent.id, displayName: agent.displayName, username: agent.username },
        type: "error",
        text: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
      });
    }
  }
}
