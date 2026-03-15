"use client";

import { useState, useEffect, useCallback } from "react";
import MessageList from "./MessageList";
import MessageInput from "./MessageInput";
import AgentActivity, { type ActivityEvent } from "./AgentActivity";
import ThreadPanel from "./ThreadPanel";
import { useSocket } from "@/components/providers/SocketProvider";
import { useUnreadStore } from "@/stores/unreadStore";
import { useReadReceipts } from "@/hooks/useReadReceipts";
import type { MessageData } from "@/types";

interface ChannelViewProps {
  channelId: string;
  channelName: string;
  initialMessages: MessageData[];
  currentUserId: string;
}

export default function ChannelView({
  channelId,
  channelName,
  initialMessages,
  currentUserId,
}: ChannelViewProps) {
  const [messages, setMessages] = useState<MessageData[]>(initialMessages);
  const [sending, setSending] = useState(false);
  const [typingUsers, setTypingUsers] = useState<Map<string, string>>(new Map());
  const [agentActivities, setAgentActivities] = useState<ActivityEvent[]>([]);
  const [openThreadId, setOpenThreadId] = useState<string | null>(null);
  // Read receipts: map of userId -> messageId they've read up to
  const [readReceipts, setReadReceipts] = useState<Map<string, string>>(new Map());
  const socket = useSocket();
  const setActiveChannel = useUnreadStore((s) => s.setActive);
  const clearUnread = useUnreadStore((s) => s.clear);

  // Auto-send read receipts when viewing messages
  const lastMessageId = messages.length > 0 ? messages[messages.length - 1].id : null;
  useReadReceipts(channelId, lastMessageId, currentUserId);

  // Mark this channel as active + clear unreads on mount/channel change
  useEffect(() => {
    setActiveChannel(channelId);
    clearUnread(channelId);
    return () => setActiveChannel(null);
  }, [channelId, setActiveChannel, clearUnread]);

  useEffect(() => {
    if (!socket) return;

    socket.emit("join:channel", channelId);

    const handleNewMessage = (msg: MessageData) => {
      if (msg.channelId === channelId) {
        setMessages((prev) => {
          if (prev.some((m) => m.id === msg.id)) return prev;
          return [...prev, msg];
        });
        if (msg.author.isAgent) {
          setTimeout(() => {
            setAgentActivities((prev) =>
              prev.filter((a) => a.agent.id !== msg.author.id)
            );
          }, 3000);
        }
        setTypingUsers((prev) => {
          const next = new Map(prev);
          next.delete(msg.author.id);
          return next;
        });
      }
    };

    const handleTypingStart = ({ channelId: cid, user }: { channelId: string; user: { id: string; name: string } }) => {
      if (cid === channelId && user.id !== currentUserId) {
        setTypingUsers((prev) => new Map(prev).set(user.id, user.name));
      }
    };

    const handleTypingStop = ({ channelId: cid, userId }: { channelId: string; userId: string }) => {
      if (cid === channelId) {
        setTypingUsers((prev) => { const next = new Map(prev); next.delete(userId); return next; });
      }
    };

    const handleAgentActivity = (event: ActivityEvent) => {
      if (event.channelId === channelId) {
        setAgentActivities((prev) => {
          if (event.type === "done" || event.type === "error") {
            const filtered = prev.filter((a) => a.agent.id !== event.agent.id);
            return [...filtered, event];
          }
          const otherAgents = prev.filter((a) => a.agent.id !== event.agent.id);
          const thisAgent = prev.filter((a) => a.agent.id === event.agent.id);
          const updated = [...thisAgent.slice(-9), event];
          return [...otherAgents, ...updated];
        });
      }
    };

    const handleMessageEdit = ({ id, content, isEdited }: { id: string; channelId: string; content: string; isEdited: boolean }) => {
      setMessages((prev) => prev.map((m) => m.id === id ? { ...m, content, isEdited } : m));
    };

    const handleMessageDelete = ({ id }: { id: string }) => {
      setMessages((prev) => prev.filter((m) => m.id !== id));
    };

    const handleReactionUpdate = ({ messageId, reactions }: { messageId: string; channelId: string; reactions: { emoji: string; count: number; users: string[] }[] }) => {
      setMessages((prev) => prev.map((m) => {
        if (m.id !== messageId) return m;
        return {
          ...m,
          reactions: reactions.map((r) => ({
            emoji: r.emoji,
            count: r.count,
            userReacted: r.users.includes(currentUserId),
          })),
        };
      }));
    };

    const handleMessagePin = ({ messageId, isPinned }: { messageId: string; channelId: string; isPinned: boolean }) => {
      setMessages((prev) => prev.map((m) => m.id === messageId ? { ...m, isPinned } : m));
    };

    const handleTranscription = ({ messageId, transcription }: { messageId: string; channelId: string; transcription: string }) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === messageId
            ? { ...m, metadata: { ...m.metadata, transcription } }
            : m
        )
      );
    };

    const handleReadUpdate = ({ userId: uid, messageId }: { channelId: string; userId: string; messageId: string }) => {
      if (uid !== currentUserId) {
        setReadReceipts((prev) => new Map(prev).set(uid, messageId));
      }
    };

    // Fetch initial read receipts
    fetch(`/api/channels/${channelId}/read`)
      .then((r) => r.ok ? r.json() : [])
      .then((receipts: { userId: string; messageId: string }[]) => {
        const map = new Map<string, string>();
        for (const r of receipts) {
          if (r.userId !== currentUserId) {
            map.set(r.userId, r.messageId);
          }
        }
        setReadReceipts(map);
      })
      .catch(() => {});

    socket.on("message:new", handleNewMessage);
    socket.on("typing:start", handleTypingStart);
    socket.on("typing:stop", handleTypingStop);
    socket.on("agent:activity", handleAgentActivity);
    socket.on("message:edit", handleMessageEdit);
    socket.on("message:delete", handleMessageDelete);
    socket.on("reaction:update", handleReactionUpdate);
    socket.on("message:pin", handleMessagePin);
    socket.on("read:update", handleReadUpdate);
    socket.on("message:transcription", handleTranscription);

    return () => {
      socket.off("message:new", handleNewMessage);
      socket.off("typing:start", handleTypingStart);
      socket.off("typing:stop", handleTypingStop);
      socket.off("agent:activity", handleAgentActivity);
      socket.off("message:edit", handleMessageEdit);
      socket.off("message:delete", handleMessageDelete);
      socket.off("reaction:update", handleReactionUpdate);
      socket.off("message:pin", handleMessagePin);
      socket.off("read:update", handleReadUpdate);
      socket.off("message:transcription", handleTranscription);
      socket.emit("leave:channel", channelId);
    };
  }, [socket, channelId, currentUserId]);

  useEffect(() => {
    setMessages(initialMessages);
    setAgentActivities([]);
    setOpenThreadId(null);
  }, [channelId, initialMessages]);

  const handleSend = async (content: string) => {
    setSending(true);
    try {
      const res = await fetch(`/api/channels/${channelId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
      if (res.ok) {
        const msg = await res.json();
        setMessages((prev) => [...prev, msg]);
        socket?.emit("message:send", msg);
        socket?.emit("typing:stop", { channelId, userId: currentUserId });
      }
    } finally {
      setSending(false);
    }
  };

  const handleTyping = useCallback(() => {
    socket?.emit("typing:start", {
      channelId,
      user: { id: currentUserId, name: "" },
    });
  }, [socket, channelId, currentUserId]);

  const handleReaction = async (messageId: string, emoji: string) => {
    await fetch(`/api/channels/${channelId}/messages/${messageId}/reactions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ emoji }),
    });
  };

  const handleEdit = async (messageId: string, content: string) => {
    const res = await fetch(`/api/channels/${channelId}/messages/${messageId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    if (res.ok) {
      setMessages((prev) => prev.map((m) => m.id === messageId ? { ...m, content, isEdited: true } : m));
    }
  };

  const handleDelete = async (messageId: string) => {
    const res = await fetch(`/api/channels/${channelId}/messages/${messageId}`, {
      method: "DELETE",
    });
    if (res.ok) {
      setMessages((prev) => prev.filter((m) => m.id !== messageId));
    }
  };

  const handlePin = async (messageId: string, isPinned: boolean) => {
    const res = await fetch(`/api/channels/${channelId}/pins`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messageId }),
    });
    if (res.ok) {
      setMessages((prev) => prev.map((m) => m.id === messageId ? { ...m, isPinned: !isPinned } : m));
    }
  };

  const handleOpenThread = (messageId: string) => {
    setOpenThreadId(messageId);
  };

  const handleVoiceSend = async (audioBlob: Blob, duration: number) => {
    // Upload audio to S3
    const formData = new FormData();
    formData.append("file", audioBlob, "voice-message.webm");
    const uploadRes = await fetch("/api/upload", { method: "POST", body: formData });
    if (!uploadRes.ok) return;
    const { s3Key, url, fileName, fileSize, mimeType } = await uploadRes.json();

    // Send message with audio attachment + voice metadata
    const res = await fetch(`/api/channels/${channelId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: "🎙️ Voice message",
        attachments: [{ s3Key, url, fileName, fileSize, mimeType }],
        metadata: { type: "voice", duration },
      }),
    });
    if (res.ok) {
      const msg = await res.json();
      setMessages((prev) => [...prev, msg]);
      socket?.emit("message:send", msg);

      // Auto-transcribe in the background
      console.log("[VOICE] Triggering transcription for message:", msg.id, "audioUrl:", url);
      fetch("/api/voice/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messageId: msg.id, audioUrl: url }),
      })
        .then((r) => r.ok ? r.json() : null)
        .then((data) => {
          if (data?.transcription) {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === msg.id
                  ? { ...m, metadata: { ...m.metadata, transcription: data.transcription } }
                  : m
              )
            );
          }
        })
        .catch((err) => console.error("[VOICE] Transcription request failed:", err));
    }
  };

  const handleFileUpload = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    const uploadRes = await fetch("/api/upload", { method: "POST", body: formData });
    if (!uploadRes.ok) return;
    const { s3Key, url, fileName, fileSize, mimeType } = await uploadRes.json();

    const res = await fetch(`/api/channels/${channelId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: `📎 ${fileName}`,
        attachments: [{ s3Key, url, fileName, fileSize, mimeType }],
      }),
    });
    if (res.ok) {
      const msg = await res.json();
      setMessages((prev) => [...prev, msg]);
      socket?.emit("message:send", msg);
    }
  };

  const typingNames = Array.from(typingUsers.values()).filter(Boolean);

  const threadParent = openThreadId ? messages.find((m) => m.id === openThreadId) : null;

  return (
    <div className="flex flex-1 min-h-0">
      <div className="flex flex-col flex-1 min-w-0">
        <MessageList
          messages={messages}
          currentUserId={currentUserId}
          readReceipts={readReceipts}
          onReaction={handleReaction}
          onEdit={handleEdit}
          onDelete={handleDelete}
          onPin={handlePin}
          onOpenThread={handleOpenThread}
        />
        <AgentActivity activities={agentActivities} />
        {typingNames.length > 0 && (
          <div className="px-4 py-1 text-2xs text-text-muted animate-pulse">
            {typingNames.join(", ")} {typingNames.length === 1 ? "is" : "are"} typing...
          </div>
        )}
        <MessageInput
          channelId={channelId}
          channelName={channelName}
          onSend={handleSend}
          onTyping={handleTyping}
          disabled={sending}
          onVoiceSend={handleVoiceSend}
          onFileUpload={handleFileUpload}
        />
      </div>

      {openThreadId && threadParent && (
        <ThreadPanel
          channelId={channelId}
          parentMessage={threadParent}
          currentUserId={currentUserId}
          onClose={() => setOpenThreadId(null)}
        />
      )}
    </div>
  );
}
