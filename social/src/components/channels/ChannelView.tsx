"use client";

import { useState, useEffect, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import MessageList from "./MessageList";
import MessageInput from "./MessageInput";
import AgentActivity, { type ActivityEvent } from "./AgentActivity";
import ThreadPanel from "./ThreadPanel";
import { useSocket } from "@/components/providers/SocketProvider";
import { useUnreadStore } from "@/stores/unreadStore";
import { useReadReceipts } from "@/hooks/useReadReceipts";
import { apiUrl } from "@/lib/apiUrl";
import type { MessageData } from "@/types";
import type { MessageEmptyState } from "./MessageList";

interface ChannelViewProps {
  channelId: string;
  channelName: string;
  initialMessages: MessageData[];
  initialOldestMessageCursor?: string | null;
  initialUnreadAfter?: string | null;
  currentUserId: string;
  placeholder?: string;
  emptyState: MessageEmptyState;
  canManageMessages?: boolean;
}

const toThreadParticipant = (author: MessageData["author"]) => ({
  id: author.id,
  displayName: author.displayName,
  avatarUrl: author.avatarUrl,
  isAgent: author.isAgent,
});

const addThreadReplyPreview = (
  message: MessageData,
  reply: MessageData,
): MessageData => {
  if (message.threadPreview?.latestReply.id === reply.id) return message;

  const replyParticipant = toThreadParticipant(reply.author);
  const existingParticipants = message.threadPreview?.participants || [];
  const participants = [
    replyParticipant,
    ...existingParticipants.filter(
      (participant) => participant.id !== replyParticipant.id,
    ),
  ].slice(0, 3);

  return {
    ...message,
    replyCount: (message.replyCount || 0) + 1,
    threadPreview: {
      participants,
      latestReply: {
        id: reply.id,
        content: reply.content,
        createdAt: reply.createdAt,
        author: replyParticipant,
      },
    },
  };
};

async function apiErrorMessage(res: Response, fallback: string) {
  const data = (await res.json().catch(() => null)) as {
    error?: string;
  } | null;
  return data?.error || fallback;
}

export default function ChannelView({
  channelId,
  channelName,
  initialMessages,
  initialOldestMessageCursor = null,
  initialUnreadAfter = null,
  currentUserId,
  placeholder,
  emptyState,
  canManageMessages = false,
}: ChannelViewProps) {
  const [messages, setMessages] = useState<MessageData[]>(initialMessages);
  const [sending, setSending] = useState(false);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [oldestMessageCursor, setOldestMessageCursor] = useState<string | null>(
    initialOldestMessageCursor,
  );
  const [olderLoadError, setOlderLoadError] = useState<string | null>(null);
  const [autoScrollKey, setAutoScrollKey] = useState(0);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const [typingUsers, setTypingUsers] = useState<Map<string, string>>(
    new Map(),
  );
  const [agentActivities, setAgentActivities] = useState<ActivityEvent[]>([]);
  const [openThreadId, setOpenThreadId] = useState<string | null>(null);
  // Read receipts: map of userId -> messageId they've read up to
  const [readReceipts, setReadReceipts] = useState<Map<string, string>>(
    new Map(),
  );
  const searchParams = useSearchParams();
  const highlightedMessageId = searchParams.get("message");
  const socket = useSocket();
  const setActiveChannel = useUnreadStore((s) => s.setActive);
  const clearUnread = useUnreadStore((s) => s.clear);

  // Auto-send read receipts when viewing messages
  const lastMessageId =
    messages.length > 0 ? messages[messages.length - 1].id : null;
  useReadReceipts(channelId, lastMessageId, currentUserId);

  // Mark this channel as active + clear unreads on mount/channel change
  useEffect(() => {
    setActiveChannel(channelId);
    clearUnread(channelId);
    return () => setActiveChannel(null);
  }, [channelId, setActiveChannel, clearUnread]);

  useEffect(() => {
    setOlderLoadError(null);
  }, [channelId]);

  const patchMessageMetadata = useCallback(
    (messageId: string, patch: NonNullable<MessageData["metadata"]>) => {
      setMessages((prev) =>
        prev.map((message) =>
          message.id === messageId
            ? {
                ...message,
                metadata: {
                  ...message.metadata,
                  ...patch,
                },
              }
            : message,
        ),
      );
    },
    [],
  );

  const requestVoiceTranscription = useCallback(
    async (messageId: string, audioUrl: string) => {
      patchMessageMetadata(messageId, {
        transcriptionStatus: "pending",
        transcriptionError: undefined,
      });

      try {
        const res = await fetch(apiUrl("/api/voice/transcribe"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messageId, audioUrl }),
        });

        if (!res.ok) {
          throw new Error(
            await apiErrorMessage(
              res,
              "Voice transcription could not be generated.",
            ),
          );
        }

        const data = (await res.json()) as { transcription?: string };
        const transcription = data.transcription?.trim();
        patchMessageMetadata(messageId, {
          transcription: transcription || undefined,
          transcriptionStatus: "complete",
          transcriptionError: undefined,
        });
      } catch (error) {
        patchMessageMetadata(messageId, {
          transcriptionStatus: "failed",
          transcriptionError:
            error instanceof Error
              ? error.message
              : "Voice transcription could not be generated.",
        });
      }
    },
    [patchMessageMetadata],
  );

  useEffect(() => {
    if (!socket) return;

    socket.emit("join:channel", channelId);

    const handleNewMessage = (msg: MessageData) => {
      if (msg.channelId === channelId) {
        if (msg.parentId) {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === msg.parentId
                ? addThreadReplyPreview(message, msg)
                : message,
            ),
          );
          return;
        }

        setMessages((prev) => {
          if (prev.some((m) => m.id === msg.id)) return prev;
          return [...prev, msg];
        });
        setAutoScrollKey((current) => current + 1);
        setShowJumpToLatest(false);
        if (msg.author.isAgent) {
          setTimeout(() => {
            setAgentActivities((prev) =>
              prev.filter((a) => a.agent.id !== msg.author.id),
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

    const handleTypingStart = ({
      channelId: cid,
      user,
    }: {
      channelId: string;
      user: { id: string; name: string };
    }) => {
      if (cid === channelId && user.id !== currentUserId) {
        setTypingUsers((prev) => new Map(prev).set(user.id, user.name));
      }
    };

    const handleTypingStop = ({
      channelId: cid,
      userId,
    }: {
      channelId: string;
      userId: string;
    }) => {
      if (cid === channelId) {
        setTypingUsers((prev) => {
          const next = new Map(prev);
          next.delete(userId);
          return next;
        });
      }
    };

    const handleAgentActivity = (event: ActivityEvent) => {
      if (event.channelId === channelId) {
        setAgentActivities((prev) => {
          if (event.type === "done" || event.type === "error") {
            // Auto-clear after 3 seconds
            setTimeout(() => {
              setAgentActivities((p) =>
                p.filter((a) => a.agent.id !== event.agent.id),
              );
            }, 3000);
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

    const handleMessageEdit = ({
      id,
      content,
      isEdited,
    }: {
      id: string;
      channelId: string;
      content: string;
      isEdited: boolean;
    }) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === id ? { ...m, content, isEdited } : m)),
      );
    };

    const handleMessageDelete = ({ id }: { id: string }) => {
      setMessages((prev) => prev.filter((m) => m.id !== id));
    };

    const handleReactionUpdate = ({
      messageId,
      reactions,
    }: {
      messageId: string;
      channelId: string;
      reactions: { emoji: string; count: number; users: string[] }[];
    }) => {
      setMessages((prev) =>
        prev.map((m) => {
          if (m.id !== messageId) return m;
          return {
            ...m,
            reactions: reactions.map((r) => ({
              emoji: r.emoji,
              count: r.count,
              userReacted: r.users.includes(currentUserId),
            })),
          };
        }),
      );
    };

    const handleMessagePin = ({
      messageId,
      isPinned,
    }: {
      messageId: string;
      channelId: string;
      isPinned: boolean;
    }) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === messageId ? { ...m, isPinned } : m)),
      );
    };

    const handleTranscription = ({
      messageId,
      transcription,
    }: {
      messageId: string;
      channelId: string;
      transcription: string;
    }) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === messageId
            ? {
                ...m,
                metadata: {
                  ...m.metadata,
                  transcription,
                  transcriptionStatus: "complete",
                  transcriptionError: undefined,
                },
              }
            : m,
        ),
      );
    };

    const handleReadUpdate = ({
      userId: uid,
      messageId,
    }: {
      channelId: string;
      userId: string;
      messageId: string;
    }) => {
      if (uid !== currentUserId) {
        setReadReceipts((prev) => new Map(prev).set(uid, messageId));
      }
    };

    // Fetch initial read receipts
    fetch(apiUrl(`/api/channels/${channelId}/read`))
      .then((r) => (r.ok ? r.json() : []))
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
    setOldestMessageCursor(initialOldestMessageCursor);
    setAutoScrollKey((current) => current + 1);
    setShowJumpToLatest(false);
    setAgentActivities([]);
    setOpenThreadId(null);
  }, [channelId, initialMessages, initialOldestMessageCursor]);

  useEffect(() => {
    if (!highlightedMessageId || messages.length === 0) return;

    const timeout = window.setTimeout(() => {
      document
        .getElementById(`message-${highlightedMessageId}`)
        ?.scrollIntoView({
          block: "center",
          behavior: "smooth",
        });
    }, 150);

    return () => window.clearTimeout(timeout);
  }, [channelId, highlightedMessageId, messages.length]);

  const handleSend = async (content: string) => {
    setSending(true);
    try {
      const res = await fetch(apiUrl(`/api/channels/${channelId}/messages`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
      if (!res.ok) {
        throw new Error(
          await apiErrorMessage(res, "Message could not be sent."),
        );
      }

      const msg = await res.json();
      setMessages((prev) => [...prev, msg]);
      setAutoScrollKey((current) => current + 1);
      setShowJumpToLatest(false);
      socket?.emit("message:send", msg);
      socket?.emit("typing:stop", { channelId, userId: currentUserId });
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
    const res = await fetch(
      apiUrl(`/api/channels/${channelId}/messages/${messageId}/reactions`),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ emoji }),
      },
    );
    if (!res.ok) {
      throw new Error(
        await apiErrorMessage(res, "Reaction could not be updated."),
      );
    }

    const data = (await res.json().catch(() => null)) as {
      reactions?: { emoji: string; count: number; users: string[] }[];
    } | null;
    if (data?.reactions) {
      setMessages((prev) =>
        prev.map((message) =>
          message.id === messageId
            ? {
                ...message,
                reactions: data.reactions!.map((reaction) => ({
                  emoji: reaction.emoji,
                  count: reaction.count,
                  userReacted: reaction.users.includes(currentUserId),
                })),
              }
            : message,
        ),
      );
    }
  };

  const handleEdit = async (messageId: string, content: string) => {
    const res = await fetch(
      apiUrl(`/api/channels/${channelId}/messages/${messageId}`),
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      },
    );
    if (!res.ok) {
      throw new Error(
        await apiErrorMessage(res, "Message could not be edited."),
      );
    }
    setMessages((prev) =>
      prev.map((m) =>
        m.id === messageId ? { ...m, content, isEdited: true } : m,
      ),
    );
  };

  const handleDelete = async (messageId: string, reason?: string) => {
    const res = await fetch(
      apiUrl(`/api/channels/${channelId}/messages/${messageId}`),
      {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason }),
      },
    );
    if (!res.ok) {
      throw new Error(
        await apiErrorMessage(res, "Message could not be deleted."),
      );
    }
    setMessages((prev) => prev.filter((m) => m.id !== messageId));
  };

  const handleLoadOlder = async () => {
    if (!oldestMessageCursor || loadingOlder) return;
    setLoadingOlder(true);
    setOlderLoadError(null);

    try {
      const params = new URLSearchParams({
        cursor: oldestMessageCursor,
        limit: "50",
      });
      const res = await fetch(
        apiUrl(`/api/channels/${channelId}/messages?${params.toString()}`),
        { cache: "no-store" },
      );
      if (!res.ok) {
        throw new Error(
          await apiErrorMessage(res, "Older messages could not be loaded."),
        );
      }

      const data = (await res.json()) as {
        messages?: MessageData[];
        nextCursor?: string | null;
      };
      const olderMessages = data.messages || [];
      setMessages((prev) => {
        const existingIds = new Set(prev.map((message) => message.id));
        const uniqueOlderMessages = olderMessages.filter(
          (message) => !existingIds.has(message.id),
        );
        return [...uniqueOlderMessages, ...prev];
      });
      if (olderMessages.length > 0) setShowJumpToLatest(true);
      setOldestMessageCursor(data.nextCursor || null);
    } catch (error) {
      setOlderLoadError(
        error instanceof Error
          ? error.message
          : "Older messages could not be loaded.",
      );
    } finally {
      setLoadingOlder(false);
    }
  };

  const handleJumpToLatest = () => {
    setAutoScrollKey((current) => current + 1);
    setShowJumpToLatest(false);
  };

  const handlePin = async (messageId: string, isPinned: boolean) => {
    const res = await fetch(apiUrl(`/api/channels/${channelId}/pins`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messageId }),
    });
    if (!res.ok) {
      throw new Error(
        await apiErrorMessage(res, "Message pin could not be updated."),
      );
    }
    const data = (await res.json().catch(() => null)) as {
      isPinned?: boolean;
    } | null;
    setMessages((prev) =>
      prev.map((m) =>
        m.id === messageId
          ? { ...m, isPinned: data?.isPinned ?? !isPinned }
          : m,
      ),
    );
  };

  const handleToggleSaved = async (messageId: string, isSaved: boolean) => {
    const res = await fetch(apiUrl("/api/saved"), {
      method: isSaved ? "DELETE" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messageId }),
    });
    if (!res.ok) {
      throw new Error(
        await apiErrorMessage(res, "Saved state could not be updated."),
      );
    }
    const data = (await res.json().catch(() => null)) as {
      saved?: boolean;
    } | null;
    setMessages((prev) =>
      prev.map((m) =>
        m.id === messageId ? { ...m, isSaved: data?.saved ?? !isSaved } : m,
      ),
    );
  };

  const handleOpenThread = (messageId: string) => {
    setOpenThreadId(messageId);
  };

  const handleEmailReply = async (messageId: string, content: string) => {
    const res = await fetch(apiUrl(`/api/email-import/${messageId}/reply`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    if (!res.ok) {
      throw new Error(
        await apiErrorMessage(res, "Email reply could not be sent."),
      );
    }

    const data = (await res.json()) as { message?: MessageData };
    if (!data.message) throw new Error("Email reply could not be recorded.");

    setMessages((prev) =>
      prev.map((message) =>
        message.id === messageId
          ? addThreadReplyPreview(message, data.message!)
          : message,
      ),
    );
    socket?.emit("message:send", data.message);
    return data.message;
  };

  const handleReplyCreated = (reply: MessageData) => {
    if (!reply.parentId) return;
    setMessages((prev) =>
      prev.map((message) =>
        message.id === reply.parentId
          ? addThreadReplyPreview(message, reply)
          : message,
      ),
    );
  };

  const handleVoiceSend = async (audioBlob: Blob, duration: number) => {
    const audioMimeType = audioBlob.type || "audio/webm";
    const audioExtension = audioMimeType.includes("wav")
      ? "wav"
      : audioMimeType.includes("mp4")
        ? "m4a"
        : "webm";
    // Upload audio to S3
    const formData = new FormData();
    formData.append("file", audioBlob, `voice-message.${audioExtension}`);
    const uploadRes = await fetch(apiUrl("/api/upload"), {
      method: "POST",
      body: formData,
    });
    if (!uploadRes.ok) {
      throw new Error(
        await apiErrorMessage(uploadRes, "Voice message could not upload."),
      );
    }
    const { s3Key, url, fileName, fileSize, mimeType } = await uploadRes.json();

    // Send message with audio attachment + voice metadata
    const res = await fetch(apiUrl(`/api/channels/${channelId}/messages`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: "🎙️ Voice message",
        attachments: [{ s3Key, url, fileName, fileSize, mimeType }],
        metadata: { type: "voice", duration },
      }),
    });
    if (!res.ok) {
      throw new Error(
        await apiErrorMessage(res, "Voice message could not be sent."),
      );
    }
    const msg = await res.json();
    const messageWithTranscriptionState: MessageData = {
      ...msg,
      metadata: {
        ...msg.metadata,
        transcriptionStatus: "pending",
        transcriptionError: undefined,
      },
    };
    setMessages((prev) => [...prev, messageWithTranscriptionState]);
    setAutoScrollKey((current) => current + 1);
    setShowJumpToLatest(false);
    socket?.emit("message:send", messageWithTranscriptionState);

    void requestVoiceTranscription(msg.id, url);
  };

  const handleFileUpload = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    const uploadRes = await fetch(apiUrl("/api/upload"), {
      method: "POST",
      body: formData,
    });
    if (!uploadRes.ok) {
      throw new Error(
        await apiErrorMessage(uploadRes, "File could not upload."),
      );
    }
    const { s3Key, url, fileName, fileSize, mimeType } = await uploadRes.json();

    const res = await fetch(apiUrl(`/api/channels/${channelId}/messages`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: `📎 ${fileName}`,
        attachments: [{ s3Key, url, fileName, fileSize, mimeType }],
      }),
    });
    if (!res.ok) {
      throw new Error(
        await apiErrorMessage(res, "File message could not be sent."),
      );
    }
    const msg = await res.json();
    setMessages((prev) => [...prev, msg]);
    setAutoScrollKey((current) => current + 1);
    setShowJumpToLatest(false);
    socket?.emit("message:send", msg);
  };

  const typingNames = Array.from(typingUsers.values()).filter(Boolean);

  const threadParent = openThreadId
    ? messages.find((m) => m.id === openThreadId)
    : null;

  return (
    <div className="relative flex flex-1 min-h-0">
      <div className="flex flex-col flex-1 min-w-0">
        <MessageList
          messages={messages}
          currentUserId={currentUserId}
          emptyState={emptyState}
          highlightedMessageId={highlightedMessageId}
          readReceipts={readReceipts}
          hasMoreMessages={Boolean(oldestMessageCursor)}
          loadingOlder={loadingOlder}
          olderLoadError={olderLoadError}
          autoScrollKey={autoScrollKey}
          showJumpToLatest={showJumpToLatest}
          unreadAfter={initialUnreadAfter}
          canModerateMessages={canManageMessages}
          onLoadOlder={handleLoadOlder}
          onDismissOlderLoadError={() => setOlderLoadError(null)}
          onJumpToLatest={handleJumpToLatest}
          onReaction={handleReaction}
          onEdit={handleEdit}
          onDelete={handleDelete}
          onPin={handlePin}
          onToggleSaved={handleToggleSaved}
          onOpenThread={handleOpenThread}
          onEmailReply={handleEmailReply}
          onRetryVoiceTranscription={(messageId, audioUrl) =>
            void requestVoiceTranscription(messageId, audioUrl)
          }
        />
        <AgentActivity activities={agentActivities} />
        {typingNames.length > 0 && (
          <div className="px-4 py-1 text-2xs text-text-muted animate-pulse">
            {typingNames.join(", ")} {typingNames.length === 1 ? "is" : "are"}{" "}
            typing...
          </div>
        )}
        <MessageInput
          channelId={channelId}
          channelName={channelName}
          onSend={handleSend}
          onTyping={handleTyping}
          disabled={sending}
          placeholder={placeholder}
          draftId={`${currentUserId}:channel:${channelId}`}
          onVoiceSend={handleVoiceSend}
          onFileUpload={handleFileUpload}
        />
      </div>

      {openThreadId && threadParent && (
        <ThreadPanel
          channelId={channelId}
          parentMessage={threadParent}
          currentUserId={currentUserId}
          onReplyCreated={handleReplyCreated}
          onClose={() => setOpenThreadId(null)}
        />
      )}
    </div>
  );
}
