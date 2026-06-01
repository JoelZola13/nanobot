"use client";

import { useCallback, useState, useEffect, useRef } from "react";
import { AlertCircle, Loader2, MailCheck, RefreshCw, X, Reply } from "lucide-react";
import MessageInput from "./MessageInput";
import MarkdownContent from "./MarkdownContent";
import { useSocket } from "@/components/providers/SocketProvider";
import { formatDistanceToNow } from "date-fns";
import type { MessageData } from "@/types";
import { apiUrl } from "@/lib/apiUrl";

interface ThreadPanelProps {
  channelId: string;
  parentMessage: MessageData;
  currentUserId: string;
  onReplyCreated?: (reply: MessageData) => void;
  onClose: () => void;
}

async function apiErrorMessage(res: Response, fallback: string) {
  const data = (await res.json().catch(() => null)) as {
    error?: string;
  } | null;
  return data?.error || fallback;
}

export default function ThreadPanel({
  channelId,
  parentMessage,
  currentUserId,
  onReplyCreated,
  onClose,
}: ThreadPanelProps) {
  const [replies, setReplies] = useState<MessageData[]>([]);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const socket = useSocket();

  const loadReplies = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        apiUrl(
          `/api/channels/${channelId}/messages?parentId=${parentMessage.id}`,
        ),
        { cache: "no-store" },
      );
      if (!res.ok) {
        throw new Error(
          await apiErrorMessage(res, "Thread replies could not load."),
        );
      }
      const data = (await res.json()) as { messages?: MessageData[] };
      setReplies(data.messages || []);
      setTimeout(
        () => bottomRef.current?.scrollIntoView({ behavior: "smooth" }),
        100,
      );
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : null;
      setError(message || "Thread replies could not load.");
    } finally {
      setLoading(false);
    }
  }, [channelId, parentMessage.id]);

  useEffect(() => {
    void loadReplies();
  }, [loadReplies]);

  useEffect(() => {
    if (!socket) return;

    const handleNewMessage = (msg: MessageData) => {
      if (msg.channelId === channelId && msg.parentId === parentMessage.id) {
        setReplies((prev) => {
          if (prev.some((m) => m.id === msg.id)) return prev;
          return [...prev, msg];
        });
        setError(null);
        setTimeout(
          () => bottomRef.current?.scrollIntoView({ behavior: "smooth" }),
          100,
        );
      }
    };

    socket.on("message:new", handleNewMessage);
    return () => {
      socket.off("message:new", handleNewMessage);
    };
  }, [socket, channelId, parentMessage.id]);

  const handleSendReply = async (content: string) => {
    setSending(true);
    try {
      const res = await fetch(apiUrl(`/api/channels/${channelId}/messages`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content, parentId: parentMessage.id }),
      });
      if (!res.ok)
        throw new Error(await apiErrorMessage(res, "Reply could not be sent."));

      const msg = await res.json();
      setReplies((prev) => [...prev, msg]);
      onReplyCreated?.(msg);
      socket?.emit("message:send", msg);
      setTimeout(
        () => bottomRef.current?.scrollIntoView({ behavior: "smooth" }),
        100,
      );
    } finally {
      setSending(false);
    }
  };

  return (
    <div
      data-testid="thread-panel"
      role="complementary"
      aria-label="Thread"
      className="sv-thread-panel w-[400px] border-l border-border flex flex-col bg-bg-base shrink-0"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <Reply size={16} className="text-text-muted" />
          <span className="font-heading font-semibold text-sm text-text-primary">
            Thread
          </span>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors"
          type="button"
          title="Close thread"
          aria-label="Close thread"
        >
          <X size={16} />
        </button>
      </div>

      {/* Parent message */}
      <div className="px-4 py-3 border-b border-border bg-bg-surface/50">
        <div className="flex items-baseline gap-2 mb-1">
          <span
            className={`font-medium text-sm ${parentMessage.author.isAgent ? "text-teal" : "text-text-primary"}`}
          >
            {parentMessage.author.displayName}
          </span>
          <span className="text-2xs text-text-muted">
            {formatDistanceToNow(new Date(parentMessage.createdAt), {
              addSuffix: true,
            })}
          </span>
        </div>
        <div className="text-sm text-text-primary/90 leading-relaxed">
          <MarkdownContent content={parentMessage.content} />
        </div>
      </div>

      {/* Replies */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center gap-2 py-8 text-sm text-text-muted">
            <Loader2 size={15} className="animate-spin" />
            <span>Loading replies...</span>
          </div>
        ) : error ? (
          <div
            data-testid="thread-load-error"
            className="m-4 rounded-lg border border-red-300 bg-red-50 px-3 py-3 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
          >
            <div className="flex items-start gap-2">
              <AlertCircle size={15} className="mt-0.5 shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="font-medium">Thread replies could not load</div>
                <div className="mt-0.5 text-xs opacity-90">{error}</div>
              </div>
            </div>
            <button
              type="button"
              onClick={() => void loadReplies()}
              className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-red-300 bg-white px-2 py-1 text-xs font-medium text-red-700 hover:bg-red-50 dark:border-red-900/70 dark:bg-red-950/40 dark:text-red-100 dark:hover:bg-red-900/30"
            >
              <RefreshCw size={12} />
              Retry replies
            </button>
          </div>
        ) : replies.length === 0 ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-sm text-text-muted">No replies yet</div>
          </div>
        ) : (
          <div className="py-2">
            {replies.map((reply) => (
              <div
                key={reply.id}
                className="px-4 py-2 hover:bg-bg-hover/50 transition-colors"
              >
                <div className="flex items-baseline gap-2 mb-0.5">
                  <span
                    className={`font-medium text-sm ${reply.author.isAgent ? "text-teal" : "text-text-primary"}`}
                  >
                    {reply.author.displayName}
                  </span>
                  <span className="text-2xs text-text-muted">
                    {formatDistanceToNow(new Date(reply.createdAt), {
                      addSuffix: true,
                    })}
                  </span>
                  {reply.metadata?.type === "email_reply" && (
                    <span className="inline-flex items-center gap-1 rounded-full bg-teal-muted px-1.5 py-0.5 text-2xs font-medium text-teal">
                      <MailCheck size={10} />
                      Email sent
                    </span>
                  )}
                </div>
                <div className="text-sm text-text-primary/90 leading-relaxed">
                  <MarkdownContent content={reply.content} />
                </div>
              </div>
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Reply input */}
      <MessageInput
        channelId={channelId}
        channelName="thread"
        onSend={handleSendReply}
        disabled={sending}
        placeholder="Reply in thread..."
        draftId={`${currentUserId}:thread:${channelId}:${parentMessage.id}`}
      />
    </div>
  );
}
