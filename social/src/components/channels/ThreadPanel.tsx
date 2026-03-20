"use client";

import { useState, useEffect, useRef } from "react";
import { X, Reply } from "lucide-react";
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
  onClose: () => void;
}

export default function ThreadPanel({ channelId, parentMessage, currentUserId: _currentUserId, onClose }: ThreadPanelProps) {
  const [replies, setReplies] = useState<MessageData[]>([]);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const socket = useSocket();

  useEffect(() => {
    setLoading(true);
    fetch(apiUrl(`/api/channels/${channelId}/messages?parentId=${parentMessage.id}`))
      .then((r) => r.json())
      .then((data) => {
        setReplies(data.messages || []);
        setLoading(false);
        setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
      })
      .catch(() => setLoading(false));
  }, [channelId, parentMessage.id]);

  useEffect(() => {
    if (!socket) return;

    const handleNewMessage = (msg: MessageData) => {
      if (msg.channelId === channelId && msg.parentId === parentMessage.id) {
        setReplies((prev) => {
          if (prev.some((m) => m.id === msg.id)) return prev;
          return [...prev, msg];
        });
        setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
      }
    };

    socket.on("message:new", handleNewMessage);
    return () => { socket.off("message:new", handleNewMessage); };
  }, [socket, channelId, parentMessage.id]);

  const handleSendReply = async (content: string) => {
    setSending(true);
    try {
      const res = await fetch(apiUrl(`/api/channels/${channelId}/messages`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content, parentId: parentMessage.id }),
      });
      if (res.ok) {
        const msg = await res.json();
        setReplies((prev) => [...prev, msg]);
        socket?.emit("message:send", msg);
        setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
      }
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="w-[400px] border-l border-border flex flex-col bg-bg-base shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <Reply size={16} className="text-text-muted" />
          <span className="font-heading font-semibold text-sm text-text-primary">Thread</span>
        </div>
        <button onClick={onClose} className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors">
          <X size={16} />
        </button>
      </div>

      {/* Parent message */}
      <div className="px-4 py-3 border-b border-border bg-bg-surface/50">
        <div className="flex items-baseline gap-2 mb-1">
          <span className={`font-medium text-sm ${parentMessage.author.isAgent ? "text-teal" : "text-text-primary"}`}>
            {parentMessage.author.displayName}
          </span>
          <span className="text-2xs text-text-muted">
            {formatDistanceToNow(new Date(parentMessage.createdAt), { addSuffix: true })}
          </span>
        </div>
        <div className="text-sm text-text-primary/90 leading-relaxed">
          <MarkdownContent content={parentMessage.content} />
        </div>
      </div>

      {/* Replies */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-sm text-text-muted">Loading replies...</div>
          </div>
        ) : replies.length === 0 ? (
          <div className="flex items-center justify-center py-8">
            <div className="text-sm text-text-muted">No replies yet</div>
          </div>
        ) : (
          <div className="py-2">
            {replies.map((reply) => (
              <div key={reply.id} className="px-4 py-2 hover:bg-bg-hover/50 transition-colors">
                <div className="flex items-baseline gap-2 mb-0.5">
                  <span className={`font-medium text-sm ${reply.author.isAgent ? "text-teal" : "text-text-primary"}`}>
                    {reply.author.displayName}
                  </span>
                  <span className="text-2xs text-text-muted">
                    {formatDistanceToNow(new Date(reply.createdAt), { addSuffix: true })}
                  </span>
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
      />
    </div>
  );
}
