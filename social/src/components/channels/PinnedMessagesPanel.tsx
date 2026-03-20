"use client";

import { useState, useEffect } from "react";
import { Pin, X } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import MarkdownContent from "./MarkdownContent";
import { apiUrl } from "@/lib/apiUrl";

interface PinnedMessage {
  id: string;
  channelId: string;
  content: string;
  createdAt: string;
  author: { id: string; displayName: string; isAgent: boolean };
}

interface PinnedMessagesPanelProps {
  channelId: string;
  onClose: () => void;
}

export default function PinnedMessagesPanel({ channelId, onClose }: PinnedMessagesPanelProps) {
  const [pins, setPins] = useState<PinnedMessage[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(apiUrl(`/api/channels/${channelId}/pins`))
      .then((r) => r.json())
      .then((data) => {
        setPins(data.pins || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [channelId]);

  const handleUnpin = async (messageId: string) => {
    await fetch(apiUrl(`/api/channels/${channelId}/pins`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messageId }),
    });
    setPins((prev) => prev.filter((p) => p.id !== messageId));
  };

  return (
    <div className="absolute right-4 top-14 z-40 w-80 bg-bg-surface border border-border rounded-xl shadow-xl overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div className="flex items-center gap-2">
          <Pin size={14} className="text-accent" />
          <span className="font-heading font-semibold text-sm text-text-primary">Pinned Messages</span>
        </div>
        <button onClick={onClose} className="p-1 rounded hover:bg-bg-hover text-text-muted">
          <X size={14} />
        </button>
      </div>

      <div className="max-h-80 overflow-y-auto">
        {loading && (
          <div className="px-4 py-6 text-center text-sm text-text-muted">Loading...</div>
        )}
        {!loading && pins.length === 0 && (
          <div className="px-4 py-6 text-center text-sm text-text-muted">No pinned messages</div>
        )}
        {!loading && pins.map((pin) => (
          <div key={pin.id} className="px-4 py-3 border-b border-border/50 last:border-0 hover:bg-bg-hover/50 transition-colors group">
            <div className="flex items-center justify-between mb-1">
              <span className={`text-xs font-medium ${pin.author.isAgent ? "text-teal" : "text-text-primary"}`}>
                {pin.author.displayName}
              </span>
              <div className="flex items-center gap-1">
                <span className="text-2xs text-text-muted">
                  {formatDistanceToNow(new Date(pin.createdAt), { addSuffix: true })}
                </span>
                <button
                  onClick={() => handleUnpin(pin.id)}
                  className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-bg-elevated text-text-muted hover:text-danger transition-all"
                  title="Unpin"
                >
                  <X size={10} />
                </button>
              </div>
            </div>
            <div className="text-sm text-text-primary/80 line-clamp-3">
              <MarkdownContent content={pin.content} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
