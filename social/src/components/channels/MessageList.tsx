"use client";

import { useRef, useEffect, useState } from "react";
import { formatDistanceToNow, format } from "date-fns";
import { Bot, Reply, SmilePlus, MoreHorizontal, Pencil, Trash2, Pin, Check, CheckCheck, X } from "lucide-react";
import MarkdownContent from "./MarkdownContent";
import EmojiPicker from "./EmojiPicker";
import VoicePlayer from "./VoicePlayer";
import type { MessageData } from "@/types";

interface MessageListProps {
  messages: MessageData[];
  currentUserId: string;
  readReceipts?: Map<string, string>;
  onReaction?: (messageId: string, emoji: string) => void;
  onEdit?: (messageId: string, content: string) => void;
  onDelete?: (messageId: string) => void;
  onPin?: (messageId: string, isPinned: boolean) => void;
  onOpenThread?: (messageId: string) => void;
}

export default function MessageList({
  messages,
  currentUserId,
  readReceipts,
  onReaction,
  onEdit,
  onDelete,
  onPin,
  onOpenThread,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-3">
            <span role="img" aria-label="wave">&#128075;</span>
          </div>
          <h3 className="font-heading text-lg font-semibold text-text-primary mb-1">No messages yet</h3>
          <p className="text-sm text-text-muted">Be the first to say something.</p>
        </div>
      </div>
    );
  }

  // Build a map of messageId -> index for fast lookup
  const msgIndexMap = new Map<string, number>();
  messages.forEach((m, i) => msgIndexMap.set(m.id, i));

  // For each message, compute how many people have read up to it or beyond
  // readReceipts is Map<userId, messageId> — each user's latest read message
  const getReadByCount = (msgIndex: number): number => {
    if (!readReceipts || readReceipts.size === 0) return 0;
    let count = 0;
    readReceipts.forEach((readMsgId) => {
      const readIdx = msgIndexMap.get(readMsgId);
      if (readIdx !== undefined && readIdx >= msgIndex) count++;
    });
    return count;
  };

  let lastAuthor = "";
  let lastDate = "";

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="py-2">
        {messages.map((msg, msgIdx) => {
          const msgDate = format(new Date(msg.createdAt), "MMM d, yyyy");
          const showDateDivider = msgDate !== lastDate;
          const isGrouped = msg.author.id === lastAuthor && !showDateDivider;
          lastAuthor = msg.author.id;
          lastDate = msgDate;

          return (
            <div key={msg.id}>
              {showDateDivider && (
                <div className="flex items-center gap-3 px-4 py-3">
                  <div className="flex-1 h-px bg-border" />
                  <span className="text-2xs font-medium text-text-muted uppercase tracking-wider">{msgDate}</span>
                  <div className="flex-1 h-px bg-border" />
                </div>
              )}
              <MessageRow
                msg={msg}
                isGrouped={isGrouped}
                isOwn={msg.author.id === currentUserId}
                readByCount={getReadByCount(msgIdx)}
                onReaction={onReaction}
                onEdit={onEdit}
                onDelete={onDelete}
                onPin={onPin}
                onOpenThread={onOpenThread}
              />
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

function MessageRow({
  msg, isGrouped, isOwn, readByCount, onReaction, onEdit, onDelete, onPin, onOpenThread,
}: {
  msg: MessageData; isGrouped: boolean; isOwn: boolean; readByCount: number;
  onReaction?: (messageId: string, emoji: string) => void;
  onEdit?: (messageId: string, content: string) => void;
  onDelete?: (messageId: string) => void;
  onPin?: (messageId: string, isPinned: boolean) => void;
  onOpenThread?: (messageId: string) => void;
}) {
  const [showMenu, setShowMenu] = useState(false);
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState(msg.content);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!showMenu && !showEmojiPicker) return;
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setShowMenu(false);
        setShowEmojiPicker(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showMenu, showEmojiPicker]);

  const handleSaveEdit = () => {
    if (editContent.trim() && editContent !== msg.content) {
      onEdit?.(msg.id, editContent.trim());
    }
    setEditing(false);
  };

  return (
    <div className="channel-message group">
      <div className="flex gap-3">
        {isGrouped ? (
          <div className="w-8 shrink-0 flex items-start justify-center pt-1 opacity-0 group-hover:opacity-100">
            <span className="text-2xs text-text-muted">
              {new Date(msg.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </span>
          </div>
        ) : (
          <div className={`w-8 h-8 avatar text-xs mt-0.5 ${msg.author.isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"}`}>
            {msg.author.isAgent ? <Bot size={14} /> : msg.author.displayName[0]?.toUpperCase()}
          </div>
        )}

        <div className="flex-1 min-w-0">
          {!isGrouped && (
            <div className="flex items-baseline gap-2 mb-0.5">
              <span className={`font-medium text-sm ${msg.author.isAgent ? "text-teal" : "text-text-primary"}`}>
                {msg.author.displayName}
              </span>
              {msg.author.isAgent && <span className="badge-teal">agent</span>}
              <span className="text-2xs text-text-muted">
                {formatDistanceToNow(new Date(msg.createdAt), { addSuffix: true })}
              </span>
              {msg.isEdited && <span className="text-2xs text-text-muted">(edited)</span>}
              {msg.isPinned && <Pin size={10} className="text-accent" />}
            </div>
          )}

          {editing ? (
            <div className="space-y-2">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full bg-bg-elevated border border-border rounded-lg px-3 py-2 text-sm text-text-primary resize-none focus:outline-none focus:border-accent"
                rows={Math.min(editContent.split("\n").length + 1, 6)}
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSaveEdit(); }
                  if (e.key === "Escape") { setEditContent(msg.content); setEditing(false); }
                }}
              />
              <div className="flex items-center gap-2 text-2xs text-text-muted">
                <button onClick={handleSaveEdit} className="flex items-center gap-1 text-teal hover:underline"><Check size={10} /> Save</button>
                <span>·</span>
                <button onClick={() => { setEditContent(msg.content); setEditing(false); }} className="flex items-center gap-1 hover:underline"><X size={10} /> Cancel</button>
                <span className="ml-2">Esc to cancel · Enter to save</span>
              </div>
            </div>
          ) : (
            <div className="text-sm text-text-primary/90 whitespace-pre-wrap break-words leading-relaxed">
              <MarkdownContent content={msg.content} />
            </div>
          )}

          {/* Voice message */}
          {msg.metadata?.type === "voice" && msg.attachments.length > 0 && (
            <div className="mt-1">
              <VoicePlayer
                url={msg.attachments[0].url}
                duration={msg.metadata.duration}
                transcription={msg.metadata.transcription}
                isAgent={msg.author.isAgent}
              />
            </div>
          )}

          {/* File attachments (non-voice) */}
          {msg.attachments.length > 0 && msg.metadata?.type !== "voice" && (
            <div className="flex flex-wrap gap-2 mt-2">
              {msg.attachments.map((att) => {
                const isImage = att.mimeType?.startsWith("image/");
                const isAudio = att.mimeType?.startsWith("audio/");
                if (isAudio) {
                  return (
                    <div key={att.id}>
                      <VoicePlayer url={att.url} isAgent={msg.author.isAgent} />
                    </div>
                  );
                }
                return isImage ? (
                  <a key={att.id} href={att.url} target="_blank" rel="noopener noreferrer">
                    <img src={att.url} alt={att.fileName} className="max-w-sm max-h-64 rounded-lg border border-border" />
                  </a>
                ) : (
                  <a key={att.id} href={att.url} target="_blank" rel="noopener noreferrer"
                    className="flex items-center gap-2 px-3 py-2 rounded-lg bg-bg-elevated border border-border hover:border-accent transition-colors text-xs">
                    <span>📎</span>
                    <span className="text-text-primary truncate max-w-[200px]">{att.fileName}</span>
                  </a>
                );
              })}
            </div>
          )}

          {msg.reactions.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {msg.reactions.map((r) => (
                <button key={r.emoji} onClick={() => onReaction?.(msg.id, r.emoji)}
                  className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-2xs border transition-colors ${
                    r.userReacted ? "border-accent bg-accent-muted text-accent" : "border-border bg-bg-elevated text-text-secondary hover:border-text-muted"
                  }`}>
                  <span>{r.emoji}</span><span>{r.count}</span>
                </button>
              ))}
              <button onClick={() => setShowEmojiPicker(true)}
                className="flex items-center px-1.5 py-0.5 rounded-full text-2xs border border-dashed border-border text-text-muted hover:border-text-muted transition-colors">
                <SmilePlus size={10} />
              </button>
            </div>
          )}

          {(msg.replyCount || 0) > 0 && (
            <button onClick={() => onOpenThread?.(msg.id)} className="flex items-center gap-1.5 mt-1.5 text-2xs text-accent hover:underline">
              <Reply size={12} />
              <span>{msg.replyCount} {msg.replyCount === 1 ? "reply" : "replies"}</span>
            </button>
          )}

          {/* Read receipts — show for own messages */}
          {isOwn && (
            <div className="flex items-center gap-1 mt-1 text-2xs" title={readByCount > 0 ? `Seen by ${readByCount}` : "Sent"}>
              {readByCount > 0 ? (
                <>
                  <CheckCheck size={12} className="text-teal" />
                  {readByCount > 1 && <span className="text-text-muted">Seen by {readByCount}</span>}
                </>
              ) : (
                <Check size={12} className="text-text-muted/50" />
              )}
            </div>
          )}
        </div>

        {/* Hover actions */}
        <div ref={menuRef} className="opacity-0 group-hover:opacity-100 flex items-start gap-0.5 pt-0.5 transition-opacity relative">
          <button onClick={() => setShowEmojiPicker(!showEmojiPicker)} className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors">
            <SmilePlus size={14} />
          </button>
          <button onClick={() => onOpenThread?.(msg.id)} className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors">
            <Reply size={14} />
          </button>
          <button onClick={() => setShowMenu(!showMenu)} className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors">
            <MoreHorizontal size={14} />
          </button>

          {showEmojiPicker && (
            <div className="absolute right-0 top-8 z-50">
              <EmojiPicker onSelect={(emoji) => { onReaction?.(msg.id, emoji); setShowEmojiPicker(false); }} onClose={() => setShowEmojiPicker(false)} />
            </div>
          )}

          {showMenu && (
            <div className="absolute right-0 top-8 z-50 w-44 bg-bg-surface border border-border rounded-lg shadow-lg py-1">
              {isOwn && (
                <button onClick={() => { setEditing(true); setShowMenu(false); }}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover hover:text-text-primary transition-colors">
                  <Pencil size={12} /> Edit message
                </button>
              )}
              <button onClick={() => { onPin?.(msg.id, msg.isPinned); setShowMenu(false); }}
                className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover hover:text-text-primary transition-colors">
                <Pin size={12} /> {msg.isPinned ? "Unpin" : "Pin"} message
              </button>
              {isOwn && (
                <button onClick={() => { onDelete?.(msg.id); setShowMenu(false); }}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-danger hover:bg-danger-muted transition-colors">
                  <Trash2 size={12} /> Delete message
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
