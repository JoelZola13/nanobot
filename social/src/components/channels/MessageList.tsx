"use client";

import { useRef, useEffect, useState } from "react";
import { formatDistanceToNow, format } from "date-fns";
import {
  Bookmark,
  BookmarkCheck,
  Bot,
  Check,
  CheckCheck,
  Hash,
  Lock,
  Link2,
  MessageSquare,
  MoreHorizontal,
  Pencil,
  Pin,
  Reply,
  SmilePlus,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import MarkdownContent from "./MarkdownContent";
import EmojiPicker from "./EmojiPicker";
import VoicePlayer from "./VoicePlayer";
import ProfilePopover from "@/components/users/ProfilePopover";
import type { MessageData } from "@/types";

export type MessageEmptyState = {
  kind: "channel" | "dm";
  name: string;
  description?: string;
  isPrivate?: boolean;
  isAgent?: boolean;
  memberCount?: number;
  status?: string;
};

interface MessageListProps {
  messages: MessageData[];
  currentUserId: string;
  emptyState: MessageEmptyState;
  highlightedMessageId?: string | null;
  readReceipts?: Map<string, string>;
  onReaction?: (messageId: string, emoji: string) => void;
  onEdit?: (messageId: string, content: string) => void;
  onDelete?: (messageId: string) => void;
  onPin?: (messageId: string, isPinned: boolean) => void;
  onToggleSaved?: (messageId: string, isSaved: boolean) => void;
  onOpenThread?: (messageId: string) => void;
}

export default function MessageList({
  messages,
  currentUserId,
  emptyState,
  highlightedMessageId,
  readReceipts,
  onReaction,
  onEdit,
  onDelete,
  onPin,
  onToggleSaved,
  onOpenThread,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (highlightedMessageId) return;
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, highlightedMessageId]);

  if (messages.length === 0) {
    return <EmptyConversation state={emptyState} />;
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
                channelKind={emptyState.kind}
                isHighlighted={msg.id === highlightedMessageId}
                onReaction={onReaction}
                onEdit={onEdit}
                onDelete={onDelete}
                onPin={onPin}
                onToggleSaved={onToggleSaved}
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

function EmptyConversation({ state }: { state: MessageEmptyState }) {
  const isDm = state.kind === "dm";
  const Icon = isDm ? (state.isAgent ? Bot : MessageSquare) : state.isPrivate ? Lock : Hash;
  const title = isDm
    ? state.isAgent
      ? `Ask ${state.name} for help`
      : `Start a DM with ${state.name}`
    : `Start #${state.name}`;
  const subtitle = isDm
    ? state.isAgent
      ? `${state.name} is ready for drafts, research, summaries, and follow-ups.`
      : `This conversation is just between you and ${state.name}.`
    : state.description || `${state.name} is ready for team updates, questions, and decisions.`;
  const detail = isDm
    ? state.isAgent
      ? "Send a clear task below and keep the useful output in one place."
      : "Send the first note below when you are ready."
    : "Send the first message below to get the channel moving.";
  const meta = isDm
    ? state.isAgent
      ? "AI agent"
      : state.status === "online"
        ? "Online teammate"
        : "Teammate"
    : `${state.isPrivate ? "Private channel" : "Channel"}${state.memberCount ? ` / ${state.memberCount} member${state.memberCount === 1 ? "" : "s"}` : ""}`;

  return (
    <div className="flex flex-1 items-center justify-center px-5 py-10">
      <div className="max-w-md text-center">
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-xl border border-border bg-bg-elevated text-text-primary">
          <Icon size={24} />
        </div>
        <div className="mb-2 flex items-center justify-center gap-2 text-2xs font-semibold uppercase text-text-muted">
          {state.isAgent && <Sparkles size={12} className="text-teal" />}
          <span>{meta}</span>
        </div>
        <h3 className="mb-2 font-heading text-xl font-semibold text-text-primary">{title}</h3>
        <p className="text-sm leading-6 text-text-secondary">{subtitle}</p>
        <p className="mt-2 text-sm text-text-muted">{detail}</p>
      </div>
    </div>
  );
}

function copyWithHiddenTextarea(value: string) {
  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.top = "0";
  textarea.style.left = "0";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  textarea.setSelectionRange(0, value.length);

  try {
    return document.execCommand("copy");
  } finally {
    document.body.removeChild(textarea);
  }
}

function MessageRow({
  msg,
  isGrouped,
  isOwn,
  readByCount,
  channelKind,
  isHighlighted,
  onReaction,
  onEdit,
  onDelete,
  onPin,
  onToggleSaved,
  onOpenThread,
}: {
  msg: MessageData;
  isGrouped: boolean;
  isOwn: boolean;
  readByCount: number;
  channelKind: "channel" | "dm";
  isHighlighted: boolean;
  onReaction?: (messageId: string, emoji: string) => void;
  onEdit?: (messageId: string, content: string) => void;
  onDelete?: (messageId: string) => void;
  onPin?: (messageId: string, isPinned: boolean) => void;
  onToggleSaved?: (messageId: string, isSaved: boolean) => void;
  onOpenThread?: (messageId: string) => void;
}) {
  const [showMenu, setShowMenu] = useState(false);
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const [editing, setEditing] = useState(false);
  const [copyNotice, setCopyNotice] = useState<{ status: "copied" | "ready"; permalink: string } | null>(null);
  const [editContent, setEditContent] = useState(msg.content);
  const menuRef = useRef<HTMLDivElement>(null);
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout>>();
  const saveLabel = msg.isSaved ? "Remove from Later" : "Save for later";

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

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
    };
  }, []);

  const handleSaveEdit = () => {
    if (editContent.trim() && editContent !== msg.content) {
      onEdit?.(msg.id, editContent.trim());
    }
    setEditing(false);
  };

  const handleCopyLink = async () => {
    const params = new URLSearchParams({
      channel: `${channelKind === "dm" ? "dm" : "channel"}-${msg.channelId}`,
      message: msg.id,
    });
    const permalink = `${window.location.origin}/messages?${params.toString()}`;

    try {
      if (navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(permalink);
        } catch {
          if (!copyWithHiddenTextarea(permalink)) {
            throw new Error("Copy command failed");
          }
        }
      } else if (!copyWithHiddenTextarea(permalink)) {
        throw new Error("Copy command failed");
      }

      setCopyNotice({ status: "copied", permalink });
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
      copyTimeoutRef.current = setTimeout(() => setCopyNotice(null), 1800);
    } catch {
      setCopyNotice({ status: "ready", permalink });
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
      copyTimeoutRef.current = setTimeout(() => setCopyNotice(null), 8000);
    }
  };

  return (
    <div
      id={`message-${msg.id}`}
      data-testid="message-row"
      data-message-id={msg.id}
      className={`channel-message group scroll-mt-20 ${isHighlighted ? "bg-accent-muted ring-1 ring-inset ring-accent" : ""}`}
      aria-current={isHighlighted ? "true" : undefined}
    >
      <div className="flex gap-3">
        {isGrouped ? (
          <div className="w-8 shrink-0 flex items-start justify-center pt-1 opacity-0 group-hover:opacity-100">
            <span className="text-2xs text-text-muted">
              {new Date(msg.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </span>
          </div>
        ) : (
          <ProfilePopover
            user={msg.author}
            className="mt-0.5 shrink-0"
            triggerClassName="rounded-full focus:outline-none focus:ring-2 focus:ring-accent/40"
          >
            <span className={`w-8 h-8 avatar text-xs ${msg.author.isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"}`}>
              {msg.author.isAgent ? <Bot size={14} /> : msg.author.displayName[0]?.toUpperCase()}
            </span>
          </ProfilePopover>
        )}

        <div className="flex-1 min-w-0">
          {!isGrouped && (
            <div className="flex items-baseline gap-2 mb-0.5">
              <ProfilePopover
                user={msg.author}
                triggerClassName={`rounded-sm text-sm font-medium focus:outline-none focus:ring-2 focus:ring-accent/40 ${msg.author.isAgent ? "text-teal" : "text-text-primary"}`}
              >
                {msg.author.displayName}
              </ProfilePopover>
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
            <ThreadReplyPreview msg={msg} onOpenThread={onOpenThread} />
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
          {copyNotice && (
            copyNotice.status === "copied" ? (
              <div className="mt-1 text-2xs font-medium text-accent">Link copied</div>
            ) : (
              <div className="mt-2 max-w-xl rounded-lg border border-border bg-bg-elevated p-2">
                <div className="mb-1 text-2xs font-medium text-text-muted">Link ready</div>
                <input
                  readOnly
                  value={copyNotice.permalink}
                  onFocus={(event) => event.currentTarget.select()}
                  className="w-full rounded border border-border bg-bg-surface px-2 py-1 text-2xs text-text-secondary outline-none focus:border-accent"
                />
              </div>
            )
          )}
        </div>

        {/* Hover actions */}
        <div ref={menuRef} className="opacity-0 group-hover:opacity-100 flex items-start gap-0.5 pt-0.5 transition-opacity relative">
          <button
            onClick={() => onToggleSaved?.(msg.id, Boolean(msg.isSaved))}
            className={`p-1 rounded hover:bg-bg-hover transition-colors ${
              msg.isSaved ? "text-accent" : "text-text-muted hover:text-text-primary"
            }`}
            title={saveLabel}
            aria-label={saveLabel}
          >
            {msg.isSaved ? <BookmarkCheck size={14} /> : <Bookmark size={14} />}
          </button>
          <button
            onClick={() => void handleCopyLink()}
            className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors"
            title={copyNotice?.status === "copied" ? "Link copied" : "Copy message link"}
            aria-label={copyNotice?.status === "copied" ? "Link copied" : "Copy message link"}
          >
            <Link2 size={14} />
          </button>
          <button
            onClick={() => setShowEmojiPicker(!showEmojiPicker)}
            className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors"
            title="Add reaction"
            aria-label="Add reaction"
          >
            <SmilePlus size={14} />
          </button>
          <button
            onClick={() => onOpenThread?.(msg.id)}
            className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors"
            title="Reply in thread"
            aria-label="Reply in thread"
          >
            <Reply size={14} />
          </button>
          <button
            onClick={() => setShowMenu(!showMenu)}
            className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors"
            title="More message actions"
            aria-label="More message actions"
          >
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
              <button onClick={() => { onToggleSaved?.(msg.id, Boolean(msg.isSaved)); setShowMenu(false); }}
                className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover hover:text-text-primary transition-colors">
                {msg.isSaved ? <BookmarkCheck size={12} /> : <Bookmark size={12} />} {saveLabel}
              </button>
              <button onClick={() => { void handleCopyLink(); setShowMenu(false); }}
                className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover hover:text-text-primary transition-colors">
                <Link2 size={12} /> Copy link
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

function ThreadReplyPreview({
  msg,
  onOpenThread,
}: {
  msg: MessageData;
  onOpenThread?: (messageId: string) => void;
}) {
  const replyCount = msg.replyCount || 0;
  const latestReply = msg.threadPreview?.latestReply;
  const participants = msg.threadPreview?.participants || [];
  const previewText = latestReply?.content.replace(/\s+/g, " ").trim();

  return (
    <button
      type="button"
      onClick={() => onOpenThread?.(msg.id)}
      className="mt-2 flex w-full max-w-2xl items-center gap-3 rounded-lg border border-border bg-bg-surface px-3 py-2 text-left transition-colors hover:border-accent/70 hover:bg-bg-hover"
    >
      {participants.length > 0 ? (
        <div className="flex shrink-0 -space-x-1.5">
          {participants.map((participant) => (
            <ThreadParticipantAvatar key={participant.id} participant={participant} />
          ))}
        </div>
      ) : (
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-border bg-bg-elevated text-accent">
          <Reply size={13} />
        </div>
      )}

      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-center gap-2 text-2xs">
          <span className="shrink-0 font-semibold text-accent">
            {replyCount} {replyCount === 1 ? "reply" : "replies"}
          </span>
          {latestReply && (
            <>
              <span className="text-text-muted">/</span>
              <span className="truncate text-text-muted">
                Latest from {latestReply.author.displayName} {formatDistanceToNow(new Date(latestReply.createdAt), { addSuffix: true })}
              </span>
            </>
          )}
        </div>
        {previewText && (
          <div className="mt-0.5 truncate text-xs text-text-secondary">{previewText}</div>
        )}
      </div>
    </button>
  );
}

function ThreadParticipantAvatar({
  participant,
}: {
  participant: {
    id: string;
    displayName: string;
    avatarUrl: string | null;
    isAgent: boolean;
  };
}) {
  return (
    <ProfilePopover
      user={participant}
      triggerClassName="rounded-full focus:outline-none focus:ring-2 focus:ring-accent/40"
    >
      <span
        className={`flex h-7 w-7 items-center justify-center rounded-full border-2 border-bg text-2xs font-semibold ${
          participant.isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"
        }`}
        title={participant.displayName}
      >
        {participant.avatarUrl ? (
          <img src={participant.avatarUrl} alt="" className="h-full w-full rounded-full object-cover" />
        ) : participant.isAgent ? (
          <Bot size={12} />
        ) : (
          participant.displayName[0]?.toUpperCase()
        )}
      </span>
    </ProfilePopover>
  );
}
