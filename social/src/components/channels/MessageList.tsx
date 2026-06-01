"use client";

import { useRef, useEffect, useState } from "react";
import { formatDistanceToNow, format } from "date-fns";
import {
  AlertCircle,
  ArrowDown,
  Bookmark,
  BookmarkCheck,
  Bot,
  Check,
  CheckCheck,
  Hash,
  Loader2,
  Lock,
  Link2,
  Mail,
  MessageSquare,
  MoreHorizontal,
  Pencil,
  Pin,
  Reply,
  Send,
  SmilePlus,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import MarkdownContent from "./MarkdownContent";
import EmojiPicker from "./EmojiPicker";
import VoicePlayer from "./VoicePlayer";
import ProfilePopover from "@/components/users/ProfilePopover";
import { findFirstUnreadMessageId } from "@/lib/unreadMarker";
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
  hasMoreMessages?: boolean;
  loadingOlder?: boolean;
  olderLoadError?: string | null;
  autoScrollKey?: number;
  showJumpToLatest?: boolean;
  unreadAfter?: string | null;
  canModerateMessages?: boolean;
  onLoadOlder?: () => void;
  onDismissOlderLoadError?: () => void;
  onJumpToLatest?: () => void;
  onReaction?: (messageId: string, emoji: string) => void | Promise<void>;
  onEdit?: (messageId: string, content: string) => void | Promise<void>;
  onDelete?: (messageId: string, reason?: string) => void | Promise<void>;
  onPin?: (messageId: string, isPinned: boolean) => void | Promise<void>;
  onToggleSaved?: (messageId: string, isSaved: boolean) => void | Promise<void>;
  onOpenThread?: (messageId: string) => void;
  onEmailReply?: (
    messageId: string,
    content: string,
  ) => void | Promise<MessageData>;
  onRetryVoiceTranscription?: (messageId: string, audioUrl: string) => void;
}

export default function MessageList({
  messages,
  currentUserId,
  emptyState,
  highlightedMessageId,
  readReceipts,
  hasMoreMessages = false,
  loadingOlder = false,
  olderLoadError = null,
  autoScrollKey,
  showJumpToLatest = false,
  unreadAfter = null,
  canModerateMessages = false,
  onJumpToLatest,
  onLoadOlder,
  onDismissOlderLoadError,
  onReaction,
  onEdit,
  onDelete,
  onPin,
  onToggleSaved,
  onOpenThread,
  onEmailReply,
  onRetryVoiceTranscription,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const autoScrollDependency = autoScrollKey ?? messages.length;

  useEffect(() => {
    if (highlightedMessageId) return;
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [autoScrollDependency, highlightedMessageId]);

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
  const firstUnreadMessageId = findFirstUnreadMessageId(
    messages,
    currentUserId,
    unreadAfter,
  );

  return (
    <div className="relative flex-1 overflow-y-auto">
      <div className="py-2">
        {hasMoreMessages && (
          <div className="flex flex-col items-center gap-2 px-4 py-3">
            <button
              type="button"
              data-testid="load-older-messages"
              onClick={onLoadOlder}
              disabled={loadingOlder}
              aria-label={
                loadingOlder
                  ? "Loading older messages"
                  : olderLoadError
                    ? "Retry older messages"
                    : "Load older messages"
              }
              className="rounded-full border border-border bg-bg-surface px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loadingOlder
                ? "Loading..."
                : olderLoadError
                  ? "Retry older messages"
                  : "Load older messages"}
            </button>
            {olderLoadError && (
              <div
                data-testid="older-messages-error"
                role="alert"
                className="flex max-w-xl items-start gap-2 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
              >
                <AlertCircle size={14} className="mt-0.5 shrink-0" />
                <span className="min-w-0 flex-1">{olderLoadError}</span>
                {onDismissOlderLoadError && (
                  <button
                    type="button"
                    onClick={onDismissOlderLoadError}
                    className="rounded p-0.5 hover:bg-red-100 dark:hover:bg-red-900/40"
                    aria-label="Dismiss older messages error"
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
            )}
          </div>
        )}
        {messages.map((msg, msgIdx) => {
          const msgDate = format(new Date(msg.createdAt), "MMM d, yyyy");
          const showDateDivider = msgDate !== lastDate;
          const showUnreadDivider = msg.id === firstUnreadMessageId;
          const isGrouped =
            msg.author.id === lastAuthor &&
            !showDateDivider &&
            !showUnreadDivider;
          lastAuthor = msg.author.id;
          lastDate = msgDate;

          return (
            <div key={msg.id}>
              {showDateDivider && (
                <div className="flex items-center gap-3 px-4 py-3">
                  <div className="flex-1 h-px bg-border" />
                  <span className="text-2xs font-medium text-text-muted uppercase tracking-wider">
                    {msgDate}
                  </span>
                  <div className="flex-1 h-px bg-border" />
                </div>
              )}
              {showUnreadDivider && <UnreadDivider />}
              <MessageRow
                msg={msg}
                isGrouped={isGrouped}
                isOwn={msg.author.id === currentUserId}
                readByCount={getReadByCount(msgIdx)}
                channelKind={emptyState.kind}
                isHighlighted={msg.id === highlightedMessageId}
                canModerateMessages={canModerateMessages}
                onReaction={onReaction}
                onEdit={onEdit}
                onDelete={onDelete}
                onPin={onPin}
                onToggleSaved={onToggleSaved}
                onOpenThread={onOpenThread}
                onEmailReply={onEmailReply}
                onRetryVoiceTranscription={onRetryVoiceTranscription}
              />
            </div>
          );
        })}
        <div ref={bottomRef} />
        {showJumpToLatest && (
          <div className="sticky bottom-3 z-10 flex justify-center px-4 pointer-events-none">
            <button
              type="button"
              data-testid="jump-to-latest"
              onClick={onJumpToLatest}
              aria-label="Jump to latest messages"
              className="pointer-events-auto inline-flex items-center gap-1.5 rounded-full border border-accent/40 bg-bg-surface px-3 py-1.5 text-xs font-semibold text-accent shadow-lg transition-colors hover:bg-accent-muted"
            >
              <ArrowDown size={13} />
              <span>Jump to latest</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function UnreadDivider() {
  return (
    <div
      role="separator"
      aria-label="Unread messages"
      className="flex items-center gap-3 px-4 py-2"
    >
      <div className="h-px flex-1 bg-danger" />
      <span className="rounded-full bg-danger-muted px-2 py-0.5 text-2xs font-semibold uppercase tracking-wider text-danger">
        New
      </span>
      <div className="h-px flex-1 bg-danger" />
    </div>
  );
}

function EmptyConversation({ state }: { state: MessageEmptyState }) {
  const isDm = state.kind === "dm";
  const Icon = isDm
    ? state.isAgent
      ? Bot
      : MessageSquare
    : state.isPrivate
      ? Lock
      : Hash;
  const title = isDm
    ? state.isAgent
      ? `Ask ${state.name} for help`
      : `Start a DM with ${state.name}`
    : `Start #${state.name}`;
  const subtitle = isDm
    ? state.isAgent
      ? `${state.name} is ready for drafts, research, summaries, and follow-ups.`
      : `This conversation is just between you and ${state.name}.`
    : state.description ||
      `${state.name} is ready for team updates, questions, and decisions.`;
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
        <h3 className="mb-2 font-heading text-xl font-semibold text-text-primary">
          {title}
        </h3>
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

function formatEmailAddressLabel(address?: { name?: string; email?: string }) {
  if (!address) return "sender";
  if (address.name && address.email)
    return `${address.name} <${address.email}>`;
  return address.name || address.email || "sender";
}

function MessageRow({
  msg,
  isGrouped,
  isOwn,
  readByCount,
  channelKind,
  isHighlighted,
  canModerateMessages,
  onReaction,
  onEdit,
  onDelete,
  onPin,
  onToggleSaved,
  onOpenThread,
  onEmailReply,
  onRetryVoiceTranscription,
}: {
  msg: MessageData;
  isGrouped: boolean;
  isOwn: boolean;
  readByCount: number;
  channelKind: "channel" | "dm";
  isHighlighted: boolean;
  canModerateMessages: boolean;
  onReaction?: (messageId: string, emoji: string) => void | Promise<void>;
  onEdit?: (messageId: string, content: string) => void | Promise<void>;
  onDelete?: (messageId: string, reason?: string) => void | Promise<void>;
  onPin?: (messageId: string, isPinned: boolean) => void | Promise<void>;
  onToggleSaved?: (messageId: string, isSaved: boolean) => void | Promise<void>;
  onOpenThread?: (messageId: string) => void;
  onEmailReply?: (
    messageId: string,
    content: string,
  ) => void | Promise<MessageData>;
  onRetryVoiceTranscription?: (messageId: string, audioUrl: string) => void;
}) {
  const [showMenu, setShowMenu] = useState(false);
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const [editing, setEditing] = useState(false);
  const [emailReplyOpen, setEmailReplyOpen] = useState(false);
  const [emailReplyContent, setEmailReplyContent] = useState("");
  const [emailReplyNotice, setEmailReplyNotice] = useState<string | null>(null);
  const [copyNotice, setCopyNotice] = useState<{
    status: "copied" | "blocked";
    permalink: string;
  } | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [editContent, setEditContent] = useState(msg.content);
  const menuRef = useRef<HTMLDivElement>(null);
  const permalinkInputRef = useRef<HTMLInputElement>(null);
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout>>();
  const saveLabel = msg.isSaved ? "Remove from Later" : "Save for later";
  const canDeleteMessage = isOwn || canModerateMessages;
  const actionBusy = Boolean(pendingAction);
  const importedEmail =
    msg.metadata?.type === "email_import" ? msg.metadata.email : undefined;
  const canReplyToImportedEmail = Boolean(
    importedEmail?.from?.email && onEmailReply,
  );
  const emailReplyRecipient = formatEmailAddressLabel(importedEmail?.from);

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

  useEffect(() => {
    if (copyNotice?.status !== "blocked") return;

    const timeout = window.setTimeout(() => {
      permalinkInputRef.current?.focus();
      permalinkInputRef.current?.select();
    }, 0);

    return () => window.clearTimeout(timeout);
  }, [copyNotice]);

  const runAction = async (
    actionKey: string,
    actionLabel: string,
    action: () => void | Promise<void>,
    onSuccess?: () => void,
  ) => {
    if (pendingAction) return;

    setPendingAction(actionKey);
    setActionError(null);
    try {
      await action();
      onSuccess?.();
    } catch (error) {
      const message = error instanceof Error ? error.message : null;
      setActionError(message || `${actionLabel} failed.`);
    } finally {
      setPendingAction(null);
    }
  };

  const handleSaveEdit = () => {
    const trimmedContent = editContent.trim();
    if (!trimmedContent) {
      setActionError("Message content is required.");
      return;
    }

    if (trimmedContent === msg.content) {
      setEditing(false);
      return;
    }

    void runAction(
      "edit",
      "Edit message",
      () => onEdit?.(msg.id, trimmedContent),
      () => setEditing(false),
    );
  };

  const handleSelectReaction = (emoji: string) => {
    void runAction(
      "reaction",
      "Reaction",
      () => onReaction?.(msg.id, emoji),
      () => setShowEmojiPicker(false),
    );
  };

  const handleToggleSaved = () => {
    void runAction("saved", saveLabel, () =>
      onToggleSaved?.(msg.id, Boolean(msg.isSaved)),
    );
  };

  const handleTogglePin = () => {
    void runAction(
      "pin",
      msg.isPinned ? "Unpin message" : "Pin message",
      () => onPin?.(msg.id, msg.isPinned),
      () => setShowMenu(false),
    );
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
      setCopyNotice({ status: "blocked", permalink });
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
    }
  };

  const handleSelectPermalink = () => {
    permalinkInputRef.current?.focus();
    permalinkInputRef.current?.select();
  };

  const handleDeleteMessage = () => {
    if (!canDeleteMessage) return;

    if (isOwn) {
      void runAction(
        "delete",
        "Delete message",
        () => onDelete?.(msg.id),
        () => setShowMenu(false),
      );
      return;
    }

    const reason = window.prompt("Reason for removing this message (optional)");
    if (reason === null) return;
    void runAction(
      "delete",
      "Delete message",
      () => onDelete?.(msg.id, reason.trim() || undefined),
      () => setShowMenu(false),
    );
  };

  const handleSendEmailReply = () => {
    if (!onEmailReply) return;

    const trimmedContent = emailReplyContent.trim();
    if (!trimmedContent) {
      setActionError("Email reply content is required.");
      return;
    }

    void runAction(
      "email-reply",
      "Email reply",
      async () => {
        await onEmailReply(msg.id, trimmedContent);
      },
      () => {
        setEmailReplyContent("");
        setEmailReplyOpen(false);
        setEmailReplyNotice("Email sent");
        setShowMenu(false);
      },
    );
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
              {new Date(msg.createdAt).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          </div>
        ) : (
          <ProfilePopover
            user={msg.author}
            className="mt-0.5 shrink-0"
            triggerClassName="rounded-full focus:outline-none focus:ring-2 focus:ring-accent/40"
          >
            <span
              className={`w-8 h-8 avatar text-xs ${msg.author.isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"}`}
            >
              {msg.author.isAgent ? (
                <Bot size={14} />
              ) : (
                msg.author.displayName[0]?.toUpperCase()
              )}
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
                {formatDistanceToNow(new Date(msg.createdAt), {
                  addSuffix: true,
                })}
              </span>
              {msg.isEdited && (
                <span className="text-2xs text-text-muted">(edited)</span>
              )}
              {msg.isPinned && <Pin size={10} className="text-accent" />}
            </div>
          )}

          {importedEmail && (
            <div className="mb-2 max-w-2xl rounded-lg border border-border bg-bg-surface px-3 py-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex min-w-0 items-center gap-2 text-xs text-text-secondary">
                  <Mail size={13} className="shrink-0 text-accent" />
                  <span className="truncate">
                    Reply to {emailReplyRecipient}
                  </span>
                </div>
                {canReplyToImportedEmail && (
                  <button
                    type="button"
                    data-testid="email-reply-toggle"
                    onClick={() => {
                      setEmailReplyOpen((open) => !open);
                      setActionError(null);
                      setEmailReplyNotice(null);
                    }}
                    disabled={actionBusy}
                    className="inline-flex items-center gap-1.5 rounded-md border border-border bg-bg-elevated px-2 py-1 text-2xs font-semibold text-text-secondary transition-colors hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <Reply size={12} />
                    Reply by email
                  </button>
                )}
              </div>

              {emailReplyOpen && canReplyToImportedEmail && (
                <div className="mt-2 space-y-2">
                  <textarea
                    data-testid="email-reply-input"
                    aria-label={`Email reply to ${emailReplyRecipient}`}
                    value={emailReplyContent}
                    onChange={(event) => {
                      setEmailReplyContent(event.target.value);
                      setActionError(null);
                    }}
                    className="min-h-24 w-full resize-y rounded-md border border-border bg-bg-base px-3 py-2 text-sm text-text-primary outline-none transition-colors placeholder:text-text-muted focus:border-accent"
                    placeholder="Write an email reply..."
                    disabled={actionBusy}
                    autoFocus
                  />
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      data-testid="email-reply-send"
                      aria-label="Send email reply"
                      onClick={handleSendEmailReply}
                      disabled={actionBusy}
                      className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {pendingAction === "email-reply" ? (
                        <Loader2 size={13} className="animate-spin" />
                      ) : (
                        <Send size={13} />
                      )}
                      Send email
                    </button>
                    <button
                      type="button"
                      data-testid="email-reply-cancel"
                      aria-label="Cancel email reply"
                      onClick={() => {
                        setEmailReplyOpen(false);
                        setActionError(null);
                      }}
                      disabled={actionBusy}
                      className="rounded-md border border-border bg-bg-elevated px-3 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:border-text-muted hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              {emailReplyNotice && (
                <div
                  data-testid="email-reply-notice"
                  className="mt-2 flex items-center gap-1.5 text-2xs font-medium text-teal"
                >
                  <Check size={11} />
                  {emailReplyNotice}
                </div>
              )}
            </div>
          )}

          {editing ? (
            <div className="space-y-2">
              <textarea
                data-testid="message-edit-input"
                aria-label="Edit message content"
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full bg-bg-elevated border border-border rounded-lg px-3 py-2 text-sm text-text-primary resize-none focus:outline-none focus:border-accent"
                rows={Math.min(editContent.split("\n").length + 1, 6)}
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSaveEdit();
                  }
                  if (e.key === "Escape") {
                    setEditContent(msg.content);
                    setEditing(false);
                    setActionError(null);
                  }
                }}
              />
              <div className="flex items-center gap-2 text-2xs text-text-muted">
                <button
                  type="button"
                  data-testid="message-edit-save"
                  aria-label="Save message edit"
                  onClick={handleSaveEdit}
                  disabled={actionBusy}
                  className="flex items-center gap-1 text-teal hover:underline disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {pendingAction === "edit" ? (
                    <Loader2 size={10} className="animate-spin" />
                  ) : (
                    <Check size={10} />
                  )}
                  Save
                </button>
                <span>·</span>
                <button
                  type="button"
                  data-testid="message-edit-cancel"
                  aria-label="Cancel message edit"
                  onClick={() => {
                    setEditContent(msg.content);
                    setEditing(false);
                    setActionError(null);
                  }}
                  disabled={actionBusy}
                  className="flex items-center gap-1 hover:underline disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <X size={10} /> Cancel
                </button>
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
                transcriptionStatus={msg.metadata.transcriptionStatus}
                transcriptionError={msg.metadata.transcriptionError}
                isAgent={msg.author.isAgent}
                onRetryTranscription={() =>
                  onRetryVoiceTranscription?.(msg.id, msg.attachments[0].url)
                }
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
                  <a
                    key={att.id}
                    href={att.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label={`Open attachment ${att.fileName}`}
                  >
                    <img
                      src={att.url}
                      alt={att.fileName}
                      className="max-w-sm max-h-64 rounded-lg border border-border"
                    />
                  </a>
                ) : (
                  <a
                    key={att.id}
                    href={att.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label={`Open attachment ${att.fileName}`}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg bg-bg-elevated border border-border hover:border-accent transition-colors text-xs"
                  >
                    <span>📎</span>
                    <span className="text-text-primary truncate max-w-[200px]">
                      {att.fileName}
                    </span>
                  </a>
                );
              })}
            </div>
          )}

          {msg.reactions.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {msg.reactions.map((r) => (
                <button
                  key={r.emoji}
                  type="button"
                  onClick={() => handleSelectReaction(r.emoji)}
                  disabled={actionBusy}
                  aria-label={`Toggle ${r.emoji} reaction`}
                  className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-2xs border transition-colors ${
                    r.userReacted
                      ? "border-accent bg-accent-muted text-accent"
                      : "border-border bg-bg-elevated text-text-secondary hover:border-text-muted"
                  } disabled:cursor-not-allowed disabled:opacity-60`}
                >
                  <span>{r.emoji}</span>
                  <span>{r.count}</span>
                </button>
              ))}
              <button
                type="button"
                onClick={() => setShowEmojiPicker(true)}
                disabled={actionBusy}
                aria-label="Add another reaction"
                className="flex items-center px-1.5 py-0.5 rounded-full text-2xs border border-dashed border-border text-text-muted hover:border-text-muted transition-colors disabled:cursor-not-allowed disabled:opacity-60"
              >
                <SmilePlus size={10} />
              </button>
            </div>
          )}

          {(msg.replyCount || 0) > 0 && (
            <ThreadReplyPreview msg={msg} onOpenThread={onOpenThread} />
          )}

          {/* Read receipts — show for own messages */}
          {isOwn && (
            <div
              className="flex items-center gap-1 mt-1 text-2xs"
              title={readByCount > 0 ? `Seen by ${readByCount}` : "Sent"}
            >
              {readByCount > 0 ? (
                <>
                  <CheckCheck size={12} className="text-teal" />
                  {readByCount > 1 && (
                    <span className="text-text-muted">
                      Seen by {readByCount}
                    </span>
                  )}
                </>
              ) : (
                <Check size={12} className="text-text-muted/50" />
              )}
            </div>
          )}
          {copyNotice &&
            (copyNotice.status === "copied" ? (
              <div className="mt-1 text-2xs font-medium text-accent">
                Link copied
              </div>
            ) : (
              <div
                data-testid="message-copy-fallback"
                className="mt-2 max-w-xl rounded-lg border border-accent/40 bg-accent-muted/40 p-3"
              >
                <div className="mb-2 flex items-start gap-2 text-xs text-text-secondary">
                  <AlertCircle
                    size={14}
                    className="mt-0.5 shrink-0 text-accent"
                  />
                  <div>
                    <div className="font-semibold text-text-primary">
                      Clipboard access is blocked
                    </div>
                    <div className="mt-0.5 text-text-muted">
                      Select this permalink and copy it manually.
                    </div>
                  </div>
                </div>
                <input
                  ref={permalinkInputRef}
                  data-testid="message-copy-fallback-input"
                  aria-label="Message permalink"
                  readOnly
                  value={copyNotice.permalink}
                  onFocus={(event) => event.currentTarget.select()}
                  className="w-full rounded border border-border bg-bg-surface px-2 py-1 text-2xs text-text-secondary outline-none focus:border-accent"
                />
                <div className="mt-2 flex justify-end">
                  <button
                    type="button"
                    data-testid="message-copy-fallback-select"
                    onClick={handleSelectPermalink}
                    className="rounded-md border border-border bg-bg-surface px-2 py-1 text-2xs font-medium text-text-secondary hover:border-accent hover:text-accent"
                  >
                    Select link
                  </button>
                </div>
              </div>
            ))}
          {actionError && (
            <div
              data-testid="message-action-error"
              className="mt-2 flex max-w-xl items-start gap-2 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
            >
              <AlertCircle size={14} className="mt-0.5 shrink-0" />
              <div className="min-w-0 flex-1">{actionError}</div>
              <button
                type="button"
                onClick={() => setActionError(null)}
                className="rounded p-0.5 hover:bg-red-100 dark:hover:bg-red-900/40"
                aria-label="Dismiss message action error"
              >
                <X size={12} />
              </button>
            </div>
          )}
          {pendingAction && !editing && (
            <div
              data-testid="message-action-pending"
              className="mt-1 flex items-center gap-1.5 text-2xs text-text-muted"
            >
              <Loader2 size={11} className="animate-spin" />
              <span>
                {pendingAction === "email-reply"
                  ? "Sending email..."
                  : "Updating message..."}
              </span>
            </div>
          )}
        </div>

        {/* Hover actions */}
        <div
          ref={menuRef}
          className="opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 flex items-start gap-0.5 pt-0.5 transition-opacity relative"
        >
          {canReplyToImportedEmail && (
            <button
              type="button"
              onClick={() => {
                setEmailReplyOpen(true);
                setActionError(null);
                setEmailReplyNotice(null);
              }}
              disabled={actionBusy}
              className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors disabled:cursor-not-allowed disabled:opacity-60"
              title="Reply by email"
              aria-label="Reply by email"
            >
              <Mail size={14} />
            </button>
          )}
          <button
            type="button"
            onClick={handleToggleSaved}
            disabled={actionBusy}
            className={`p-1 rounded hover:bg-bg-hover transition-colors ${
              msg.isSaved
                ? "text-accent"
                : "text-text-muted hover:text-text-primary"
            } disabled:cursor-not-allowed disabled:opacity-60`}
            title={saveLabel}
            aria-label={saveLabel}
          >
            {pendingAction === "saved" ? (
              <Loader2 size={14} className="animate-spin" />
            ) : msg.isSaved ? (
              <BookmarkCheck size={14} />
            ) : (
              <Bookmark size={14} />
            )}
          </button>
          <button
            type="button"
            onClick={() => void handleCopyLink()}
            className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors"
            title={
              copyNotice?.status === "copied"
                ? "Link copied"
                : "Copy message link"
            }
            aria-label={
              copyNotice?.status === "copied"
                ? "Link copied"
                : "Copy message link"
            }
          >
            <Link2 size={14} />
          </button>
          <button
            type="button"
            onClick={() => {
              setShowEmojiPicker(!showEmojiPicker);
              setShowMenu(false);
            }}
            disabled={actionBusy}
            className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors disabled:cursor-not-allowed disabled:opacity-60"
            title="Add reaction"
            aria-label="Add reaction"
            aria-expanded={showEmojiPicker}
          >
            <SmilePlus size={14} />
          </button>
          <button
            type="button"
            onClick={() => onOpenThread?.(msg.id)}
            className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors"
            title="Reply in thread"
            aria-label="Reply in thread"
          >
            <Reply size={14} />
          </button>
          <button
            type="button"
            onClick={() => {
              setShowMenu(!showMenu);
              setShowEmojiPicker(false);
            }}
            disabled={actionBusy}
            className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors disabled:cursor-not-allowed disabled:opacity-60"
            title="More message actions"
            aria-label="More message actions"
            aria-expanded={showMenu}
          >
            {pendingAction && pendingAction !== "reaction" ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <MoreHorizontal size={14} />
            )}
          </button>

          {showEmojiPicker && (
            <div className="absolute right-0 top-8 z-50">
              <EmojiPicker
                onSelect={handleSelectReaction}
                onClose={() => setShowEmojiPicker(false)}
              />
            </div>
          )}

          {showMenu && (
            <div
              className="absolute right-0 top-8 z-50 w-44 bg-bg-surface border border-border rounded-lg shadow-lg py-1"
              role="menu"
            >
              {isOwn && (
                <button
                  type="button"
                  role="menuitem"
                  onClick={() => {
                    setEditing(true);
                    setActionError(null);
                    setShowMenu(false);
                  }}
                  disabled={actionBusy}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover hover:text-text-primary transition-colors disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <Pencil size={12} /> Edit message
                </button>
              )}
              <button
                type="button"
                role="menuitem"
                onClick={handleTogglePin}
                disabled={actionBusy}
                className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover hover:text-text-primary transition-colors disabled:cursor-not-allowed disabled:opacity-60"
              >
                {pendingAction === "pin" ? (
                  <Loader2 size={12} className="animate-spin" />
                ) : (
                  <Pin size={12} />
                )}
                {msg.isPinned ? "Unpin" : "Pin"} message
              </button>
              <button
                type="button"
                role="menuitem"
                onClick={() =>
                  void runAction(
                    "saved",
                    saveLabel,
                    () => onToggleSaved?.(msg.id, Boolean(msg.isSaved)),
                    () => setShowMenu(false),
                  )
                }
                disabled={actionBusy}
                className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover hover:text-text-primary transition-colors disabled:cursor-not-allowed disabled:opacity-60"
              >
                {pendingAction === "saved" ? (
                  <Loader2 size={12} className="animate-spin" />
                ) : msg.isSaved ? (
                  <BookmarkCheck size={12} />
                ) : (
                  <Bookmark size={12} />
                )}
                {saveLabel}
              </button>
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  void handleCopyLink();
                  setShowMenu(false);
                }}
                className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-text-secondary hover:bg-bg-hover hover:text-text-primary transition-colors"
              >
                <Link2 size={12} /> Copy link
              </button>
              {canDeleteMessage && (
                <button
                  type="button"
                  role="menuitem"
                  onClick={handleDeleteMessage}
                  disabled={actionBusy}
                  className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-danger hover:bg-danger-muted transition-colors disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {pendingAction === "delete" ? (
                    <Loader2 size={12} className="animate-spin" />
                  ) : (
                    <Trash2 size={12} />
                  )}
                  Delete message
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
            <ThreadParticipantAvatar
              key={participant.id}
              participant={participant}
            />
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
                Latest from {latestReply.author.displayName}{" "}
                {formatDistanceToNow(new Date(latestReply.createdAt), {
                  addSuffix: true,
                })}
              </span>
            </>
          )}
        </div>
        {previewText && (
          <div className="mt-0.5 truncate text-xs text-text-secondary">
            {previewText}
          </div>
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
          participant.isAgent
            ? "bg-teal-muted text-teal"
            : "bg-accent-muted text-accent"
        }`}
        title={participant.displayName}
      >
        {participant.avatarUrl ? (
          <img
            src={participant.avatarUrl}
            alt=""
            className="h-full w-full rounded-full object-cover"
          />
        ) : participant.isAgent ? (
          <Bot size={12} />
        ) : (
          participant.displayName[0]?.toUpperCase()
        )}
      </span>
    </ProfilePopover>
  );
}
