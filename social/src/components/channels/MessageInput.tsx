"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import {
  AtSign,
  Bot,
  ClipboardList,
  HelpCircle,
  ListChecks,
  Loader2,
  Megaphone,
  Mic,
  Paperclip,
  Search,
  Send,
  Smile,
  X,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import VoiceRecorder from "./VoiceRecorder";
import {
  clearMessageDraft,
  getBrowserMessageDraftStorage,
  readMessageDraft,
  writeMessageDraft,
} from "@/lib/messageDrafts";
import { apiUrl } from "@/lib/apiUrl";

type SlashCommand = {
  name: string;
  title: string;
  description: string;
  insertText: string;
  Icon: LucideIcon;
};

type MentionUser = {
  id: string;
  username: string;
  displayName: string;
  avatarUrl: string | null;
  isAgent: boolean;
  status: string;
};

async function responseErrorMessage(res: Response, fallback: string) {
  const payload = (await res.json().catch(() => null)) as {
    error?: unknown;
    message?: unknown;
  } | null;

  if (typeof payload?.error === "string" && payload.error.trim()) {
    return payload.error;
  }

  if (typeof payload?.message === "string" && payload.message.trim()) {
    return payload.message;
  }

  return fallback;
}

const SLASH_COMMANDS: SlashCommand[] = [
  {
    name: "todo",
    title: "Action list",
    description: "Start a checklist for follow-ups.",
    insertText: "- [ ] ",
    Icon: ListChecks,
  },
  {
    name: "decision",
    title: "Decision note",
    description: "Capture context, owner, and next step.",
    insertText: "Decision:\nContext:\nOwner:\nNext step:",
    Icon: ClipboardList,
  },
  {
    name: "handoff",
    title: "Handoff",
    description: "Summarize status, blockers, and next moves.",
    insertText: "Handoff:\nStatus:\nBlockers:\nNext:",
    Icon: Megaphone,
  },
  {
    name: "help",
    title: "Command list",
    description: "Insert the current command catalog.",
    insertText:
      "Available commands:\n/todo - action list\n/decision - decision note\n/handoff - handoff\n/help - command list",
    Icon: HelpCircle,
  },
];

const EMOJI_OPTIONS = [
  { label: "Thumbs up", value: "👍" },
  { label: "Raised hands", value: "🙌" },
  { label: "Fire", value: "🔥" },
  { label: "Heart", value: "❤️" },
  { label: "Eyes", value: "👀" },
  { label: "Check mark", value: "✅" },
  { label: "Sparkles", value: "✨" },
  { label: "Thinking", value: "🤔" },
];

interface MessageInputProps {
  channelId: string;
  channelName: string;
  onSend: (content: string) => void | Promise<void>;
  onTyping?: () => void;
  disabled?: boolean;
  placeholder?: string;
  draftId?: string;
  onVoiceSend?: (audioBlob: Blob, duration: number) => Promise<void>;
  onFileUpload?: (file: File) => Promise<void>;
}

export default function MessageInput({
  channelId,
  channelName,
  onSend,
  onTyping,
  disabled,
  placeholder,
  draftId,
  onVoiceSend,
  onFileUpload,
}: MessageInputProps) {
  const [content, setContent] = useState("");
  const [recording, setRecording] = useState(false);
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const [showMentionPicker, setShowMentionPicker] = useState(false);
  const [mentionQuery, setMentionQuery] = useState("");
  const [mentionResults, setMentionResults] = useState<MentionUser[]>([]);
  const [mentionLoading, setMentionLoading] = useState(false);
  const [mentionError, setMentionError] = useState<string | null>(null);
  const [mentionRetryKey, setMentionRetryKey] = useState(0);
  const [composerError, setComposerError] = useState<string | null>(null);
  const [uploadingFile, setUploadingFile] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [failedUploadFile, setFailedUploadFile] = useState<File | null>(null);
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);
  const [slashMenuDismissed, setSlashMenuDismissed] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const mentionInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const typingTimeout = useRef<ReturnType<typeof setTimeout>>();
  const skipNextDraftSaveRef = useRef(false);
  const resolvedDraftId = draftId || channelId;

  const slashMatch = content.match(/^\/([a-z-]*)$/i);
  const slashQuery = slashMatch?.[1].toLowerCase() ?? null;
  const filteredSlashCommands = useMemo(() => {
    if (slashQuery === null) return [];
    return SLASH_COMMANDS.filter(
      (command) =>
        command.name.includes(slashQuery) ||
        command.title.toLowerCase().includes(slashQuery),
    );
  }, [slashQuery]);
  const showSlashCommands = slashQuery !== null && !slashMenuDismissed;

  useEffect(() => {
    setSelectedCommandIndex(0);
  }, [slashQuery]);

  useEffect(() => {
    if (!showMentionPicker) return;
    const focusTimer = window.setTimeout(
      () => mentionInputRef.current?.focus(),
      0,
    );
    return () => window.clearTimeout(focusTimer);
  }, [showMentionPicker]);

  useEffect(() => {
    if (!showMentionPicker || mentionQuery.trim().length < 2) {
      setMentionResults([]);
      setMentionLoading(false);
      setMentionError(null);
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setMentionLoading(true);
      setMentionError(null);
      try {
        const response = await fetch(
          apiUrl(
            `/api/users/search?q=${encodeURIComponent(mentionQuery.trim())}`,
          ),
          { signal: controller.signal },
        );
        if (!response.ok) {
          throw new Error(
            await responseErrorMessage(
              response,
              "Mention search temporarily unavailable.",
            ),
          );
        }
        setMentionResults((await response.json()) as MentionUser[]);
      } catch (error) {
        if (!controller.signal.aborted) {
          const message =
            error instanceof Error && error.message.trim()
              ? error.message
              : "Mention search temporarily unavailable.";
          setMentionResults([]);
          setMentionError(message);
        }
      } finally {
        if (!controller.signal.aborted) setMentionLoading(false);
      }
    }, 180);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [mentionQuery, mentionRetryKey, showMentionPicker]);

  useEffect(() => {
    if (selectedCommandIndex < filteredSlashCommands.length) return;
    setSelectedCommandIndex(Math.max(filteredSlashCommands.length - 1, 0));
  }, [filteredSlashCommands.length, selectedCommandIndex]);

  const resizeTextarea = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
  }, []);

  useEffect(() => {
    skipNextDraftSaveRef.current = true;
    setContent(
      readMessageDraft(resolvedDraftId, getBrowserMessageDraftStorage()),
    );
    setSlashMenuDismissed(false);
  }, [resolvedDraftId]);

  useEffect(() => {
    resizeTextarea();
  }, [content, resizeTextarea]);

  useEffect(() => {
    if (skipNextDraftSaveRef.current) {
      skipNextDraftSaveRef.current = false;
      return;
    }

    writeMessageDraft(
      resolvedDraftId,
      content,
      getBrowserMessageDraftStorage(),
    );
  }, [content, resolvedDraftId]);

  const applySlashCommand = useCallback(
    (command: SlashCommand) => {
      setContent(command.insertText);
      setShowEmojiPicker(false);
      setShowMentionPicker(false);
      setSlashMenuDismissed(false);
      window.setTimeout(() => {
        const textarea = textareaRef.current;
        if (!textarea) return;
        textarea.focus();
        const cursorPosition = command.insertText.length;
        textarea.selectionStart = cursorPosition;
        textarea.selectionEnd = cursorPosition;
        resizeTextarea();
      }, 0);
    },
    [resizeTextarea],
  );

  const insertTextAtCursor = useCallback(
    (text: string) => {
      const textarea = textareaRef.current;
      const start = textarea?.selectionStart ?? content.length;
      const end = textarea?.selectionEnd ?? content.length;
      const nextContent = content.slice(0, start) + text + content.slice(end);

      setContent(nextContent);
      setComposerError(null);
      setSlashMenuDismissed(false);

      window.setTimeout(() => {
        const activeTextarea = textareaRef.current;
        if (!activeTextarea) return;
        const cursorPosition = start + text.length;
        activeTextarea.focus();
        activeTextarea.selectionStart = cursorPosition;
        activeTextarea.selectionEnd = cursorPosition;
        resizeTextarea();
      }, 0);
    },
    [content, resizeTextarea],
  );

  const handleSubmit = useCallback(async () => {
    const trimmed = content.trim();
    if (!trimmed || disabled) return;
    if (showSlashCommands) {
      const selectedCommand = filteredSlashCommands[selectedCommandIndex];
      if (selectedCommand) {
        applySlashCommand(selectedCommand);
      }
      return;
    }
    if (slashQuery !== null && !content.includes(" ")) return;

    try {
      await onSend(trimmed);
      clearMessageDraft(resolvedDraftId, getBrowserMessageDraftStorage());
      setContent("");
      setShowEmojiPicker(false);
      setShowMentionPicker(false);
      setComposerError(null);
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    } catch (error) {
      writeMessageDraft(
        resolvedDraftId,
        content,
        getBrowserMessageDraftStorage(),
      );
      const message =
        error instanceof Error && error.message.trim()
          ? error.message
          : "Message could not be sent.";
      setComposerError(`${message} Draft saved.`);
    }
  }, [
    applySlashCommand,
    content,
    disabled,
    filteredSlashCommands,
    onSend,
    resolvedDraftId,
    selectedCommandIndex,
    showSlashCommands,
    slashQuery,
  ]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showSlashCommands) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedCommandIndex(
          (current) =>
            (current + 1) % Math.max(filteredSlashCommands.length, 1),
        );
        return;
      }

      if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedCommandIndex(
          (current) =>
            (current - 1 + Math.max(filteredSlashCommands.length, 1)) %
            Math.max(filteredSlashCommands.length, 1),
        );
        return;
      }

      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        const selectedCommand = filteredSlashCommands[selectedCommandIndex];
        if (selectedCommand) {
          applySlashCommand(selectedCommand);
        }
        return;
      }

      if (e.key === "Escape") {
        e.preventDefault();
        setSlashMenuDismissed(true);
        return;
      }
    }

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSubmit();
    }
    // Markdown shortcuts
    if ((e.metaKey || e.ctrlKey) && e.key === "b") {
      e.preventDefault();
      wrapSelection("**");
    }
    if ((e.metaKey || e.ctrlKey) && e.key === "i") {
      e.preventDefault();
      wrapSelection("_");
    }
  };

  const wrapSelection = (wrapper: string) => {
    const ta = textareaRef.current;
    if (!ta) return;
    const start = ta.selectionStart;
    const end = ta.selectionEnd;
    const selected = content.substring(start, end);
    const newContent =
      content.substring(0, start) +
      wrapper +
      selected +
      wrapper +
      content.substring(end);
    setContent(newContent);
    setTimeout(() => {
      ta.selectionStart = start + wrapper.length;
      ta.selectionEnd = end + wrapper.length;
      ta.focus();
    }, 0);
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setContent(e.target.value);
    setShowEmojiPicker(false);
    setSlashMenuDismissed(false);
    setComposerError(null);
    resizeTextarea();

    if (onTyping) {
      if (typingTimeout.current) clearTimeout(typingTimeout.current);
      onTyping();
      typingTimeout.current = setTimeout(() => {}, 3000);
    }
  };

  const handleVoiceSend = async (audioBlob: Blob, duration: number) => {
    if (onVoiceSend) {
      await onVoiceSend(audioBlob, duration);
    }
    setComposerError(null);
    setRecording(false);
  };

  const uploadSelectedFile = async (file: File) => {
    if (!onFileUpload) return;

    setUploadingFile(true);
    setUploadError(null);
    setFailedUploadFile(null);
    try {
      await onFileUpload(file);
    } catch (error) {
      const message =
        error instanceof Error && error.message.trim()
          ? error.message
          : "File could not be uploaded.";
      setUploadError(message);
      setFailedUploadFile(file);
    } finally {
      setUploadingFile(false);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && onFileUpload) {
      await uploadSelectedFile(file);
    }
    // Reset input
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const openMentionPicker = () => {
    setShowEmojiPicker(false);
    setShowMentionPicker(true);
    setMentionQuery("");
    setMentionResults([]);
    setMentionError(null);
    setComposerError(null);
  };

  const closeMentionPicker = () => {
    setShowMentionPicker(false);
    setMentionQuery("");
    setMentionResults([]);
    setMentionError(null);
    textareaRef.current?.focus();
  };

  const mentionLabel = (user: MentionUser) =>
    user.username ||
    user.displayName
      .toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-")
      .replace(/^-|-$/g, "");

  const insertMention = (user: MentionUser) => {
    const label = mentionLabel(user);
    if (!label) return;
    insertTextAtCursor(`@${label} `);
    setShowMentionPicker(false);
    setMentionQuery("");
    setMentionResults([]);
    setMentionError(null);
  };

  if (recording) {
    return (
      <VoiceRecorder
        onSend={handleVoiceSend}
        onCancel={() => setRecording(false)}
      />
    );
  }

  const composerPlaceholder = placeholder || `Message ${channelName}`;

  return (
    <div className="px-4 pb-4 pt-2">
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        onChange={handleFileChange}
        accept="image/*,video/*,audio/*,.pdf,.doc,.docx,.txt,.zip,.csv,.xlsx"
      />
      {showSlashCommands && (
        <div className="mb-2 overflow-hidden rounded-lg border border-border bg-bg-surface shadow-lg">
          <div className="border-b border-border px-3 py-2 text-2xs font-semibold uppercase text-text-muted">
            Commands
          </div>
          <div className="max-h-64 overflow-y-auto py-1">
            {filteredSlashCommands.length > 0 ? (
              filteredSlashCommands.map((command, index) => {
                const Icon = command.Icon;
                const selected = index === selectedCommandIndex;

                return (
                  <button
                    key={command.name}
                    type="button"
                    onMouseDown={(event) => {
                      event.preventDefault();
                      applySlashCommand(command);
                    }}
                    onMouseEnter={() => setSelectedCommandIndex(index)}
                    className={`flex w-full items-center gap-3 px-3 py-2 text-left transition-colors ${
                      selected
                        ? "bg-bg-hover text-text-primary"
                        : "text-text-secondary hover:bg-bg-hover hover:text-text-primary"
                    }`}
                  >
                    <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border bg-bg-elevated text-accent">
                      <Icon size={15} />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="flex min-w-0 items-baseline gap-2">
                        <span className="shrink-0 text-sm font-semibold text-text-primary">
                          /{command.name}
                        </span>
                        <span className="truncate text-xs text-text-muted">
                          {command.title}
                        </span>
                      </span>
                      <span className="block truncate text-xs text-text-muted">
                        {command.description}
                      </span>
                    </span>
                  </button>
                );
              })
            ) : (
              <div className="px-3 py-3 text-xs text-text-muted">
                No command found
              </div>
            )}
          </div>
        </div>
      )}
      {showMentionPicker && (
        <div
          className="mb-2 overflow-hidden rounded-lg border border-border bg-bg-surface shadow-lg"
          role="dialog"
          aria-label="Mention someone"
        >
          <div className="flex items-center gap-2 border-b border-border px-3 py-2">
            <Search size={14} className="shrink-0 text-text-muted" />
            <input
              ref={mentionInputRef}
              value={mentionQuery}
              onChange={(event) => setMentionQuery(event.target.value)}
              placeholder="Search people or agents"
              className="min-w-0 flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-muted"
            />
            <button
              type="button"
              onClick={closeMentionPicker}
              className="flex h-7 w-7 items-center justify-center rounded-md text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
              title="Close mention picker"
              aria-label="Close mention picker"
            >
              <X size={14} />
            </button>
          </div>
          <div className="max-h-56 overflow-y-auto py-1">
            {mentionQuery.trim().length < 2 ? (
              <div className="px-3 py-3 text-xs text-text-muted">
                Type at least two characters to find a teammate or agent.
              </div>
            ) : mentionLoading ? (
              <div className="flex items-center gap-2 px-3 py-3 text-xs text-text-muted">
                <Loader2 size={13} className="animate-spin" />
                <span>Searching...</span>
              </div>
            ) : mentionError ? (
              <div
                className="px-3 py-3 text-xs text-text-muted"
                data-testid="mention-search-error"
              >
                <div className="font-medium text-text-primary">
                  Mention search could not load
                </div>
                <div className="mt-1">{mentionError}</div>
                <button
                  type="button"
                  onClick={() => setMentionRetryKey((current) => current + 1)}
                  className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-border px-2 py-1 text-2xs font-medium text-text-secondary transition-colors hover:border-accent hover:text-accent"
                  aria-label="Retry mention search"
                >
                  Retry
                </button>
              </div>
            ) : mentionResults.length > 0 ? (
              mentionResults.map((user) => (
                <button
                  key={user.id}
                  type="button"
                  onClick={() => insertMention(user)}
                  className="flex w-full items-center gap-3 px-3 py-2 text-left text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
                  aria-label={`Mention ${user.displayName}`}
                >
                  <span
                    className={`avatar h-8 w-8 text-xs ${user.isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"}`}
                  >
                    {user.avatarUrl ? (
                      <img
                        src={user.avatarUrl}
                        alt=""
                        className="h-full w-full rounded-full object-cover"
                      />
                    ) : user.isAgent ? (
                      <Bot size={14} />
                    ) : (
                      user.displayName[0]?.toUpperCase()
                    )}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex min-w-0 items-center gap-2">
                      <span className="truncate text-sm font-semibold text-text-primary">
                        {user.displayName}
                      </span>
                      {user.isAgent && (
                        <span className="badge-teal text-2xs">agent</span>
                      )}
                    </span>
                    <span className="block truncate text-xs text-text-muted">
                      @{mentionLabel(user)}
                    </span>
                  </span>
                </button>
              ))
            ) : (
              <div className="px-3 py-3 text-xs text-text-muted">
                No teammates found
              </div>
            )}
          </div>
        </div>
      )}
      <div className="overflow-hidden rounded-lg border border-border bg-bg-surface shadow-sm transition-colors focus-within:border-accent">
        <textarea
          data-testid="message-composer"
          ref={textareaRef}
          value={content}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={composerPlaceholder}
          rows={1}
          disabled={disabled}
          className="w-full resize-none bg-transparent px-3 pt-3 pb-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none"
        />
        {showEmojiPicker && (
          <div className="border-t border-border px-2 py-2">
            <div
              className="flex flex-wrap gap-1"
              role="group"
              aria-label="Emoji picker"
            >
              {EMOJI_OPTIONS.map((emoji) => (
                <button
                  key={emoji.label}
                  type="button"
                  onClick={() => {
                    insertTextAtCursor(emoji.value);
                    setShowEmojiPicker(false);
                  }}
                  className="flex h-8 w-8 items-center justify-center rounded-md text-base transition-colors hover:bg-bg-hover focus:outline-none focus:ring-2 focus:ring-accent/40"
                  aria-label={`Insert ${emoji.label} emoji`}
                >
                  {emoji.value}
                </button>
              ))}
            </div>
          </div>
        )}
        <div className="flex items-center justify-between border-t border-border px-2 py-1.5">
          <div className="flex items-center gap-0.5">
            {onFileUpload && (
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={disabled || uploadingFile}
                className="flex h-8 w-8 items-center justify-center rounded-md text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
                title={uploadingFile ? "Uploading file" : "Attach file"}
                aria-label={uploadingFile ? "Uploading file" : "Attach file"}
              >
                {uploadingFile ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Paperclip size={16} />
                )}
              </button>
            )}
            <button
              type="button"
              onClick={openMentionPicker}
              disabled={disabled}
              className="flex h-8 w-8 items-center justify-center rounded-md text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
              title="Mention someone"
              aria-label="Mention someone"
              aria-expanded={showMentionPicker}
            >
              <AtSign size={16} />
            </button>
            <button
              type="button"
              onClick={() => {
                setShowMentionPicker(false);
                setShowEmojiPicker((visible) => !visible);
              }}
              disabled={disabled}
              className="flex h-8 w-8 items-center justify-center rounded-md text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
              title="Add emoji"
              aria-label="Add emoji"
              aria-expanded={showEmojiPicker}
            >
              <Smile size={16} />
            </button>
            {onVoiceSend && (
              <button
                type="button"
                onClick={() => setRecording(true)}
                disabled={disabled}
                className="flex h-8 w-8 items-center justify-center rounded-md text-text-muted transition-colors hover:bg-bg-hover hover:text-accent"
                title="Record voice message"
                aria-label="Record voice message"
              >
                <Mic size={16} />
              </button>
            )}
          </div>
          <button
            type="button"
            data-testid="message-send-button"
            onClick={() => void handleSubmit()}
            disabled={!content.trim() || disabled}
            title="Send message"
            aria-label="Send message"
            className={`flex h-8 w-8 items-center justify-center rounded-md transition-all ${
              content.trim()
                ? "bg-accent text-white hover:bg-accent-hover"
                : "bg-bg-elevated text-text-muted"
            }`}
          >
            <Send size={16} />
          </button>
        </div>
      </div>
      {composerError && (
        <div
          className="mt-2 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
          data-testid="message-composer-error"
        >
          {composerError}
        </div>
      )}
      {uploadError && failedUploadFile && (
        <div
          className="mt-2 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
          data-testid="file-upload-error"
        >
          <div className="font-medium">File upload could not complete</div>
          <div className="mt-1">
            {failedUploadFile.name}: {uploadError}
          </div>
          <button
            type="button"
            onClick={() => void uploadSelectedFile(failedUploadFile)}
            disabled={uploadingFile}
            className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-red-300 bg-white px-2 py-1 text-2xs font-medium text-red-700 hover:bg-red-50 disabled:cursor-wait disabled:opacity-70 dark:border-red-900/70 dark:bg-red-950/40 dark:text-red-100 dark:hover:bg-red-900/30"
            aria-label={`Retry file upload ${failedUploadFile.name}`}
          >
            {uploadingFile && <Loader2 size={12} className="animate-spin" />}
            Retry upload
          </button>
        </div>
      )}
    </div>
  );
}
