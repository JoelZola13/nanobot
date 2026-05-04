"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import {
  AtSign,
  ClipboardList,
  HelpCircle,
  ListChecks,
  Megaphone,
  Mic,
  Paperclip,
  Send,
  Smile,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import VoiceRecorder from "./VoiceRecorder";
import {
  clearMessageDraft,
  getBrowserMessageDraftStorage,
  readMessageDraft,
  writeMessageDraft,
} from "@/lib/messageDrafts";

type SlashCommand = {
  name: string;
  title: string;
  description: string;
  insertText: string;
  Icon: LucideIcon;
};

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
    insertText: "Available commands:\n/todo - action list\n/decision - decision note\n/handoff - handoff\n/help - command list",
    Icon: HelpCircle,
  },
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
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);
  const [slashMenuDismissed, setSlashMenuDismissed] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
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
    setContent(readMessageDraft(resolvedDraftId, getBrowserMessageDraftStorage()));
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

    writeMessageDraft(resolvedDraftId, content, getBrowserMessageDraftStorage());
  }, [content, resolvedDraftId]);

  const applySlashCommand = useCallback((command: SlashCommand) => {
    setContent(command.insertText);
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
  }, [resizeTextarea]);

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
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    } catch {
      writeMessageDraft(resolvedDraftId, content, getBrowserMessageDraftStorage());
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
        setSelectedCommandIndex((current) => (current + 1) % Math.max(filteredSlashCommands.length, 1));
        return;
      }

      if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedCommandIndex((current) =>
          (current - 1 + Math.max(filteredSlashCommands.length, 1)) % Math.max(filteredSlashCommands.length, 1),
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
    const newContent = content.substring(0, start) + wrapper + selected + wrapper + content.substring(end);
    setContent(newContent);
    setTimeout(() => {
      ta.selectionStart = start + wrapper.length;
      ta.selectionEnd = end + wrapper.length;
      ta.focus();
    }, 0);
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setContent(e.target.value);
    setSlashMenuDismissed(false);
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
    setRecording(false);
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && onFileUpload) {
      await onFileUpload(file);
    }
    // Reset input
    if (fileInputRef.current) fileInputRef.current.value = "";
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
                      selected ? "bg-bg-hover text-text-primary" : "text-text-secondary hover:bg-bg-hover hover:text-text-primary"
                    }`}
                  >
                    <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border bg-bg-elevated text-accent">
                      <Icon size={15} />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="flex min-w-0 items-baseline gap-2">
                        <span className="shrink-0 text-sm font-semibold text-text-primary">/{command.name}</span>
                        <span className="truncate text-xs text-text-muted">{command.title}</span>
                      </span>
                      <span className="block truncate text-xs text-text-muted">{command.description}</span>
                    </span>
                  </button>
                );
              })
            ) : (
              <div className="px-3 py-3 text-xs text-text-muted">No command found</div>
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
          className="w-full resize-none bg-transparent px-3 pt-3 pb-2 text-sm text-text-primary placeholder-text-muted focus:outline-none"
        />
        <div className="flex items-center justify-between border-t border-border px-2 py-1.5">
          <div className="flex items-center gap-0.5">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex h-8 w-8 items-center justify-center rounded-md text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
              title="Attach file"
            >
              <Paperclip size={16} />
            </button>
            <button
              type="button"
              className="flex h-8 w-8 items-center justify-center rounded-md text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
              title="Mention someone"
            >
              <AtSign size={16} />
            </button>
            <button
              type="button"
              className="flex h-8 w-8 items-center justify-center rounded-md text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
              title="Add emoji"
            >
              <Smile size={16} />
            </button>
            <button
              type="button"
              onClick={() => setRecording(true)}
              className="flex h-8 w-8 items-center justify-center rounded-md text-text-muted transition-colors hover:bg-bg-hover hover:text-accent"
              title="Record voice message"
            >
              <Mic size={16} />
            </button>
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
    </div>
  );
}
