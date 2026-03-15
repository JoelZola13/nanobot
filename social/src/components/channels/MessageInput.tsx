"use client";

import { useState, useRef, useCallback } from "react";
import { Send, Paperclip, Smile, AtSign, Mic } from "lucide-react";
import VoiceRecorder from "./VoiceRecorder";

interface MessageInputProps {
  channelId: string;
  channelName: string;
  onSend: (content: string) => void;
  onTyping?: () => void;
  disabled?: boolean;
  placeholder?: string;
  onVoiceSend?: (audioBlob: Blob, duration: number) => Promise<void>;
  onFileUpload?: (file: File) => Promise<void>;
}

export default function MessageInput({
  channelId: _channelId,
  channelName,
  onSend,
  onTyping,
  disabled,
  placeholder,
  onVoiceSend,
  onFileUpload,
}: MessageInputProps) {
  const [content, setContent] = useState("");
  const [recording, setRecording] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const typingTimeout = useRef<ReturnType<typeof setTimeout>>();

  const handleSubmit = useCallback(() => {
    const trimmed = content.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setContent("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [content, disabled, onSend]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
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
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";

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

  return (
    <div className="px-4 pb-4 pt-2">
      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        onChange={handleFileChange}
        accept="image/*,video/*,audio/*,.pdf,.doc,.docx,.txt,.zip,.csv,.xlsx"
      />
      <div className="bg-bg-surface border border-border rounded-xl overflow-hidden focus-within:border-accent transition-colors">
        <textarea
          ref={textareaRef}
          value={content}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || `Message #${channelName}`}
          rows={1}
          disabled={disabled}
          className="w-full bg-transparent px-4 pt-3 pb-1 text-sm text-text-primary placeholder-text-muted resize-none focus:outline-none"
        />
        <div className="flex items-center justify-between px-3 pb-2">
          <div className="flex items-center gap-0.5">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="p-1.5 rounded-lg text-text-muted hover:text-text-primary hover:bg-bg-hover transition-colors"
              title="Attach file"
            >
              <Paperclip size={16} />
            </button>
            <button
              type="button"
              className="p-1.5 rounded-lg text-text-muted hover:text-text-primary hover:bg-bg-hover transition-colors"
              title="Mention someone"
            >
              <AtSign size={16} />
            </button>
            <button
              type="button"
              className="p-1.5 rounded-lg text-text-muted hover:text-text-primary hover:bg-bg-hover transition-colors"
              title="Add emoji"
            >
              <Smile size={16} />
            </button>
            <button
              type="button"
              onClick={() => setRecording(true)}
              className="p-1.5 rounded-lg text-text-muted hover:text-accent hover:bg-bg-hover transition-colors"
              title="Record voice message"
            >
              <Mic size={16} />
            </button>
          </div>
          <button
            onClick={handleSubmit}
            disabled={!content.trim() || disabled}
            className={`p-2 rounded-lg transition-all ${
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
