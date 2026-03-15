"use client";

import { useRef, useEffect } from "react";

const EMOJI_CATEGORIES = [
  {
    name: "Smileys",
    emojis: ["😀", "😂", "🤣", "😍", "🥰", "😎", "🤩", "😇", "🥳", "😏", "🤔", "😬", "🫠", "🙃"],
  },
  {
    name: "Gestures",
    emojis: ["👍", "👎", "👏", "🙌", "🤝", "✌️", "🤞", "💪", "🫡", "🫶", "👀", "🧠", "❤️", "🔥"],
  },
  {
    name: "Reactions",
    emojis: ["✅", "❌", "⭐", "💯", "🎯", "🚀", "💡", "⚡", "🎉", "🏆", "💎", "🌟", "📌", "🔔"],
  },
  {
    name: "Objects",
    emojis: ["📝", "📎", "🔗", "📊", "📈", "🗂️", "💻", "🛠️", "⚙️", "🔧", "📦", "🗑️", "📅", "⏰"],
  },
  {
    name: "Faces",
    emojis: ["😢", "😤", "🤯", "😱", "🥺", "😴", "🤮", "🤡", "👻", "💀", "🤖", "👽", "😈", "🫣"],
  },
  {
    name: "Nature",
    emojis: ["🌈", "☀️", "🌙", "⛈️", "🌊", "🌸", "🍀", "🐶", "🐱", "🦊", "🐻", "🦅", "🐙", "🦋"],
  },
];

interface EmojiPickerProps {
  onSelect: (emoji: string) => void;
  onClose: () => void;
}

export default function EmojiPicker({ onSelect, onClose }: EmojiPickerProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  return (
    <div
      ref={ref}
      className="w-72 bg-bg-surface border border-border rounded-xl shadow-xl overflow-hidden"
    >
      <div className="max-h-64 overflow-y-auto p-2 space-y-2">
        {EMOJI_CATEGORIES.map((cat) => (
          <div key={cat.name}>
            <div className="text-2xs font-medium text-text-muted uppercase tracking-wider px-1 mb-1">
              {cat.name}
            </div>
            <div className="grid grid-cols-7 gap-0.5">
              {cat.emojis.map((emoji) => (
                <button
                  key={emoji}
                  onClick={() => onSelect(emoji)}
                  className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-bg-hover text-base transition-colors"
                >
                  {emoji}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
