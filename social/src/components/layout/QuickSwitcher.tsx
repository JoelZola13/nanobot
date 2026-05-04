"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Bot,
  Hash,
  Lock,
  MessageSquare,
  Search,
  Sparkles,
  UserRound,
  Users,
  X,
} from "lucide-react";
import type { ChannelInfo } from "@/types";
import { apiUrl } from "@/lib/apiUrl";

type DmChannel = ChannelInfo & {
  otherUser?: {
    id: string;
    displayName: string;
    avatarUrl: string | null;
    isAgent: boolean;
    status: string;
  } | null;
};

type SearchUser = {
  id: string;
  username: string;
  displayName: string;
  avatarUrl: string | null;
  isAgent: boolean;
  status: string;
};

type SwitcherItem =
  | {
      id: string;
      type: "channel";
      label: string;
      description: string;
      href: string;
      isPrivate: boolean;
    }
  | {
      id: string;
      type: "dm";
      label: string;
      description: string;
      href: string;
      avatarUrl: string | null;
      isAgent: boolean;
      status: string;
    }
  | {
      id: string;
      type: "person";
      label: string;
      description: string;
      userId: string;
      avatarUrl: string | null;
      isAgent: boolean;
      status: string;
    }
  | {
      id: string;
      type: "browse";
      label: string;
      description: string;
      href: string;
    };

interface QuickSwitcherProps {
  open: boolean;
  onClose: () => void;
  mode: QuickSwitcherMode;
  channels: ChannelInfo[];
  dms: DmChannel[];
  userId: string;
}

export type QuickSwitcherMode = "jump" | "compose";

const normalized = (value: string | null | undefined) => value?.toLowerCase() || "";

export default function QuickSwitcher({
  open,
  onClose,
  mode,
  channels,
  dms,
  userId,
}: QuickSwitcherProps) {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [remoteResults, setRemoteResults] = useState<SearchUser[]>([]);
  const [searching, setSearching] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [startingId, setStartingId] = useState<string | null>(null);

  const trimmedQuery = query.trim();
  const lowerQuery = trimmedQuery.toLowerCase();
  const isComposeMode = mode === "compose";

  useEffect(() => {
    if (!open) return;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    setSelectedIndex(0);

    const focusTimer = window.setTimeout(() => inputRef.current?.focus(), 0);

    return () => {
      window.clearTimeout(focusTimer);
      document.body.style.overflow = previousOverflow;
    };
  }, [open]);

  useEffect(() => {
    if (!open) {
      setQuery("");
      setRemoteResults([]);
      setSearching(false);
      setStartingId(null);
    }
  }, [open]);

  useEffect(() => {
    if (!open || trimmedQuery.length < 2) {
      setRemoteResults([]);
      setSearching(false);
      return;
    }

    const controller = new AbortController();
    setSearching(true);

    const timer = window.setTimeout(async () => {
      try {
        const res = await fetch(apiUrl(`/api/users/search?q=${encodeURIComponent(trimmedQuery)}`), {
          signal: controller.signal,
        });
        if (res.ok) {
          const users = (await res.json()) as SearchUser[];
          setRemoteResults(users.filter((user) => user.id !== userId));
        }
      } catch {
        if (!controller.signal.aborted) setRemoteResults([]);
      } finally {
        if (!controller.signal.aborted) setSearching(false);
      }
    }, 120);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [open, trimmedQuery, userId]);

  const existingDmUserIds = useMemo(
    () => new Set(dms.map((dm) => dm.otherUser?.id).filter(Boolean) as string[]),
    [dms],
  );

  const items = useMemo<SwitcherItem[]>(() => {
    const matches = (...values: Array<string | null | undefined>) => {
      if (!lowerQuery) return true;
      return values.some((value) => normalized(value).includes(lowerQuery));
    };

    const channelItems = channels
      .filter((channel) => matches(channel.name, channel.slug, channel.description))
      .slice(0, lowerQuery ? 8 : 5)
      .map<SwitcherItem>((channel) => ({
        id: `channel-${channel.id}`,
        type: "channel",
        label: channel.name || "unnamed",
        description: channel.description || `${channel.memberCount || 1} members`,
        href: `/channels/${channel.id}`,
        isPrivate: channel.type === "PRIVATE",
      }));

    const dmItems = dms
      .filter((dm) =>
        matches(
          dm.name,
          dm.description,
          dm.otherUser?.displayName,
          dm.otherUser?.isAgent ? "agent" : "teammate",
        ),
      )
      .slice(0, lowerQuery ? 8 : 7)
      .map<SwitcherItem>((dm) => ({
        id: `dm-${dm.id}`,
        type: "dm",
        label: dm.name || dm.otherUser?.displayName || "Unknown",
        description: dm.otherUser?.isAgent ? "AI agent DM" : dm.otherUser?.status === "online" ? "Online teammate" : "Teammate DM",
        href: `/dm/${dm.id}`,
        avatarUrl: dm.otherUser?.avatarUrl || null,
        isAgent: Boolean(dm.otherUser?.isAgent),
        status: dm.otherUser?.status || "offline",
      }));

    const remoteItems = remoteResults
      .filter((user) => !existingDmUserIds.has(user.id))
      .map<SwitcherItem>((user) => ({
        id: `person-${user.id}`,
        type: "person",
        label: user.displayName,
        description: user.isAgent ? "Start an AI agent DM" : `@${user.username}`,
        userId: user.id,
        avatarUrl: user.avatarUrl,
        isAgent: user.isAgent,
        status: user.status,
      }));

    const browseItems: SwitcherItem[] = lowerQuery
      ? []
      : [
          {
            id: "browse-directory",
            type: "browse",
            label: "Browse teammates and agents",
            description: "Open the full direct message directory",
            href: "/dm",
          },
        ];

    return [...channelItems, ...dmItems, ...remoteItems, ...browseItems];
  }, [channels, dms, existingDmUserIds, lowerQuery, remoteResults]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [lowerQuery, items.length]);

  if (!open) return null;

  const close = () => {
    setQuery("");
    setRemoteResults([]);
    onClose();
  };

  const goTo = (href: string) => {
    close();
    router.push(href);
  };

  const startDM = async (targetUserId: string) => {
    if (startingId) return;
    setStartingId(targetUserId);
    try {
      const res = await fetch(apiUrl("/api/dm"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId: targetUserId }),
      });
      if (res.ok) {
        const { channelId } = (await res.json()) as { channelId: string };
        close();
        router.push(`/dm/${channelId}`);
        router.refresh();
      }
    } finally {
      setStartingId(null);
    }
  };

  const activate = (item: SwitcherItem | undefined) => {
    if (!item) return;
    if (item.type === "person") {
      void startDM(item.userId);
      return;
    }
    goTo(item.href);
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Escape") {
      event.preventDefault();
      close();
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      setSelectedIndex((index) => (items.length === 0 ? 0 : (index + 1) % items.length));
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      setSelectedIndex((index) => (items.length === 0 ? 0 : (index - 1 + items.length) % items.length));
      return;
    }

    if (event.key === "Enter") {
      event.preventDefault();
      activate(items[selectedIndex]);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 px-3 pt-[12vh]"
      role="dialog"
      aria-modal="true"
      aria-label={isComposeMode ? "New message quick switcher" : "Quick switcher"}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) close();
      }}
      onKeyDown={handleKeyDown}
    >
      <div
        className="w-full max-w-xl overflow-hidden rounded-lg border shadow-2xl"
        style={{
          background: "var(--sv-bg-elevated)",
          borderColor: "var(--sv-border)",
          color: "var(--sv-text-primary)",
        }}
      >
        <div className="flex items-center gap-3 border-b px-4 py-3" style={{ borderColor: "var(--sv-border)" }}>
          <Search size={18} className="shrink-0 text-text-muted" />
          <input
            ref={inputRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={isComposeMode ? "Start a DM or open a conversation" : "Jump to a channel, DM, or agent"}
            className="min-w-0 flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-muted"
          />
          <button
            type="button"
            onClick={close}
            className="sidebar-icon-button h-8 w-8"
            title="Close quick switcher"
          >
            <X size={16} />
          </button>
        </div>

        <div className="max-h-[52vh] overflow-y-auto p-2">
          {items.length === 0 ? (
            <div className="px-4 py-8 text-center">
              <Search size={28} className="mx-auto mb-2 text-text-muted" />
              <h3 className="font-heading text-sm font-semibold text-text-primary">No matches</h3>
              <p className="text-xs text-text-muted">
                Search by channel, teammate, username, or agent name.
              </p>
            </div>
          ) : (
            <div className="space-y-1">
              {items.map((item, index) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => activate(item)}
                  onMouseEnter={() => setSelectedIndex(index)}
                  className={`flex w-full items-center gap-3 rounded-md px-3 py-2 text-left transition-colors ${
                    selectedIndex === index ? "bg-bg-hover" : "hover:bg-bg-hover"
                  }`}
                >
                  <SwitcherIcon item={item} />
                  <div className="min-w-0 flex-1">
                    <div className="flex min-w-0 items-center gap-2">
                      <span className="truncate text-sm font-semibold text-text-primary">{item.label}</span>
                      {item.type === "person" && item.isAgent && <span className="badge-teal text-2xs">agent</span>}
                    </div>
                    <p className="truncate text-xs text-text-muted">{item.description}</p>
                  </div>
                  {item.type === "person" && startingId === item.userId ? (
                    <span className="text-2xs font-medium text-text-muted">Opening</span>
                  ) : (
                    <span className="text-2xs text-text-muted">{item.type === "person" ? "Start DM" : "Open"}</span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        <div
          className="flex items-center gap-3 border-t px-4 py-2 text-2xs text-text-muted"
          style={{ borderColor: "var(--sv-border)" }}
        >
          <span>Enter opens</span>
          <span>Arrows move</span>
          <span>Esc closes</span>
          {searching && <span className="ml-auto">Searching...</span>}
        </div>
      </div>
    </div>
  );
}

function SwitcherIcon({ item }: { item: SwitcherItem }) {
  if (item.type === "channel") {
    return (
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-bg-surface text-text-secondary">
        {item.isPrivate ? <Lock size={16} /> : <Hash size={16} />}
      </div>
    );
  }

  if (item.type === "browse") {
    return (
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-teal-muted text-teal">
        <Users size={16} />
      </div>
    );
  }

  const isAgent = item.type === "person" ? item.isAgent : item.isAgent;
  const avatarUrl = item.type === "person" ? item.avatarUrl : item.avatarUrl;
  const label = item.label;
  const status = item.type === "person" ? item.status : item.status;
  const isOnline = status === "online" || isAgent;

  return (
    <div className="relative shrink-0">
      <div className={`avatar h-9 w-9 text-sm ${isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"}`}>
        {avatarUrl ? (
          <img src={avatarUrl} alt="" className="h-full w-full rounded-full object-cover" />
        ) : isAgent ? (
          <Bot size={16} />
        ) : (
          label[0]?.toUpperCase()
        )}
      </div>
      <span
        className={`absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border ${
          isOnline ? "bg-teal" : "bg-border"
        }`}
        style={{ borderColor: "var(--sv-bg-elevated)" }}
      />
      {item.type === "dm" && (
        <span className="absolute -left-1 -top-1 rounded-full bg-bg-elevated p-0.5 text-text-muted">
          <MessageSquare size={10} />
        </span>
      )}
      {item.type === "person" && isAgent && (
        <span className="absolute -left-1 -top-1 rounded-full bg-bg-elevated p-0.5 text-teal">
          <Sparkles size={10} />
        </span>
      )}
      {item.type === "person" && !isAgent && (
        <span className="absolute -left-1 -top-1 rounded-full bg-bg-elevated p-0.5 text-text-muted">
          <UserRound size={10} />
        </span>
      )}
    </div>
  );
}
