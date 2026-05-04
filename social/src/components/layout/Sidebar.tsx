"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AtSign,
  Bot,
  Bookmark,
  ChevronDown,
  Hash,
  Lock,
  MessageSquare,
  PencilLine,
  Plus,
  Search,
  ShieldCheck,
  Sparkles,
  Users,
  X,
} from "lucide-react";
import type { ChannelInfo } from "@/types";
import QuickSwitcher, { type QuickSwitcherMode } from "./QuickSwitcher";
import { usePresenceStore } from "@/stores/presenceStore";
import { useUnreadStore } from "@/stores/unreadStore";
import { apiUrl } from "@/lib/apiUrl";

interface SidebarProps {
  channels: ChannelInfo[];
  dms: (ChannelInfo & { otherUser?: { id: string; displayName: string; avatarUrl: string | null; isAgent: boolean; status: string } | null })[];
  userId: string;
  mobileOpen?: boolean;
  onMobileClose?: () => void;
}

type SearchUser = {
  id: string;
  username: string;
  displayName: string;
  isAgent: boolean;
};

const formatUnread = (count: number) => (count > 99 ? "99+" : String(count));

const isEditableShortcutTarget = (target: EventTarget | null) => {
  if (!(target instanceof HTMLElement)) return false;
  const tagName = target.tagName.toLowerCase();
  return tagName === "input" || tagName === "textarea" || tagName === "select" || target.isContentEditable;
};

export default function Sidebar({ channels, dms, userId, mobileOpen = false, onMobileClose }: SidebarProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [showNewDM, setShowNewDM] = useState(false);
  const [showQuickSwitcher, setShowQuickSwitcher] = useState(false);
  const [quickSwitcherMode, setQuickSwitcherMode] = useState<QuickSwitcherMode>("jump");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchUser[]>([]);
  const [searching, setSearching] = useState(false);
  const presenceStatuses = usePresenceStore((s) => s.statuses);
  const unreadCounts = useUnreadStore((s) => s.counts);

  const channelUnreadTotal = channels.reduce((total, ch) => total + (unreadCounts.get(ch.id) || 0), 0);
  const dmUnreadTotal = dms.reduce((total, dm) => total + (unreadCounts.get(dm.id) || 0), 0);
  const agentDmCount = dms.filter((dm) => dm.otherUser?.isAgent).length;

  const destinations = useMemo(
    () => [
      ...channels.map((channel) => ({
        id: channel.id,
        href: `/channels/${channel.id}`,
      })),
      ...dms.map((dm) => ({
        id: dm.id,
        href: `/dm/${dm.id}`,
      })),
    ],
    [channels, dms],
  );

  const openQuickSwitcher = useCallback((mode: QuickSwitcherMode = "jump") => {
    setQuickSwitcherMode(mode);
    setShowQuickSwitcher(true);
  }, []);

  const goToRelativeDestination = useCallback(
    (direction: 1 | -1) => {
      if (destinations.length === 0) {
        router.push("/dm");
        return;
      }

      const activeIndex = destinations.findIndex((destination) => destination.href === pathname);
      const fallbackIndex = direction > 0 ? -1 : 0;
      const nextIndex = (activeIndex >= 0 ? activeIndex : fallbackIndex) + direction;
      const destination = destinations[(nextIndex + destinations.length) % destinations.length];
      router.push(destination.href);
    },
    [destinations, pathname, router],
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || isEditableShortcutTarget(event.target)) return;

      const key = event.key.toLowerCase();
      const modifierKey = event.metaKey || event.ctrlKey;

      if (modifierKey && !event.altKey && !event.shiftKey && key === "k") {
        event.preventDefault();
        openQuickSwitcher("jump");
        return;
      }

      if (modifierKey && !event.altKey && !event.shiftKey && key === "n") {
        event.preventDefault();
        openQuickSwitcher("compose");
        return;
      }

      if (!modifierKey && !event.altKey && !event.shiftKey && key === "/") {
        event.preventDefault();
        openQuickSwitcher("jump");
        return;
      }

      if (event.altKey && !modifierKey && !event.shiftKey && event.key === "ArrowDown") {
        event.preventDefault();
        goToRelativeDestination(1);
        return;
      }

      if (event.altKey && !modifierKey && !event.shiftKey && event.key === "ArrowUp") {
        event.preventDefault();
        goToRelativeDestination(-1);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [goToRelativeDestination, openQuickSwitcher]);

  useEffect(() => {
    const handleParentShortcut = (event: MessageEvent) => {
      if (event.origin !== window.location.origin) return;
      if (event.data?.source !== "librechat" || event.data?.type !== "street-voices-shortcut") return;

      switch (event.data.action) {
        case "jump":
          openQuickSwitcher("jump");
          break;
        case "compose":
          openQuickSwitcher("compose");
          break;
        case "next":
          goToRelativeDestination(1);
          break;
        case "previous":
          goToRelativeDestination(-1);
          break;
      }
    };

    window.addEventListener("message", handleParentShortcut);
    return () => window.removeEventListener("message", handleParentShortcut);
  }, [goToRelativeDestination, openQuickSwitcher]);

  const handleUserSearch = async (q: string) => {
    setSearchQuery(q);
    if (q.length < 2) {
      setSearchResults([]);
      return;
    }
    setSearching(true);
    try {
      const res = await fetch(apiUrl(`/api/users/search?q=${encodeURIComponent(q)}`));
      if (res.ok) {
        const data = await res.json();
        setSearchResults(data.filter((u: { id: string }) => u.id !== userId));
      }
    } finally {
      setSearching(false);
    }
  };

  const startDM = async (otherUserId: string) => {
    const res = await fetch(apiUrl("/api/dm"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ userId: otherUserId }),
    });
    if (res.ok) {
      const { channelId } = await res.json();
      setShowNewDM(false);
      setSearchQuery("");
      setSearchResults([]);
      onMobileClose?.();
      router.push(`/dm/${channelId}`);
      router.refresh();
    }
  };

  return (
    <aside
      aria-label="Messages workspace"
      className={`sv-messages-sidebar flex h-screen w-[280px] shrink-0 flex-col ${mobileOpen ? "is-mobile-open" : ""}`}
      style={{
        background: "var(--sv-sidebar-bg)",
        borderRight: "1px solid var(--sv-sidebar-border)",
        color: "var(--sv-sidebar-text)",
      }}
    >
      <div className="px-3 pt-3 pb-2 space-y-2" style={{ borderBottom: "1px solid var(--sv-sidebar-border)" }}>
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="sidebar-surface-button min-w-0 flex-1 flex items-center gap-2 rounded-md px-2 py-2 text-left transition-colors"
            title="Workspace"
          >
            <div className="w-8 h-8 rounded-md bg-accent text-[#1a1c24] font-heading text-sm font-bold flex items-center justify-center shrink-0">
              SV
            </div>
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold leading-5">Street Voices</div>
              <div className="truncate text-2xs" style={{ color: "var(--sv-sidebar-muted)" }}>
                Messages workspace
              </div>
            </div>
            <ChevronDown size={14} className="shrink-0" style={{ color: "var(--sv-sidebar-muted)" }} />
          </button>
          <button
            type="button"
            onClick={onMobileClose}
            className="sv-sidebar-mobile-close sidebar-icon-button h-9 w-9"
            title="Close messages sidebar"
            aria-label="Close messages sidebar"
          >
            <X size={16} />
          </button>
          <button
            type="button"
            onClick={() => openQuickSwitcher("compose")}
            className="sidebar-icon-button h-9 w-9"
            title="Open quick switcher"
          >
            <PencilLine size={16} />
          </button>
        </div>

        <button
          type="button"
          onClick={() => openQuickSwitcher("jump")}
          className="sidebar-surface-button-muted w-full flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors"
        >
          <Search size={14} />
          <span className="truncate">Jump to channel, DM, or agent</span>
          <kbd
            className="ml-auto rounded border px-1.5 py-0.5 text-2xs"
            style={{ borderColor: "var(--sv-sidebar-border)", color: "var(--sv-sidebar-muted)" }}
          >
            /
          </kbd>
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto px-2 py-3 space-y-4">
        <div className="space-y-0.5">
          <Link
            href="/dm"
            onClick={onMobileClose}
            className={`sidebar-item ${pathname === "/dm" ? "active" : ""}`}
          >
            <MessageSquare size={16} className="shrink-0" />
            <span>Direct messages</span>
            {dmUnreadTotal > 0 && (
              <span className="ml-auto rounded-full bg-accent px-1.5 py-0.5 text-2xs font-semibold text-[#1a1c24]">
                {formatUnread(dmUnreadTotal)}
              </span>
            )}
          </Link>
          <Link
            href="/mentions"
            onClick={onMobileClose}
            className={`sidebar-item ${pathname === "/mentions" ? "active" : ""}`}
          >
            <AtSign size={16} className="shrink-0" />
            <span>Mentions</span>
          </Link>
          <Link
            href="/saved"
            onClick={onMobileClose}
            className={`sidebar-item ${pathname === "/saved" ? "active" : ""}`}
          >
            <Bookmark size={16} className="shrink-0" />
            <span>Later</span>
          </Link>
          <Link
            href="/channels"
            onClick={onMobileClose}
            className={`sidebar-item ${pathname === "/channels" ? "active" : ""}`}
          >
            <Users size={16} className="shrink-0" />
            <span>Channel browser</span>
            {channelUnreadTotal > 0 && (
              <span className="ml-auto rounded-full bg-accent px-1.5 py-0.5 text-2xs font-semibold text-[#1a1c24]">
                {formatUnread(channelUnreadTotal)}
              </span>
            )}
          </Link>
        </div>

        <div>
          <div className="mb-1 flex items-center justify-between px-3">
            <span className="sidebar-section-label">Channels</span>
            <Link href="/channels" onClick={onMobileClose} className="sidebar-icon-button h-6 w-6" title="Add channel">
              <Plus size={14} />
            </Link>
          </div>
          <div className="space-y-0.5">
            {channels.length === 0 && (
              <div className="px-3 py-1.5 text-xs" style={{ color: "var(--sv-sidebar-muted)" }}>
                No channels yet
              </div>
            )}
            {channels.map((ch) => {
              const unread = unreadCounts.get(ch.id) || 0;
              const active = pathname === `/channels/${ch.id}`;
              return (
                <Link
                  key={ch.id}
                  href={`/channels/${ch.id}`}
                  onClick={onMobileClose}
                  className={`sidebar-item ${active ? "active" : ""}`}
                >
                  {ch.type === "PRIVATE" ? (
                    <Lock size={14} className="shrink-0" />
                  ) : (
                    <Hash size={14} className="shrink-0" />
                  )}
                  <span className={`truncate ${unread > 0 ? "font-semibold" : ""}`}>
                    {ch.name || "unnamed"}
                  </span>
                  {ch.isDefault && (
                    <ShieldCheck
                      size={13}
                      className="ml-auto shrink-0"
                      style={{ color: "var(--sv-sidebar-muted)" }}
                      aria-label="Default channel"
                    />
                  )}
                  {unread > 0 && (
                    <span
                      className={`${ch.isDefault ? "" : "ml-auto"} rounded-full bg-accent px-1.5 py-0.5 text-2xs font-semibold text-[#1a1c24]`}
                    >
                      {formatUnread(unread)}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        </div>

        <div>
          <div className="mb-1 flex items-center justify-between px-3">
            <span className="sidebar-section-label">People</span>
            <button
              type="button"
              onClick={() => setShowNewDM(!showNewDM)}
              className="sidebar-icon-button h-6 w-6"
              title={showNewDM ? "Close new message" : "New message"}
            >
              {showNewDM ? <X size={14} /> : <Plus size={14} />}
            </button>
          </div>

          {showNewDM && (
            <div className="mb-2 px-2">
              <input
                type="text"
                placeholder="Find people or agents"
                value={searchQuery}
                onChange={(e) => handleUserSearch(e.target.value)}
                className="w-full rounded-md border px-2.5 py-1.5 text-xs outline-none transition-colors"
                style={{
                  background: "var(--sv-sidebar-elevated)",
                  borderColor: "var(--sv-sidebar-border)",
                  color: "var(--sv-sidebar-text)",
                }}
                autoFocus
              />
              {searching && (
                <div className="px-2 py-1 text-2xs" style={{ color: "var(--sv-sidebar-muted)" }}>
                  Searching...
                </div>
              )}
              {searchResults.length > 0 && (
                <div className="mt-1 space-y-0.5">
                  {searchResults.map((user) => (
                    <button
                      key={user.id}
                      type="button"
                      onClick={() => startDM(user.id)}
                      className="w-full sidebar-item text-left"
                    >
                      <Avatar label={user.displayName} isAgent={user.isAgent} size="sm" />
                      <span className="truncate">{user.displayName}</span>
                      {user.isAgent && <span className="badge-teal ml-auto text-2xs">agent</span>}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="space-y-0.5">
            {dms.length === 0 && (
              <div className="px-3 py-1.5 text-xs" style={{ color: "var(--sv-sidebar-muted)" }}>
                Start a direct message
              </div>
            )}
            {dms.map((dm) => {
              const otherUserId = dm.otherUser?.id;
              const presence = dm.otherUser?.isAgent ? "online" : otherUserId ? presenceStatuses.get(otherUserId) : undefined;
              const dmUnread = unreadCounts.get(dm.id) || 0;
              const active = pathname === `/dm/${dm.id}`;
              return (
                <Link
                  key={dm.id}
                  href={`/dm/${dm.id}`}
                  onClick={onMobileClose}
                  className={`sidebar-item ${active ? "active" : ""}`}
                >
                  <Avatar
                    label={dm.name || "?"}
                    src={dm.otherUser?.avatarUrl}
                    isAgent={dm.otherUser?.isAgent}
                    presence={presence}
                    size="sm"
                  />
                  <span className={`truncate ${dmUnread > 0 ? "font-semibold" : ""}`}>{dm.name}</span>
                  {dmUnread > 0 && (
                    <span className="ml-auto rounded-full bg-accent px-1.5 py-0.5 text-2xs font-semibold text-[#1a1c24]">
                      {formatUnread(dmUnread)}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        </div>

        <div>
          <div className="mb-1 flex items-center px-3">
            <span className="sidebar-section-label">AI agents</span>
          </div>
          <Link href="/dm" onClick={onMobileClose} className="sidebar-item">
            <Sparkles size={16} className="shrink-0 text-teal" />
            <span>Browse agents</span>
            {agentDmCount > 0 && <span className="badge-teal ml-auto">{agentDmCount}</span>}
          </Link>
        </div>
      </nav>

      <QuickSwitcher
        open={showQuickSwitcher}
        onClose={() => setShowQuickSwitcher(false)}
        mode={quickSwitcherMode}
        channels={channels}
        dms={dms}
        userId={userId}
      />
    </aside>
  );
}

function Avatar({
  label,
  src,
  isAgent = false,
  presence,
  size = "md",
}: {
  label: string;
  src?: string | null;
  isAgent?: boolean;
  presence?: string;
  size?: "sm" | "md";
}) {
  const boxSize = size === "sm" ? "h-5 w-5 text-2xs" : "h-9 w-9 text-sm";
  const presenceSize = size === "sm" ? "h-2 w-2" : "h-2.5 w-2.5";
  const isOnline = presence === "online" || isAgent;
  const isAway = presence === "away";

  return (
    <div className="relative shrink-0">
      <div className={`${boxSize} avatar ${isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"}`}>
        {src ? (
          <img src={src} alt="" className="h-full w-full rounded-full object-cover" />
        ) : isAgent ? (
          <Bot size={size === "sm" ? 10 : 16} />
        ) : (
          label[0]?.toUpperCase()
        )}
      </div>
      {(isOnline || isAway) && (
        <span
          className={`absolute -bottom-0.5 -right-0.5 rounded-full border ${presenceSize} ${isAway ? "bg-yellow-400" : "bg-teal"}`}
          style={{ borderColor: "var(--sv-sidebar-bg)" }}
        />
      )}
    </div>
  );
}
