"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AtSign,
  Bell,
  Bot,
  Bookmark,
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
import BrowserNotificationPrompt from "./BrowserNotificationPrompt";
import { usePresenceStore } from "@/stores/presenceStore";
import { useUnreadStore } from "@/stores/unreadStore";
import { apiUrl } from "@/lib/apiUrl";
import { useEmbeddedNavigation } from "@/lib/useEmbeddedNavigation";

interface SidebarProps {
  channels: ChannelInfo[];
  dms: (ChannelInfo & {
    otherUser?: {
      id: string;
      displayName: string;
      avatarUrl: string | null;
      isAgent: boolean;
      status: string;
    } | null;
  })[];
  userId: string;
  activityUnreadCount?: number;
  mobileOpen?: boolean;
  onMobileClose?: () => void;
}

type SearchUser = {
  id: string;
  username: string;
  displayName: string;
  isAgent: boolean;
};
type SidebarSearchErrorKind = "search" | "dm";

const formatUnread = (count: number) => (count > 99 ? "99+" : String(count));

const isEditableShortcutTarget = (target: EventTarget | null) => {
  if (!(target instanceof HTMLElement)) return false;
  const tagName = target.tagName.toLowerCase();
  return (
    tagName === "input" ||
    tagName === "textarea" ||
    tagName === "select" ||
    target.isContentEditable
  );
};

export default function Sidebar({
  channels,
  dms,
  userId,
  activityUnreadCount = 0,
  mobileOpen = false,
  onMobileClose,
}: SidebarProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [showNewDM, setShowNewDM] = useState(false);
  const [showQuickSwitcher, setShowQuickSwitcher] = useState(false);
  const [quickSwitcherMode, setQuickSwitcherMode] =
    useState<QuickSwitcherMode>("jump");
  const [quickSwitcherInitialQuery, setQuickSwitcherInitialQuery] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchUser[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchErrorKind, setSearchErrorKind] =
    useState<SidebarSearchErrorKind | null>(null);
  const [startingDmId, setStartingDmId] = useState<string | null>(null);
  const [isInteractive, setIsInteractive] = useState(false);
  const presenceStatuses = usePresenceStore((s) => s.statuses);
  const unreadCounts = useUnreadStore((s) => s.counts);
  const { withEmbed } = useEmbeddedNavigation();

  const channelUnreadTotal = channels.reduce(
    (total, ch) => total + (unreadCounts.get(ch.id) || 0),
    0,
  );
  const dmUnreadTotal = dms.reduce(
    (total, dm) => total + (unreadCounts.get(dm.id) || 0),
    0,
  );
  const agentDmCount = dms.filter((dm) => dm.otherUser?.isAgent).length;
  const canCreateChannels = channels.some((channel) => channel.canCreate);
  const visibleActivityUnreadCount =
    pathname === "/activity" ? 0 : activityUnreadCount;

  const destinations = useMemo(
    () => [
      ...channels.map((channel) => ({
        id: channel.id,
        path: `/channels/${channel.id}`,
        href: withEmbed(`/channels/${channel.id}`),
      })),
      ...dms.map((dm) => ({
        id: dm.id,
        path: `/dm/${dm.id}`,
        href: withEmbed(`/dm/${dm.id}`),
      })),
    ],
    [channels, dms, withEmbed],
  );

  const openQuickSwitcher = useCallback((mode: QuickSwitcherMode = "jump", initialQuery = "") => {
    setQuickSwitcherMode(mode);
    setQuickSwitcherInitialQuery(initialQuery);
    setShowQuickSwitcher(true);
  }, []);

  useEffect(() => {
    const interactiveTimer = window.setTimeout(
      () => setIsInteractive(true),
      120,
    );
    return () => window.clearTimeout(interactiveTimer);
  }, []);

  const goToRelativeDestination = useCallback(
    (direction: 1 | -1) => {
      if (destinations.length === 0) {
        router.push(withEmbed("/dm"));
        return;
      }

      const activeIndex = destinations.findIndex(
        (destination) => destination.path === pathname,
      );
      const fallbackIndex = direction > 0 ? -1 : 0;
      const nextIndex =
        (activeIndex >= 0 ? activeIndex : fallbackIndex) + direction;
      const destination =
        destinations[(nextIndex + destinations.length) % destinations.length];
      router.push(destination.href);
    },
    [destinations, pathname, router, withEmbed],
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || isEditableShortcutTarget(event.target))
        return;

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

      if (
        event.altKey &&
        !modifierKey &&
        !event.shiftKey &&
        event.key === "ArrowDown"
      ) {
        event.preventDefault();
        goToRelativeDestination(1);
        return;
      }

      if (
        event.altKey &&
        !modifierKey &&
        !event.shiftKey &&
        event.key === "ArrowUp"
      ) {
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
      if (
        event.data?.source !== "librechat" ||
        event.data?.type !== "street-voices-shortcut"
      )
        return;

      switch (event.data.action) {
        case "jump":
          openQuickSwitcher("jump");
          break;
        case "compose":
          openQuickSwitcher(
            "compose",
            typeof event.data.query === "string" ? event.data.query : "",
          );
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
    setSearchError(null);
    setSearchErrorKind(null);
    if (q.length < 2) {
      setSearchResults([]);
      return;
    }
    setSearching(true);
    try {
      const res = await fetch(
        apiUrl(`/api/users/search?q=${encodeURIComponent(q)}`),
      );
      if (res.ok) {
        const data = await res.json();
        setSearchResults(data.filter((u: { id: string }) => u.id !== userId));
      } else {
        const data = (await res.json().catch(() => null)) as {
          error?: string;
        } | null;
        setSearchResults([]);
        setSearchError(
          data?.error || "People search is unavailable right now.",
        );
        setSearchErrorKind("search");
      }
    } catch {
      setSearchResults([]);
      setSearchError("People search is unavailable right now.");
      setSearchErrorKind("search");
    } finally {
      setSearching(false);
    }
  };

  const startDM = async (otherUserId: string) => {
    if (startingDmId) return;
    setStartingDmId(otherUserId);
    setSearchError(null);
    setSearchErrorKind(null);

    try {
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
        router.push(withEmbed(`/dm/${channelId}`));
        router.refresh();
        return;
      }

      const data = (await res.json().catch(() => null)) as {
        error?: string;
      } | null;
      setSearchError(data?.error || "Could not start that direct message.");
      setSearchErrorKind("dm");
    } catch {
      setSearchError("Could not start that direct message.");
      setSearchErrorKind("dm");
    } finally {
      setStartingDmId(null);
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
      <div
        className="px-3 pt-3 pb-2 space-y-2"
        style={{ borderBottom: "1px solid var(--sv-sidebar-border)" }}
      >
        <div className="flex items-center gap-2">
          <a
            href="/home"
            target="_top"
            className="flex min-w-0 flex-1 items-center rounded-md px-1.5 py-2 transition-opacity hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            title="Street Voices home"
            aria-label="Street Voices home"
          >
            <img
              src="/assets/streetvoices-text-dark.svg"
              alt="Street Voices"
              className="block w-[154px] max-w-full dark:hidden"
            />
            <img
              src="/assets/streetvoices-text.svg"
              alt="Street Voices"
              className="hidden w-[154px] max-w-full dark:block"
            />
          </a>
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
            className="sidebar-icon-button h-9 w-9 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={!isInteractive}
            title="Start a new message"
            aria-label="Start a new message"
          >
            <PencilLine size={16} />
          </button>
        </div>

        <button
          type="button"
          onClick={() => openQuickSwitcher("jump")}
          className="sidebar-surface-button-muted w-full flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-60"
          disabled={!isInteractive}
          title="Open quick switcher"
          aria-label="Open quick switcher"
        >
          <Search size={14} />
          <span className="truncate">Jump to channel, DM, or agent</span>
          <kbd
            className="ml-auto rounded border px-1.5 py-0.5 text-2xs"
            style={{
              borderColor: "var(--sv-sidebar-border)",
              color: "var(--sv-sidebar-muted)",
            }}
          >
            /
          </kbd>
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto px-2 py-3 space-y-4">
        <BrowserNotificationPrompt />

        <div className="space-y-0.5">
          <Link
            href={withEmbed("/dm")}
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
            href={withEmbed("/activity")}
            onClick={onMobileClose}
            className={`sidebar-item ${pathname === "/activity" ? "active" : ""}`}
          >
            <Bell size={16} className="shrink-0" />
            <span>Activity</span>
            {visibleActivityUnreadCount > 0 && (
              <span className="ml-auto rounded-full bg-accent px-1.5 py-0.5 text-2xs font-semibold text-[#1a1c24]">
                {formatUnread(visibleActivityUnreadCount)}
              </span>
            )}
          </Link>
          <Link
            href={withEmbed("/mentions")}
            onClick={onMobileClose}
            className={`sidebar-item ${pathname === "/mentions" ? "active" : ""}`}
          >
            <AtSign size={16} className="shrink-0" />
            <span>Mentions</span>
          </Link>
          <Link
            href={withEmbed("/saved")}
            onClick={onMobileClose}
            className={`sidebar-item ${pathname === "/saved" ? "active" : ""}`}
          >
            <Bookmark size={16} className="shrink-0" />
            <span>Later</span>
          </Link>
          <Link
            href={withEmbed("/channels")}
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
            {canCreateChannels && (
              <Link
                href={withEmbed("/channels?create=true")}
                onClick={onMobileClose}
                className="sidebar-icon-button h-6 w-6"
                title="Add channel"
                aria-label="Add channel"
              >
                <Plus size={14} />
              </Link>
            )}
          </div>
          <div className="space-y-0.5">
            {channels.length === 0 && (
              <div
                className="px-3 py-1.5 text-xs"
                style={{ color: "var(--sv-sidebar-muted)" }}
              >
                No channels yet
              </div>
            )}
            {channels.map((ch) => {
              const unread = unreadCounts.get(ch.id) || 0;
              const active = pathname === `/channels/${ch.id}`;
              return (
                <Link
                  key={ch.id}
                  href={withEmbed(`/channels/${ch.id}`)}
                  onClick={onMobileClose}
                  className={`sidebar-item ${active ? "active" : ""}`}
                >
                  {ch.type === "PRIVATE" ? (
                    <Lock size={14} className="shrink-0" />
                  ) : (
                    <Hash size={14} className="shrink-0" />
                  )}
                  <span
                    className={`truncate ${unread > 0 ? "font-semibold" : ""}`}
                  >
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
              aria-label={
                showNewDM
                  ? "Close new message search"
                  : "Start a new direct message"
              }
              aria-expanded={showNewDM}
            >
              {showNewDM ? <X size={14} /> : <Plus size={14} />}
            </button>
          </div>

          {showNewDM && (
            <div className="mb-2 px-2" data-testid="sidebar-new-dm-search">
              <input
                type="text"
                placeholder="Find people or agents"
                aria-label="Find people or agents"
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
                <div
                  className="px-2 py-1 text-2xs"
                  style={{ color: "var(--sv-sidebar-muted)" }}
                >
                  Searching...
                </div>
              )}
              {searchError && (
                <div
                  className="px-2 py-1 text-2xs text-red-400"
                  role="status"
                  data-testid="sidebar-new-dm-error"
                >
                  <span>{searchError}</span>
                  {searchErrorKind === "search" && searchQuery.length >= 2 && (
                    <button
                      type="button"
                      className="mt-1 block font-semibold text-accent hover:underline disabled:opacity-60"
                      disabled={searching}
                      aria-label="Retry people search"
                      onClick={() => {
                        void handleUserSearch(searchQuery);
                      }}
                    >
                      Retry
                    </button>
                  )}
                </div>
              )}
              {searchResults.length > 0 && (
                <div className="mt-1 space-y-0.5">
                  {searchResults.map((user) => (
                    <button
                      key={user.id}
                      type="button"
                      onClick={() => startDM(user.id)}
                      disabled={Boolean(startingDmId)}
                      data-testid="sidebar-new-dm-result"
                      className="w-full sidebar-item text-left"
                    >
                      <Avatar
                        label={user.displayName}
                        isAgent={user.isAgent}
                        size="sm"
                      />
                      <span className="truncate">{user.displayName}</span>
                      {startingDmId === user.id ? (
                        <span className="ml-auto text-2xs">Opening...</span>
                      ) : (
                        user.isAgent && (
                          <span className="badge-teal ml-auto text-2xs">
                            agent
                          </span>
                        )
                      )}
                    </button>
                  ))}
                </div>
              )}
              {searchQuery.length >= 2 &&
                !searching &&
                !searchError &&
                searchResults.length === 0 && (
                  <div
                    className="px-2 py-1 text-2xs"
                    style={{ color: "var(--sv-sidebar-muted)" }}
                  >
                    No people or agents found
                  </div>
                )}
            </div>
          )}

          <div className="space-y-0.5">
            {dms.length === 0 && (
              <div
                className="px-3 py-1.5 text-xs"
                style={{ color: "var(--sv-sidebar-muted)" }}
              >
                Start a direct message
              </div>
            )}
            {dms.map((dm) => {
              const otherUserId = dm.otherUser?.id;
              const presence = dm.otherUser?.isAgent
                ? "online"
                : otherUserId
                  ? presenceStatuses.get(otherUserId)
                  : undefined;
              const dmUnread = unreadCounts.get(dm.id) || 0;
              const active = pathname === `/dm/${dm.id}`;
              return (
                <Link
                  key={dm.id}
                  href={withEmbed(`/dm/${dm.id}`)}
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
                  <span
                    className={`truncate ${dmUnread > 0 ? "font-semibold" : ""}`}
                  >
                    {dm.name}
                  </span>
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
          <Link
            href={withEmbed("/dm?filter=agents")}
            onClick={onMobileClose}
            className="sidebar-item"
            aria-label="Browse AI agents"
          >
            <Sparkles size={16} className="shrink-0 text-teal" />
            <span>Browse agents</span>
            {agentDmCount > 0 && (
              <span className="badge-teal ml-auto">{agentDmCount}</span>
            )}
          </Link>
        </div>
      </nav>

      <QuickSwitcher
        open={showQuickSwitcher}
        onClose={() => {
          setShowQuickSwitcher(false);
          setQuickSwitcherInitialQuery("");
        }}
        mode={quickSwitcherMode}
        channels={channels}
        dms={dms}
        userId={userId}
        initialQuery={quickSwitcherInitialQuery}
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
      <div
        className={`${boxSize} avatar ${isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"}`}
      >
        {src ? (
          <img
            src={src}
            alt=""
            className="h-full w-full rounded-full object-cover"
          />
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
