"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import {
  Hash,
  MessageSquare,
  Users,
  Bot,
  Settings,
  Plus,
  Search,
  X,
} from "lucide-react";
import type { ChannelInfo } from "@/types";
import { usePresenceStore } from "@/stores/presenceStore";
import { useUnreadStore } from "@/stores/unreadStore";
import { apiUrl } from "@/lib/apiUrl";

interface SidebarProps {
  channels: ChannelInfo[];
  dms: (ChannelInfo & { otherUser?: { id: string; displayName: string; avatarUrl: string | null; isAgent: boolean; status: string } | null })[];
  username: string;
  userId: string;
}

export default function Sidebar({ channels, dms, username, userId }: SidebarProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [showNewDM, setShowNewDM] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<{ id: string; username: string; displayName: string; isAgent: boolean }[]>([]);
  const [searching, setSearching] = useState(false);
  const presenceStatuses = usePresenceStore((s) => s.statuses);
  const unreadCounts = useUnreadStore((s) => s.counts);

  // Hide sidebar when embedded in an iframe (profile page) or when standalone sidebar is present
  const isEmbed = typeof window !== 'undefined' && (
    window.location.search.includes('embed=true') ||
    document.getElementById('sv-standalone-sidebar')
  );
  if (isEmbed) return null;

  const handleUserSearch = async (q: string) => {
    setSearchQuery(q);
    if (q.length < 2) { setSearchResults([]); return; }
    setSearching(true);
    try {
      const res = await fetch(apiUrl(`/api/users/search?q=${encodeURIComponent(q)}`));
      if (res.ok) {
        const data = await res.json();
        // Filter out self
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
      router.push(`/dm/${channelId}`);
      router.refresh();
    }
  };

  return (
    <aside className="w-64 h-screen glass-soft border-r border-border flex flex-col shrink-0">
      {/* Brand */}
      <div className="h-14 px-4 flex items-center border-b border-border">
        <h1 className="font-heading font-bold text-lg tracking-tight">
          <span className="text-accent">SV</span>
          <span className="text-text-primary ml-1">Social</span>
        </h1>
      </div>

      {/* Search */}
      <div className="px-3 py-2">
        <button className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-bg-elevated text-text-muted text-sm hover:bg-bg-hover transition-colors">
          <Search size={14} />
          <span>Search...</span>
          <kbd className="ml-auto text-2xs bg-bg px-1.5 py-0.5 rounded border border-border">
            /
          </kbd>
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-2 py-1 space-y-4">
        {/* Main Links */}
        <div className="space-y-0.5">
          <Link
            href="/dm"
            className={`sidebar-item ${pathname.startsWith("/dm") ? "active" : ""}`}
          >
            <MessageSquare size={16} />
            <span>Messages</span>
            {dms.some((d) => (d.unreadCount || 0) > 0) && (
              <span className="ml-auto w-2 h-2 rounded-full bg-accent" />
            )}
          </Link>
        </div>

        {/* Channels */}
        <div>
          <div className="flex items-center justify-between px-3 mb-1">
            <span className="text-2xs font-semibold uppercase tracking-wider text-text-muted">
              Channels
            </span>
            <Link href="/channels" className="text-text-muted hover:text-text-primary transition-colors">
              <Plus size={14} />
            </Link>
          </div>
          <div className="space-y-0.5">
            {channels.map((ch) => {
              const unread = unreadCounts.get(ch.id) || 0;
              return (
                <Link
                  key={ch.id}
                  href={`/channels/${ch.id}`}
                  className={`sidebar-item ${pathname === `/channels/${ch.id}` ? "active" : ""}`}
                >
                  {ch.type === "PRIVATE" ? (
                    <Users size={14} className="shrink-0" />
                  ) : (
                    <Hash size={14} className="shrink-0" />
                  )}
                  <span className={`truncate ${unread > 0 ? "font-semibold text-text-primary" : ""}`}>{ch.name || "unnamed"}</span>
                  {unread > 0 && (
                    <span className="ml-auto text-2xs bg-accent text-white px-1.5 py-0.5 rounded-full font-medium">
                      {unread}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        </div>

        {/* DMs */}
        <div>
          <div className="flex items-center justify-between px-3 mb-1">
            <span className="text-2xs font-semibold uppercase tracking-wider text-text-muted">
              Direct Messages
            </span>
            <button
              onClick={() => setShowNewDM(!showNewDM)}
              className="text-text-muted hover:text-text-primary transition-colors"
            >
              {showNewDM ? <X size={14} /> : <Plus size={14} />}
            </button>
          </div>

          {/* New DM search */}
          {showNewDM && (
            <div className="px-2 mb-2">
              <input
                type="text"
                placeholder="Find a person..."
                value={searchQuery}
                onChange={(e) => handleUserSearch(e.target.value)}
                className="input-field text-xs py-1.5"
                autoFocus
              />
              {searching && (
                <div className="text-2xs text-text-muted px-2 py-1">Searching...</div>
              )}
              {searchResults.length > 0 && (
                <div className="mt-1 space-y-0.5">
                  {searchResults.map((user) => (
                    <button
                      key={user.id}
                      onClick={() => startDM(user.id)}
                      className="w-full sidebar-item text-left"
                    >
                      <div className={`w-5 h-5 avatar text-2xs ${user.isAgent ? "bg-teal-muted text-teal" : ""}`}>
                        {user.isAgent ? <Bot size={10} /> : user.displayName[0]?.toUpperCase()}
                      </div>
                      <span className="truncate">{user.displayName}</span>
                      {user.isAgent && <span className="badge-teal text-2xs ml-auto">agent</span>}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="space-y-0.5">
            {dms.map((dm) => {
              const otherUserId = dm.otherUser?.id;
              const presence = otherUserId ? presenceStatuses.get(otherUserId) : undefined;
              const dmUnread = unreadCounts.get(dm.id) || 0;
              return (
                <Link
                  key={dm.id}
                  href={`/dm/${dm.id}`}
                  className={`sidebar-item ${pathname === `/dm/${dm.id}` ? "active" : ""}`}
                >
                  <div className="relative">
                    <div className={`w-5 h-5 avatar text-2xs ${dm.otherUser?.isAgent ? "bg-teal-muted text-teal" : ""}`}>
                      {dm.otherUser?.isAgent ? <Bot size={10} /> : dm.name?.[0]?.toUpperCase() || "?"}
                    </div>
                    {(presence === "online" || dm.otherUser?.isAgent) && (
                      <span className="absolute -bottom-0.5 -right-0.5 w-2 h-2 rounded-full bg-teal border border-bg-surface" />
                    )}
                    {presence === "away" && (
                      <span className="absolute -bottom-0.5 -right-0.5 w-2 h-2 rounded-full bg-yellow-400 border border-bg-surface" />
                    )}
                  </div>
                  <span className={`truncate ${dmUnread > 0 ? "font-semibold text-text-primary" : ""}`}>{dm.name}</span>
                  {dmUnread > 0 && (
                    <span className="ml-auto text-2xs bg-accent text-white px-1.5 py-0.5 rounded-full font-medium">
                      {dmUnread}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        </div>

        {/* Agents */}
        <div>
          <div className="flex items-center px-3 mb-1">
            <span className="text-2xs font-semibold uppercase tracking-wider text-text-muted">
              AI Agents
            </span>
          </div>
          <Link
            href="/channels"
            className={`sidebar-item ${pathname === "/channels" ? "active" : ""}`}
          >
            <Bot size={16} />
            <span>Browse Agents</span>
            <span className="badge-teal ml-auto">37</span>
          </Link>
        </div>
      </nav>

      {/* User footer */}
      <div className="h-14 px-3 flex items-center gap-2 border-t border-border">
        <div className="w-8 h-8 avatar text-xs bg-accent-muted text-accent">
          {username?.[0]?.toUpperCase() || "?"}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium truncate text-text-primary">
            {username}
          </div>
          <div className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-teal" />
            <span className="text-2xs text-text-muted">Online</span>
          </div>
        </div>
        <Link
          href="/profile/settings"
          className="text-text-muted hover:text-text-primary transition-colors"
        >
          <Settings size={16} />
        </Link>
      </div>
    </aside>
  );
}
