"use client";

import { useState } from "react";
import {
  AtSign,
  Bell,
  ChevronDown,
  Hash,
  Info,
  Phone,
  Pin,
  Search,
  Users,
  Video,
} from "lucide-react";
import SearchPanel from "@/components/channels/SearchPanel";
import PinnedMessagesPanel from "@/components/channels/PinnedMessagesPanel";
import { useSocket } from "@/components/providers/SocketProvider";
import { initiateCall } from "@/components/providers/SocketProvider";
import ProfilePopover from "@/components/users/ProfilePopover";

interface TopBarProps {
  title: string;
  description?: string;
  type?: "channel" | "dm" | "feed" | "profile" | "mentions";
  memberCount?: number;
  channelId?: string;
  otherUserId?: string;
  otherUserName?: string;
}

export default function TopBar({
  title,
  description,
  type = "channel",
  memberCount,
  channelId,
  otherUserId,
  otherUserName,
}: TopBarProps) {
  const [showSearch, setShowSearch] = useState(false);
  const [showPins, setShowPins] = useState(false);
  const socket = useSocket();
  const canCall = type === "dm" && Boolean(socket && otherUserId && channelId);
  const isOnline = type === "dm" && description?.toLowerCase() === "online";
  const isOffline = type === "dm" && description?.toLowerCase() === "offline";
  const subtitle = description || (memberCount !== undefined ? `${memberCount} members` : undefined);
  const HeaderIcon = type === "channel" ? Hash : type === "mentions" ? AtSign : Users;

  const handleCall = (callType: "audio" | "video") => {
    if (socket && otherUserId && channelId) {
      initiateCall(socket, otherUserId, otherUserName || title, callType, channelId);
    }
  };

  const titleContent = (
    <>
      <h2 className="truncate font-heading text-base font-semibold text-text-primary">
        {title}
      </h2>
      <ChevronDown size={14} className="shrink-0 text-text-muted" />
    </>
  );

  return (
    <>
      <header className="h-14 px-4 flex items-center justify-between border-b border-border bg-bg-surface backdrop-blur-glass shrink-0">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-bg-elevated text-text-secondary">
            <HeaderIcon size={17} />
          </div>
          <div className="min-w-0">
            {type === "dm" && otherUserId ? (
              <ProfilePopover
                user={{ id: otherUserId, displayName: otherUserName || title }}
                triggerClassName="flex min-w-0 items-center gap-1.5 rounded-md pr-1 text-left hover:text-accent focus:outline-none focus:ring-2 focus:ring-accent/40"
              >
                {titleContent}
              </ProfilePopover>
            ) : (
              <button
                type="button"
                className="flex min-w-0 items-center gap-1.5 rounded-md pr-1 text-left hover:text-accent"
                title={title}
              >
                {titleContent}
              </button>
            )}
            {subtitle && (
              <div className="flex min-w-0 items-center gap-1.5 text-xs text-text-muted">
                {type === "dm" && (isOnline || isOffline) && (
                  <span className={`h-1.5 w-1.5 rounded-full ${isOnline ? "bg-teal" : "bg-border"}`} />
                )}
                <span className="truncate">{subtitle}</span>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1">
          {memberCount !== undefined && (
            <button className="btn-ghost h-9 px-2 flex items-center gap-1.5 text-sm" title="Members">
              <Users size={15} />
              <span>{memberCount}</span>
            </button>
          )}
          {canCall && (
            <>
              <button className="btn-ghost h-9 w-9 p-0" title="Voice call" onClick={() => handleCall("audio")}>
                <Phone size={16} />
              </button>
              <button className="btn-ghost h-9 w-9 p-0" title="Video call" onClick={() => handleCall("video")}>
                <Video size={16} />
              </button>
            </>
          )}
          <button
            className={`btn-ghost h-9 w-9 p-0 ${showPins ? "text-accent" : ""}`}
            onClick={() => setShowPins(!showPins)}
            title="Pinned messages"
          >
            <Pin size={16} />
          </button>
          <button
            className="btn-ghost h-9 w-9 p-0"
            onClick={() => setShowSearch(true)}
            title="Search messages"
          >
            <Search size={16} />
          </button>
          <button className="btn-ghost h-9 w-9 p-0" title="Channel details">
            <Info size={16} />
          </button>
          <button className="btn-ghost h-9 w-9 p-0 relative" title="Notifications">
            <Bell size={16} />
            <span className="absolute right-2 top-2 h-1.5 w-1.5 rounded-full bg-accent" />
          </button>
        </div>
      </header>

      {showSearch && <SearchPanel onClose={() => setShowSearch(false)} />}
      {showPins && channelId && (
        <PinnedMessagesPanel channelId={channelId} onClose={() => setShowPins(false)} />
      )}
    </>
  );
}
