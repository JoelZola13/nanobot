"use client";

import { useState } from "react";
import { Hash, Users, Bell, Pin, Search, Phone, Video } from "lucide-react";
import SearchPanel from "@/components/channels/SearchPanel";
import PinnedMessagesPanel from "@/components/channels/PinnedMessagesPanel";
import { useSocket } from "@/components/providers/SocketProvider";
import { initiateCall } from "@/components/providers/SocketProvider";

interface TopBarProps {
  title: string;
  description?: string;
  type?: "channel" | "dm" | "feed" | "profile";
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

  const handleCall = (callType: "audio" | "video") => {
    if (socket && otherUserId && channelId) {
      initiateCall(socket, otherUserId, otherUserName || title, callType, channelId);
    }
  };

  return (
    <>
      <header className="h-14 px-4 flex items-center justify-between border-b border-border bg-bg shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          {type === "channel" && <Hash size={18} className="text-text-muted shrink-0" />}
          {type === "dm" && <Users size={18} className="text-text-muted shrink-0" />}
          <h2 className="font-heading font-semibold text-text-primary truncate">
            {title}
          </h2>
          {description && (
            <>
              <span className="text-border mx-1">|</span>
              <span className="text-sm text-text-muted truncate">
                {description}
              </span>
            </>
          )}
        </div>

        <div className="flex items-center gap-1">
          {memberCount !== undefined && (
            <button className="btn-ghost flex items-center gap-1.5 text-sm">
              <Users size={14} />
              <span>{memberCount}</span>
            </button>
          )}
          {type === "dm" && (
            <>
              <button className="btn-ghost p-2" title="Voice call" onClick={() => handleCall("audio")}>
                <Phone size={16} />
              </button>
              <button className="btn-ghost p-2" title="Video call" onClick={() => handleCall("video")}>
                <Video size={16} />
              </button>
            </>
          )}
          <button
            className={`btn-ghost p-2 ${showPins ? "text-accent" : ""}`}
            onClick={() => setShowPins(!showPins)}
            title="Pinned messages"
          >
            <Pin size={16} />
          </button>
          <button
            className="btn-ghost p-2"
            onClick={() => setShowSearch(true)}
            title="Search messages"
          >
            <Search size={16} />
          </button>
          <button className="btn-ghost p-2 relative">
            <Bell size={16} />
            <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-accent rounded-full" />
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
