"use client";

import { useEffect, useState } from "react";
import {
  AtSign,
  Bell,
  BellOff,
  Bookmark,
  ChevronDown,
  FileText,
  Hash,
  Info,
  Phone,
  Pin,
  Search,
  ShieldAlert,
  Users,
  Video,
  X,
} from "lucide-react";
import SearchPanel from "@/components/channels/SearchPanel";
import PinnedMessagesPanel from "@/components/channels/PinnedMessagesPanel";
import NotificationPreferencesPanel from "@/components/channels/NotificationPreferencesPanel";
import FilesPanel from "@/components/channels/FilesPanel";
import ChannelMembersPanel from "@/components/channels/ChannelMembersPanel";
import RemovedMessagesPanel from "@/components/channels/RemovedMessagesPanel";
import ConversationDetailsPanel from "@/components/channels/ConversationDetailsPanel";
import {
  useNotificationPreferences,
  useSocket,
} from "@/components/providers/SocketProvider";
import { initiateCall } from "@/components/providers/SocketProvider";
import ProfilePopover from "@/components/users/ProfilePopover";
import type { NotificationLevel } from "@/lib/notificationPreferences";

interface TopBarProps {
  title: string;
  description?: string;
  type?:
    | "channel"
    | "dm"
    | "feed"
    | "profile"
    | "mentions"
    | "saved"
    | "activity";
  memberCount?: number;
  detailsMemberCount?: number;
  channelVisibility?: "PUBLIC" | "PRIVATE";
  channelId?: string;
  otherUserId?: string;
  otherUserName?: string;
  canManageChannel?: boolean;
  canEditChannelTopic?: boolean;
}

export default function TopBar({
  title,
  description,
  type = "channel",
  memberCount,
  detailsMemberCount,
  channelVisibility = "PUBLIC",
  channelId,
  otherUserId,
  otherUserName,
  canManageChannel = false,
  canEditChannelTopic = false,
}: TopBarProps) {
  const [showSearch, setShowSearch] = useState(false);
  const [showPins, setShowPins] = useState(false);
  const [showFiles, setShowFiles] = useState(false);
  const [showMembers, setShowMembers] = useState(false);
  const [showRemoved, setShowRemoved] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const [currentDescription, setCurrentDescription] = useState(description);
  const [notificationLevel, setNotificationLevel] =
    useState<NotificationLevel | null>(null);
  const [callStarting, setCallStarting] = useState<"audio" | "video" | null>(
    null,
  );
  const [callError, setCallError] = useState<string | null>(null);
  const socket = useSocket();
  const { preferences, setPreference } = useNotificationPreferences();
  const canShowCallControls =
    type === "dm" && Boolean(otherUserId && channelId);
  const canStartCall = Boolean(socket && otherUserId && channelId);
  const canConfigureNotifications = Boolean(
    channelId && (type === "channel" || type === "dm"),
  );
  const canShowRemovedMessages = Boolean(
    channelId && type === "channel" && canManageChannel,
  );
  const canShowDetails = Boolean(
    channelId && (type === "channel" || type === "dm"),
  );
  const canShowPins = Boolean(
    channelId && (type === "channel" || type === "dm"),
  );
  useEffect(() => {
    setCurrentDescription(description);
  }, [description]);
  useEffect(() => {
    setCallError(null);
    setCallStarting(null);
  }, [channelId]);

  const isOnline =
    type === "dm" && currentDescription?.toLowerCase() === "online";
  const isOffline =
    type === "dm" && currentDescription?.toLowerCase() === "offline";
  const subtitle =
    currentDescription ||
    (memberCount !== undefined ? `${memberCount} members` : undefined);
  const resolvedDetailsMemberCount = detailsMemberCount ?? memberCount;
  const HeaderIcon =
    type === "channel"
      ? Hash
      : type === "mentions"
        ? AtSign
        : type === "saved"
          ? Bookmark
          : type === "activity"
            ? Bell
            : Users;
  const resolvedNotificationLevel = channelId
    ? (preferences[channelId] ?? notificationLevel)
    : notificationLevel;
  const NotificationIcon =
    resolvedNotificationLevel === "MUTED" ? BellOff : Bell;

  const callStartErrorMessage = (
    error: unknown,
    callType: "audio" | "video",
  ) => {
    if (
      error instanceof DOMException &&
      ["NotAllowedError", "PermissionDeniedError"].includes(error.name)
    ) {
      return callType === "video"
        ? "Camera and microphone permission is required to start a video call."
        : "Microphone permission is required to start a voice call.";
    }

    if (
      error instanceof DOMException &&
      ["NotFoundError", "DevicesNotFoundError"].includes(error.name)
    ) {
      return callType === "video"
        ? "No camera or microphone was found for this video call."
        : "No microphone was found for this voice call.";
    }

    return "Could not start the call. Check your browser permissions and try again.";
  };

  const handleCall = async (callType: "audio" | "video") => {
    if (!socket || !otherUserId || !channelId) {
      setCallError("Calls are still connecting. Try again in a moment.");
      return;
    }

    setCallError(null);
    setCallStarting(callType);
    try {
      await initiateCall(
        socket,
        otherUserId,
        otherUserName || title,
        callType,
        channelId,
      );
    } catch (error) {
      setCallError(callStartErrorMessage(error, callType));
    } finally {
      setCallStarting(null);
    }
  };

  const titleContent = (
    <>
      <h2 className="truncate font-heading text-base font-semibold text-text-primary">
        {title}
      </h2>
      {(type === "dm" || canShowDetails) && (
        <ChevronDown size={14} className="shrink-0 text-text-muted" />
      )}
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
            ) : canShowDetails ? (
              <button
                type="button"
                onClick={() => {
                  setShowDetails(!showDetails);
                  setShowPins(false);
                  setShowFiles(false);
                  setShowMembers(false);
                  setShowRemoved(false);
                  setShowNotifications(false);
                }}
                className="flex min-w-0 items-center gap-1.5 rounded-md pr-1 text-left hover:text-accent"
                title={title}
                aria-label={
                  type === "dm"
                    ? "Open DM profile details"
                    : "Open conversation details from header"
                }
              >
                {titleContent}
              </button>
            ) : (
              <div
                className="flex min-w-0 items-center gap-1.5 rounded-md pr-1 text-left"
                title={title}
              >
                {titleContent}
              </div>
            )}
            {subtitle && (
              <div className="flex min-w-0 items-center gap-1.5 text-xs text-text-muted">
                {type === "dm" && (isOnline || isOffline) && (
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${isOnline ? "bg-teal" : "bg-border"}`}
                  />
                )}
                <span className="truncate">{subtitle}</span>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1">
          {memberCount !== undefined && (
            <button
              type="button"
              className={`btn-ghost h-9 px-2 flex items-center gap-1.5 text-sm ${showMembers ? "text-accent" : ""}`}
              title="Members"
              aria-label={`Members, ${memberCount}`}
              onClick={() => {
                setShowMembers(!showMembers);
                setShowPins(false);
                setShowFiles(false);
                setShowRemoved(false);
                setShowNotifications(false);
                setShowDetails(false);
              }}
            >
              <Users size={15} />
              <span>{memberCount}</span>
            </button>
          )}
          {canShowCallControls && (
            <>
              <button
                type="button"
                className="btn-ghost h-9 w-9 p-0 disabled:cursor-not-allowed disabled:opacity-50"
                title={
                  callStarting === "audio"
                    ? "Starting voice call"
                    : canStartCall
                      ? "Voice call"
                      : "Calls are connecting"
                }
                aria-label={
                  callStarting === "audio"
                    ? "Starting voice call"
                    : "Voice call"
                }
                disabled={!canStartCall || callStarting !== null}
                onClick={() => void handleCall("audio")}
              >
                <Phone size={16} />
              </button>
              <button
                type="button"
                className="btn-ghost h-9 w-9 p-0 disabled:cursor-not-allowed disabled:opacity-50"
                title={
                  callStarting === "video"
                    ? "Starting video call"
                    : canStartCall
                      ? "Video call"
                      : "Calls are connecting"
                }
                aria-label={
                  callStarting === "video"
                    ? "Starting video call"
                    : "Video call"
                }
                disabled={!canStartCall || callStarting !== null}
                onClick={() => void handleCall("video")}
              >
                <Video size={16} />
              </button>
            </>
          )}
          {canShowPins && (
            <button
              type="button"
              className={`btn-ghost h-9 w-9 p-0 ${showPins ? "text-accent" : ""}`}
              onClick={() => {
                setShowPins(!showPins);
                setShowFiles(false);
                setShowMembers(false);
                setShowRemoved(false);
                setShowNotifications(false);
                setShowDetails(false);
              }}
              title="Pinned messages"
              aria-label="Pinned messages"
            >
              <Pin size={16} />
            </button>
          )}
          {channelId && (type === "channel" || type === "dm") && (
            <button
              type="button"
              className={`btn-ghost h-9 w-9 p-0 ${showFiles ? "text-accent" : ""}`}
              onClick={() => {
                setShowFiles(!showFiles);
                setShowPins(false);
                setShowMembers(false);
                setShowRemoved(false);
                setShowNotifications(false);
                setShowDetails(false);
              }}
              title="Files"
              aria-label="Files"
            >
              <FileText size={16} />
            </button>
          )}
          {canShowRemovedMessages && (
            <button
              type="button"
              className={`btn-ghost h-9 w-9 p-0 ${showRemoved ? "text-accent" : ""}`}
              onClick={() => {
                setShowRemoved(!showRemoved);
                setShowPins(false);
                setShowFiles(false);
                setShowMembers(false);
                setShowNotifications(false);
                setShowDetails(false);
              }}
              title="Removed messages"
              aria-label="Removed messages"
            >
              <ShieldAlert size={16} />
            </button>
          )}
          <button
            type="button"
            className="btn-ghost h-9 w-9 p-0"
            onClick={() => {
              setShowSearch(true);
              setShowPins(false);
              setShowFiles(false);
              setShowMembers(false);
              setShowRemoved(false);
              setShowNotifications(false);
              setShowDetails(false);
            }}
            title="Search messages"
            aria-label="Search messages"
          >
            <Search size={16} />
          </button>
          {canShowDetails && (
            <button
              type="button"
              className={`btn-ghost h-9 w-9 p-0 ${showDetails ? "text-accent" : ""}`}
              onClick={() => {
                setShowDetails(!showDetails);
                setShowPins(false);
                setShowFiles(false);
                setShowMembers(false);
                setShowRemoved(false);
                setShowNotifications(false);
              }}
              title={type === "dm" ? "DM details" : "Channel details"}
              aria-label={type === "dm" ? "DM details" : "Channel details"}
            >
              <Info size={16} />
            </button>
          )}
          {canConfigureNotifications && channelId && (
            <button
              type="button"
              className={`btn-ghost h-9 w-9 p-0 relative ${showNotifications ? "text-accent" : ""}`}
              onClick={() => {
                setShowNotifications(!showNotifications);
                setShowPins(false);
                setShowFiles(false);
                setShowMembers(false);
                setShowRemoved(false);
                setShowDetails(false);
              }}
              title="Notifications"
              aria-label="Notifications"
            >
              <NotificationIcon size={16} />
              {resolvedNotificationLevel &&
                resolvedNotificationLevel !== "MUTED" && (
                  <span className="absolute right-2 top-2 h-1.5 w-1.5 rounded-full bg-accent" />
                )}
            </button>
          )}
        </div>
      </header>

      {callError && (
        <div
          className="flex items-start justify-between gap-3 border-b border-red-300 bg-red-50 px-4 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
          role="status"
          data-testid="call-start-error"
        >
          <span className="min-w-0 flex-1">{callError}</span>
          <button
            type="button"
            onClick={() => setCallError(null)}
            className="rounded-md p-0.5 text-red-700 hover:bg-red-100 dark:text-red-200 dark:hover:bg-red-900/30"
            aria-label="Dismiss call error"
          >
            <X size={13} />
          </button>
        </div>
      )}

      {showSearch && <SearchPanel onClose={() => setShowSearch(false)} />}
      {showPins && channelId && (
        <PinnedMessagesPanel
          channelId={channelId}
          onClose={() => setShowPins(false)}
        />
      )}
      {showFiles && channelId && (
        <FilesPanel channelId={channelId} onClose={() => setShowFiles(false)} />
      )}
      {showMembers && channelId && (
        <ChannelMembersPanel
          channelId={channelId}
          initialCanManage={canManageChannel}
          onClose={() => setShowMembers(false)}
        />
      )}
      {showRemoved && channelId && (
        <RemovedMessagesPanel
          channelId={channelId}
          onClose={() => setShowRemoved(false)}
        />
      )}
      {showDetails && canShowDetails && channelId && (
        <ConversationDetailsPanel
          channelId={channelId}
          title={title}
          description={currentDescription}
          type={type === "dm" ? "dm" : "channel"}
          channelVisibility={channelVisibility}
          memberCount={resolvedDetailsMemberCount}
          canEditTopic={canEditChannelTopic}
          onClose={() => setShowDetails(false)}
          onTopicChange={setCurrentDescription}
          onOpenMembers={
            memberCount !== undefined
              ? () => {
                  setShowDetails(false);
                  setShowMembers(true);
                }
              : undefined
          }
          onOpenPins={() => {
            setShowDetails(false);
            setShowPins(true);
          }}
          onOpenFiles={() => {
            setShowDetails(false);
            setShowFiles(true);
          }}
          onOpenNotifications={() => {
            setShowDetails(false);
            setShowNotifications(true);
          }}
        />
      )}
      {showNotifications && channelId && (
        <NotificationPreferencesPanel
          channelId={channelId}
          onClose={() => setShowNotifications(false)}
          onLevelChange={(level) => {
            setNotificationLevel(level);
            setPreference(channelId, level);
          }}
        />
      )}
    </>
  );
}
