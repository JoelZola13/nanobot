"use client";

import Link from "next/link";
import {
  type FocusEvent,
  type MouseEvent,
  type ReactNode,
  useId,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  AlertCircle,
  Bot,
  Calendar,
  ExternalLink,
  Loader2,
  MapPin,
  RefreshCw,
  X,
} from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";
import { useEmbeddedNavigation } from "@/lib/useEmbeddedNavigation";

export type ProfilePopoverUser = {
  id?: string;
  username?: string;
  displayName?: string;
  avatarUrl?: string | null;
  bio?: string | null;
  location?: string | null;
  website?: string | null;
  status?: string | null;
  isAgent?: boolean;
  createdAt?: string | Date;
  channelCount?: number;
  postCount?: number;
};

type ProfilePopoverProps = {
  user?: ProfilePopoverUser | null;
  userId?: string;
  username?: string;
  children: ReactNode;
  className?: string;
  triggerClassName?: string;
  align?: "left" | "right";
};

const joinedFormatter = new Intl.DateTimeFormat("en", {
  month: "short",
  year: "numeric",
});

const statusLabel = (profile: ProfilePopoverUser) => {
  if (profile.isAgent) return "AI agent";
  return profile.status === "online" ? "Online" : "Away";
};

async function apiErrorMessage(res: Response, fallback: string) {
  const data = (await res.json().catch(() => null)) as {
    error?: string;
  } | null;
  return data?.error || fallback;
}

export default function ProfilePopover({
  user,
  userId,
  username,
  children,
  className = "",
  triggerClassName = "",
  align = "left",
}: ProfilePopoverProps) {
  const rootRef = useRef<HTMLSpanElement>(null);
  const closeTimeoutRef = useRef<ReturnType<typeof setTimeout>>();
  const triggerWasOpenOnPointerDownRef = useRef<boolean | null>(null);
  const dialogId = useId();
  const [open, setOpen] = useState(false);
  const [profile, setProfile] = useState<ProfilePopoverUser | null>(
    user || null,
  );
  const [loading, setLoading] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [retryKey, setRetryKey] = useState(0);
  const { withEmbed } = useEmbeddedNavigation();

  const lookup = useMemo(
    () => ({
      userId: userId || user?.id,
      username: username || user?.username,
    }),
    [user?.id, user?.username, userId, username],
  );

  useEffect(() => {
    setProfile(user || null);
    setProfileError(null);
  }, [user]);

  useEffect(() => {
    if (!open) return;
    if (profile?.channelCount !== undefined || profile?.website !== undefined) {
      return;
    }

    const params = new URLSearchParams();
    if (lookup.userId) {
      params.set("userId", lookup.userId);
    } else if (lookup.username) {
      params.set("username", lookup.username);
    } else {
      return;
    }

    const controller = new AbortController();
    setLoading(true);
    setProfileError(null);

    fetch(apiUrl(`/api/users/profile?${params.toString()}`), {
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok)
          throw new Error(await apiErrorMessage(res, "Profile unavailable."));
        return res.json() as Promise<ProfilePopoverUser>;
      })
      .then((data) => {
        setProfile(data);
        setProfileError(null);
      })
      .catch((error) => {
        if (error.name !== "AbortError") {
          setProfileError(
            error instanceof Error ? error.message : "Profile unavailable.",
          );
        }
      })
      .finally(() => setLoading(false));

    return () => controller.abort();
  }, [
    lookup.userId,
    lookup.username,
    open,
    profile?.channelCount,
    profile?.website,
    retryKey,
  ]);

  useEffect(() => {
    if (!open) return;

    const handleMouseDown = (event: globalThis.MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };

    document.addEventListener("mousedown", handleMouseDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleMouseDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  useEffect(() => {
    return () => {
      if (closeTimeoutRef.current) clearTimeout(closeTimeoutRef.current);
    };
  }, []);

  const openPopover = () => {
    if (closeTimeoutRef.current) clearTimeout(closeTimeoutRef.current);
    setOpen(true);
  };

  const scheduleClose = () => {
    if (closeTimeoutRef.current) clearTimeout(closeTimeoutRef.current);
    closeTimeoutRef.current = setTimeout(() => setOpen(false), 140);
  };

  const handleBlur = (event: FocusEvent<HTMLSpanElement>) => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
      setOpen(false);
    }
  };

  const handleClick = (event: MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const wasOpenOnPointerDown = triggerWasOpenOnPointerDownRef.current;
    triggerWasOpenOnPointerDownRef.current = null;

    if (wasOpenOnPointerDown === false) {
      setOpen(true);
      return;
    }

    setOpen((current) => !current);
  };

  const popoverClass =
    align === "right" ? "right-0 origin-top-right" : "left-0 origin-top-left";
  const triggerName =
    profile?.displayName || user?.displayName || username || user?.username;
  const triggerLabel = triggerName
    ? `Open ${triggerName} profile card`
    : "Open profile card";
  const retryProfile = () => {
    setProfileError(null);
    setRetryKey((current) => current + 1);
  };

  return (
    <span
      ref={rootRef}
      className={`relative inline-flex min-w-0 ${className}`}
      onBlur={handleBlur}
      onMouseEnter={openPopover}
      onMouseLeave={scheduleClose}
    >
      <button
        type="button"
        className={triggerClassName}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? dialogId : undefined}
        aria-label={triggerLabel}
        onPointerDown={() => {
          triggerWasOpenOnPointerDownRef.current = open;
        }}
        onClick={handleClick}
        onFocus={openPopover}
      >
        {children}
      </button>

      {open && (
        <span
          id={dialogId}
          role="dialog"
          aria-label={
            triggerName ? `${triggerName} profile card` : "Profile card"
          }
          className={`absolute top-full z-[80] mt-2 w-72 rounded-lg border border-border bg-bg-surface p-3 text-left shadow-xl ${popoverClass}`}
          onMouseEnter={openPopover}
          onMouseLeave={scheduleClose}
        >
          <ProfileCard
            profile={profile}
            loading={loading}
            error={profileError}
            fallbackName={username || user?.displayName || user?.username}
            onClose={() => setOpen(false)}
            onRetry={retryProfile}
            withEmbed={withEmbed}
          />
        </span>
      )}
    </span>
  );
}

function ProfileCard({
  profile,
  loading,
  error,
  fallbackName,
  onClose,
  onRetry,
  withEmbed,
}: {
  profile: ProfilePopoverUser | null;
  loading: boolean;
  error: string | null;
  fallbackName?: string;
  onClose: () => void;
  onRetry: () => void;
  withEmbed: (href: string) => string;
}) {
  const displayName = profile?.displayName || fallbackName || "Teammate";

  if (error) {
    return (
      <span data-testid="profile-popover-card" className="block">
        <span className="mb-2 flex items-start justify-between gap-3">
          <span className="min-w-0 text-sm font-semibold text-text-primary">
            {displayName}
          </span>
          <button
            type="button"
            onClick={onClose}
            className="sidebar-icon-button h-6 w-6 shrink-0"
            title="Close profile card"
            aria-label="Close profile card"
          >
            <X size={13} />
          </button>
        </span>
        <span
          data-testid="profile-popover-error"
          className="flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 px-2.5 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
        >
          <AlertCircle size={13} className="mt-0.5 shrink-0" />
          <span className="min-w-0 flex-1">{error}</span>
        </span>
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 inline-flex items-center gap-1.5 rounded-md border border-border px-2 py-1 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent"
          aria-label="Retry profile"
        >
          <RefreshCw size={12} />
          Retry
        </button>
      </span>
    );
  }

  if (!profile && loading) {
    return (
      <span data-testid="profile-popover-card" className="block">
        <span className="mb-2 flex items-start justify-between gap-3">
          <span className="min-w-0 text-sm font-semibold text-text-primary">
            {displayName}
          </span>
          <button
            type="button"
            onClick={onClose}
            className="sidebar-icon-button h-6 w-6 shrink-0"
            title="Close profile card"
            aria-label="Close profile card"
          >
            <X size={13} />
          </button>
        </span>
        <span className="flex items-center gap-2 text-sm text-text-muted">
          <Loader2 size={14} className="animate-spin" />
          Loading profile...
        </span>
      </span>
    );
  }

  const profileUrl = profile?.id
    ? withEmbed(`/profile/${profile.id}`)
    : undefined;
  const joinedAt = profile?.createdAt ? new Date(profile.createdAt) : null;
  const joinedText =
    joinedAt && Number.isFinite(joinedAt.getTime())
      ? joinedFormatter.format(joinedAt)
      : null;

  return (
    <span data-testid="profile-popover-card" className="block">
      <span className="mb-2 flex justify-end">
        <button
          type="button"
          onClick={onClose}
          className="sidebar-icon-button h-6 w-6"
          title="Close profile card"
          aria-label="Close profile card"
        >
          <X size={13} />
        </button>
      </span>
      <span className="flex items-start gap-3">
        <span
          className={`avatar flex h-12 w-12 shrink-0 text-base ${
            profile?.isAgent
              ? "bg-teal-muted text-teal"
              : "bg-accent-muted text-accent"
          }`}
        >
          {profile?.avatarUrl ? (
            <img
              src={profile.avatarUrl}
              alt=""
              className="h-full w-full rounded-full object-cover"
            />
          ) : profile?.isAgent ? (
            <Bot size={20} />
          ) : (
            displayName[0]?.toUpperCase()
          )}
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-semibold text-text-primary">
            {displayName}
          </span>
          {profile?.username && (
            <span className="block truncate text-xs text-text-muted">
              @{profile.username}
            </span>
          )}
          <span className="mt-1 inline-flex items-center gap-1.5 text-xs text-text-muted">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                profile?.status === "online" || profile?.isAgent
                  ? "bg-teal"
                  : "bg-border"
              }`}
            />
            {profile ? statusLabel(profile) : "Profile"}
          </span>
        </span>
      </span>

      {profile?.bio && (
        <span className="mt-3 block text-sm leading-5 text-text-secondary">
          {profile.bio}
        </span>
      )}

      <span className="mt-3 flex flex-wrap gap-3 text-xs text-text-muted">
        {profile?.location && (
          <span className="inline-flex min-w-0 items-center gap-1">
            <MapPin size={12} />
            <span className="max-w-[11rem] truncate">{profile.location}</span>
          </span>
        )}
        {joinedText && (
          <span className="inline-flex items-center gap-1">
            <Calendar size={12} />
            Joined {joinedText}
          </span>
        )}
      </span>

      {(profile?.channelCount !== undefined ||
        profile?.postCount !== undefined) && (
        <span className="mt-3 flex gap-4 border-t border-border pt-3 text-xs text-text-muted">
          {profile.channelCount !== undefined && (
            <span>
              <span className="font-semibold text-text-primary">
                {profile.channelCount}
              </span>{" "}
              channels
            </span>
          )}
          {profile.postCount !== undefined && (
            <span>
              <span className="font-semibold text-text-primary">
                {profile.postCount}
              </span>{" "}
              posts
            </span>
          )}
        </span>
      )}

      {profileUrl && (
        <Link
          href={profileUrl}
          className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs font-medium text-text-secondary hover:border-accent hover:text-accent"
        >
          View profile
          <ExternalLink size={12} />
        </Link>
      )}
    </span>
  );
}
