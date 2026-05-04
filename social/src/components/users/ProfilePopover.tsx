"use client";

import Link from "next/link";
import {
  type FocusEvent,
  type MouseEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Bot, Calendar, ExternalLink, MapPin } from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";

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
  const [open, setOpen] = useState(false);
  const [profile, setProfile] = useState<ProfilePopoverUser | null>(
    user || null,
  );
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);

  const lookup = useMemo(
    () => ({
      userId: userId || user?.id,
      username: username || user?.username,
    }),
    [user?.id, user?.username, userId, username],
  );

  useEffect(() => {
    setProfile(user || null);
    setFailed(false);
  }, [user]);

  useEffect(() => {
    if (!open || failed) return;
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

    fetch(apiUrl(`/api/users/profile?${params.toString()}`), {
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) throw new Error("Profile unavailable");
        return res.json() as Promise<ProfilePopoverUser>;
      })
      .then((data) => {
        setProfile(data);
        setFailed(false);
      })
      .catch((error) => {
        if (error.name !== "AbortError") setFailed(true);
      })
      .finally(() => setLoading(false));

    return () => controller.abort();
  }, [
    failed,
    lookup.userId,
    lookup.username,
    open,
    profile?.channelCount,
    profile?.website,
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
    setOpen((current) => !current);
  };

  const popoverClass =
    align === "right"
      ? "right-0 origin-top-right"
      : "left-0 origin-top-left";

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
        onClick={handleClick}
        onFocus={openPopover}
      >
        {children}
      </button>

      {open && (
        <span
          role="dialog"
          className={`absolute top-full z-[80] mt-2 w-72 rounded-lg border border-border bg-bg-surface p-3 text-left shadow-xl ${popoverClass}`}
          onMouseEnter={openPopover}
          onMouseLeave={scheduleClose}
        >
          <ProfileCard
            profile={profile}
            loading={loading}
            failed={failed}
            fallbackName={username || user?.displayName || user?.username}
          />
        </span>
      )}
    </span>
  );
}

function ProfileCard({
  profile,
  loading,
  failed,
  fallbackName,
}: {
  profile: ProfilePopoverUser | null;
  loading: boolean;
  failed: boolean;
  fallbackName?: string;
}) {
  if (failed) {
    return (
      <span className="block text-sm text-text-secondary">
        Profile unavailable.
      </span>
    );
  }

  if (!profile && loading) {
    return (
      <span className="block text-sm text-text-muted">Loading profile...</span>
    );
  }

  const displayName = profile?.displayName || fallbackName || "Teammate";
  const profileUrl = profile?.id ? `/profile/${profile.id}` : undefined;
  const joinedAt = profile?.createdAt ? new Date(profile.createdAt) : null;
  const joinedText =
    joinedAt && Number.isFinite(joinedAt.getTime())
      ? joinedFormatter.format(joinedAt)
      : null;

  return (
    <span className="block">
      <span className="flex items-start gap-3">
        <span
          className={`avatar flex h-12 w-12 shrink-0 text-base ${
            profile?.isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"
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

      {(profile?.channelCount !== undefined || profile?.postCount !== undefined) && (
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
