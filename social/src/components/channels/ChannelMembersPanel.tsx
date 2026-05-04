"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bot,
  Check,
  Search,
  ShieldCheck,
  Trash2,
  UserPlus,
  Users,
  X,
} from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";

type MemberRole = "owner" | "admin" | "member" | string;

type UserSummary = {
  id: string;
  username: string;
  displayName: string;
  avatarUrl: string | null;
  isAgent: boolean;
  status: string;
};

type ChannelMember = {
  channelId: string;
  userId: string;
  role: MemberRole;
  joinedAt: string;
  user: UserSummary;
};

type MembersResponse = {
  channelId: string;
  canManage: boolean;
  members: ChannelMember[];
};

const roleRank = (role: MemberRole) => {
  if (role === "owner") return 0;
  if (role === "admin") return 1;
  return 2;
};

const sortMembers = (members: ChannelMember[]) =>
  [...members].sort((a, b) => {
    const byRole = roleRank(a.role) - roleRank(b.role);
    if (byRole !== 0) return byRole;
    return a.user.displayName.localeCompare(b.user.displayName);
  });

export default function ChannelMembersPanel({
  channelId,
  initialCanManage = false,
  onClose,
}: {
  channelId: string;
  initialCanManage?: boolean;
  onClose: () => void;
}) {
  const [members, setMembers] = useState<ChannelMember[]>([]);
  const [canManage, setCanManage] = useState(initialCanManage);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<UserSummary[]>([]);
  const [searching, setSearching] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    setLoading(true);
    setError(null);

    fetch(apiUrl(`/api/channels/${channelId}/members`), { cache: "no-store" })
      .then((response) => {
        if (!response.ok) throw new Error("Failed to load members");
        return response.json();
      })
      .then((data: MembersResponse) => {
        if (cancelled) return;
        setMembers(sortMembers(data.members || []));
        setCanManage(Boolean(data.canManage));
      })
      .catch(() => {
        if (!cancelled) setError("Members could not load.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [channelId]);

  useEffect(() => {
    const normalizedQuery = query.trim();
    if (!canManage || normalizedQuery.length < 2) {
      setSearchResults([]);
      setSearching(false);
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setSearching(true);
      fetch(apiUrl(`/api/users/search?q=${encodeURIComponent(normalizedQuery)}`), {
        signal: controller.signal,
      })
        .then((response) => {
          if (!response.ok) throw new Error("Failed to search users");
          return response.json();
        })
        .then((results: UserSummary[]) => {
          setSearchResults(results);
        })
        .catch((err) => {
          if (err?.name !== "AbortError") setSearchResults([]);
        })
        .finally(() => {
          setSearching(false);
        });
    }, 250);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [canManage, query]);

  const memberIds = useMemo(
    () => new Set(members.map((member) => member.userId)),
    [members],
  );
  const candidates = useMemo(
    () => searchResults.filter((user) => !memberIds.has(user.id)),
    [memberIds, searchResults],
  );

  const upsertMember = (member: ChannelMember) => {
    setMembers((currentMembers) => {
      const exists = currentMembers.some((current) => current.userId === member.userId);
      const nextMembers = exists
        ? currentMembers.map((current) =>
            current.userId === member.userId ? member : current,
          )
        : [...currentMembers, member];
      return sortMembers(nextMembers);
    });
  };

  const addMember = async (userId: string) => {
    setBusyId(`add:${userId}`);
    setError(null);

    try {
      const response = await fetch(apiUrl(`/api/channels/${channelId}/members`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId }),
      });
      if (!response.ok) throw new Error("Failed to add member");

      const member = (await response.json()) as ChannelMember;
      upsertMember(member);
      setQuery("");
      setSearchResults([]);
    } catch {
      setError("Member could not be added.");
    } finally {
      setBusyId(null);
    }
  };

  const updateRole = async (member: ChannelMember, role: "admin" | "member") => {
    if (member.role === role) return;
    setBusyId(`role:${member.userId}`);
    setError(null);

    try {
      const response = await fetch(
        apiUrl(`/api/channels/${channelId}/members/${member.userId}`),
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ role }),
        },
      );
      if (!response.ok) throw new Error("Failed to update member role");

      upsertMember((await response.json()) as ChannelMember);
    } catch {
      setError("Member role could not be updated.");
    } finally {
      setBusyId(null);
    }
  };

  const removeMember = async (member: ChannelMember) => {
    setBusyId(`remove:${member.userId}`);
    setError(null);

    try {
      const response = await fetch(
        apiUrl(`/api/channels/${channelId}/members/${member.userId}`),
        { method: "DELETE" },
      );
      if (!response.ok) throw new Error("Failed to remove member");

      setMembers((currentMembers) =>
        currentMembers.filter((current) => current.userId !== member.userId),
      );
    } catch {
      setError("Member could not be removed.");
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="absolute right-4 top-14 z-40 w-[26rem] max-w-[calc(100vw-2rem)] overflow-hidden rounded-xl border border-border bg-bg-surface shadow-xl">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex min-w-0 items-center gap-2">
          <Users size={14} className="text-accent" />
          <span className="font-heading text-sm font-semibold text-text-primary">
            Members
          </span>
          <span className="rounded-full bg-bg-elevated px-1.5 py-0.5 text-2xs font-medium text-text-muted">
            {members.length}
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-text-muted hover:bg-bg-hover"
          aria-label="Close members"
          title="Close"
        >
          <X size={14} />
        </button>
      </div>

      {canManage && (
        <div className="border-b border-border p-3">
          <label className="mb-1.5 block text-xs font-semibold uppercase text-text-muted">
            Add teammate
          </label>
          <div className="flex items-center gap-2 rounded-lg border border-border bg-bg-base px-3 py-2">
            <Search size={14} className="shrink-0 text-text-muted" />
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search people or agents"
              className="min-w-0 flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-muted"
            />
          </div>
          {query.trim().length >= 2 && (
            <div className="mt-2 max-h-40 overflow-y-auto rounded-lg border border-border bg-bg-base">
              {searching && (
                <div className="px-3 py-3 text-sm text-text-muted">
                  Searching...
                </div>
              )}
              {!searching && candidates.length === 0 && (
                <div className="px-3 py-3 text-sm text-text-muted">
                  No teammates to add
                </div>
              )}
              {!searching &&
                candidates.map((user) => (
                  <SearchResultRow
                    key={user.id}
                    user={user}
                    busy={busyId === `add:${user.id}`}
                    onAdd={() => addMember(user.id)}
                  />
                ))}
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="m-3 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200">
          {error}
        </div>
      )}

      <div className="max-h-[28rem] overflow-y-auto">
        {loading && (
          <div className="px-4 py-6 text-center text-sm text-text-muted">
            Loading...
          </div>
        )}

        {!loading && members.length === 0 && (
          <div className="px-6 py-10 text-center">
            <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-lg border border-border bg-bg-elevated text-text-muted">
              <Users size={20} />
            </div>
            <div className="font-heading text-sm font-semibold text-text-primary">
              No members yet
            </div>
            <div className="mt-1 text-sm leading-5 text-text-muted">
              Add teammates to make this channel available.
            </div>
          </div>
        )}

        {!loading &&
          members.map((member) => (
            <MemberRow
              key={member.userId}
              member={member}
              canManage={canManage}
              busyId={busyId}
              onUpdateRole={updateRole}
              onRemove={removeMember}
            />
          ))}
      </div>
    </div>
  );
}

function Avatar({ user }: { user: UserSummary }) {
  return (
    <span
      className={`avatar h-9 w-9 text-xs ${
        user.isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"
      }`}
    >
      {user.avatarUrl ? (
        <img src={user.avatarUrl} alt="" className="h-full w-full rounded-full object-cover" />
      ) : user.isAgent ? (
        <Bot size={16} />
      ) : (
        user.displayName[0]?.toUpperCase()
      )}
    </span>
  );
}

function RoleBadge({ role }: { role: MemberRole }) {
  if (role === "owner") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-accent/40 bg-accent-muted px-2 py-0.5 text-2xs font-semibold uppercase text-accent">
        <ShieldCheck size={11} />
        Owner
      </span>
    );
  }

  if (role === "admin") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full border border-teal/30 bg-teal-muted px-2 py-0.5 text-2xs font-semibold uppercase text-teal">
        <Check size={11} />
        Admin
      </span>
    );
  }

  return (
    <span className="rounded-full border border-border bg-bg-elevated px-2 py-0.5 text-2xs font-semibold uppercase text-text-muted">
      Member
    </span>
  );
}

function SearchResultRow({
  user,
  busy,
  onAdd,
}: {
  user: UserSummary;
  busy: boolean;
  onAdd: () => void;
}) {
  return (
    <div className="flex items-center gap-3 border-t border-border px-3 py-2 first:border-t-0">
      <Avatar user={user} />
      <div className="min-w-0 flex-1">
        <div className="truncate text-sm font-medium text-text-primary">
          {user.displayName}
        </div>
        <div className="truncate text-xs text-text-muted">
          @{user.username}
        </div>
      </div>
      <button
        type="button"
        onClick={onAdd}
        disabled={busy}
        className="btn-ghost inline-flex h-8 items-center gap-1.5 px-2 text-xs"
      >
        <UserPlus size={13} />
        <span>{busy ? "Adding" : "Add"}</span>
      </button>
    </div>
  );
}

function MemberRow({
  member,
  canManage,
  busyId,
  onUpdateRole,
  onRemove,
}: {
  member: ChannelMember;
  canManage: boolean;
  busyId: string | null;
  onUpdateRole: (member: ChannelMember, role: "admin" | "member") => void;
  onRemove: (member: ChannelMember) => void;
}) {
  const canEditMember = canManage && member.role !== "owner";
  const roleBusy = busyId === `role:${member.userId}`;
  const removeBusy = busyId === `remove:${member.userId}`;

  return (
    <div className="flex items-center gap-3 border-b border-border/50 px-4 py-3 last:border-b-0 hover:bg-bg-hover/60">
      <Avatar user={member.user} />
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate text-sm font-semibold text-text-primary">
            {member.user.displayName}
          </span>
          {member.user.isAgent && <span className="badge-teal text-2xs">agent</span>}
        </div>
        <div className="mt-0.5 flex min-w-0 items-center gap-2 text-xs text-text-muted">
          <span className="truncate">@{member.user.username}</span>
          <span>/</span>
          <RoleBadge role={member.role} />
        </div>
      </div>

      {canEditMember && (
        <div className="flex shrink-0 items-center gap-1.5">
          <select
            value={member.role === "admin" ? "admin" : "member"}
            onChange={(event) =>
              onUpdateRole(member, event.target.value as "admin" | "member")
            }
            disabled={roleBusy || removeBusy}
            className="h-8 rounded-md border border-border bg-bg-base px-2 text-xs text-text-primary outline-none focus:border-accent"
            aria-label={`Role for ${member.user.displayName}`}
          >
            <option value="member">Member</option>
            <option value="admin">Admin</option>
          </select>
          <button
            type="button"
            onClick={() => onRemove(member)}
            disabled={roleBusy || removeBusy}
            className="btn-ghost h-8 w-8 p-0 text-text-muted hover:text-danger"
            title="Remove member"
            aria-label={`Remove ${member.user.displayName}`}
          >
            <Trash2 size={13} />
          </button>
        </div>
      )}
    </div>
  );
}
