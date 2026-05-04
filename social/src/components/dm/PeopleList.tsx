"use client";

import { useState } from "react";
import type { ReactNode } from "react";
import { useRouter } from "next/navigation";
import { Bot, CheckCircle2, MapPin, MessageSquare, Search, Sparkles, UserPlus, UserRound } from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";
import ProfilePopover from "@/components/users/ProfilePopover";

interface Person {
  id: string;
  username: string;
  displayName: string;
  avatarUrl: string | null;
  bio: string | null;
  status: string;
  location: string | null;
  lastSeenAt: string | null;
  createdAt: string;
  isAgent?: boolean;
}

const RECENT_TEAMMATE_WINDOW_MS = 30 * 24 * 60 * 60 * 1000;

const isAwaitingFirstSignIn = (person: Person) => !person.isAgent && !person.lastSeenAt && person.status !== "online";

const isRecentlyAdded = (person: Person) => {
  if (person.isAgent) return false;
  const createdAt = new Date(person.createdAt).getTime();
  return Number.isFinite(createdAt) && Date.now() - createdAt <= RECENT_TEAMMATE_WINDOW_MS;
};

export default function PeopleList({
  people,
  agents = [],
  currentUserId: _currentUserId,
}: {
  people: Person[];
  agents?: Person[];
  currentUserId: string;
}) {
  const router = useRouter();
  const [starting, setStarting] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  const startDM = async (userId: string) => {
    setStarting(userId);
    try {
      const res = await fetch(apiUrl("/api/dm"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId }),
      });
      if (res.ok) {
        const { channelId } = await res.json();
        router.push(`/dm/${channelId}`);
        router.refresh();
      }
    } finally {
      setStarting(null);
    }
  };

  const normalizedQuery = query.trim().toLowerCase();
  const matches = (person: Person) => {
    if (!normalizedQuery) return true;
    return [
      person.displayName,
      person.username,
      person.bio,
      person.location,
    ]
      .filter(Boolean)
      .some((value) => value!.toLowerCase().includes(normalizedQuery));
  };

  const online = people.filter((p) => p.status === "online" && matches(p));
  const offline = people.filter((p) => p.status !== "online" && matches(p));
  const visibleAgents = agents.filter(matches);
  const visibleCount = online.length + offline.length + visibleAgents.length;
  const onboardingCount = people.filter(isAwaitingFirstSignIn).length;
  const recentlyAddedCount = people.filter(isRecentlyAdded).length;

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl px-6 py-5">
        <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="min-w-0">
            <h2 className="font-heading text-xl font-semibold text-text-primary">
              Direct messages
            </h2>
            <p className="text-sm text-text-muted">
              {people.length} teammates, {agents.length} agents
            </p>
          </div>
          <div className="relative w-full md:w-80">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search people and agents"
              className="input-field w-full pl-9 text-sm"
            />
          </div>
        </div>

        {onboardingCount > 0 && (
          <div className="mb-4 border-y border-border bg-bg-surface px-4 py-3">
            <div className="flex flex-col gap-3 md:flex-row md:items-center">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-border bg-bg-elevated text-accent">
                <UserPlus size={17} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="break-words text-sm font-semibold leading-5 text-text-primary">
                  {onboardingCount} teammate{onboardingCount === 1 ? "" : "s"} waiting on first sign-in
                </div>
                <div className="mt-0.5 break-words text-xs leading-5 text-text-muted">
                  Share LibChatMain + NanobotMain, then have them sign in to LibreChat once. Messages uses that same session.
                </div>
              </div>
              {recentlyAddedCount > 0 && (
                <div className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-border px-2 py-1 text-2xs font-semibold uppercase text-text-muted">
                  <CheckCircle2 size={12} className="text-teal" />
                  {recentlyAddedCount} added recently
                </div>
              )}
            </div>
          </div>
        )}

        <div className="overflow-hidden rounded-lg border border-border bg-bg-surface">
          {visibleCount === 0 ? (
            <div className="px-5 py-10 text-center">
              <UserRound size={32} className="mx-auto mb-2 text-text-muted" />
              <h3 className="font-heading text-base font-semibold text-text-primary">No matches</h3>
              <p className="text-sm text-text-muted">Try a different name, username, location, or agent.</p>
            </div>
          ) : (
            <>
              <DirectorySection
                title="Online"
                count={online.length}
                icon={<span className="h-2 w-2 rounded-full bg-teal" />}
                people={online}
                onMessage={startDM}
                loadingId={starting}
              />
              <DirectorySection
                title="Teammates"
                count={offline.length}
                icon={<UserRound size={13} />}
                people={offline}
                onMessage={startDM}
                loadingId={starting}
              />
              <DirectorySection
                title="AI agents"
                count={visibleAgents.length}
                icon={<Sparkles size={13} className="text-teal" />}
                people={visibleAgents}
                onMessage={startDM}
                loadingId={starting}
                isAgentSection
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function DirectorySection({
  title,
  count,
  icon,
  people,
  onMessage,
  loadingId,
  isAgentSection = false,
}: {
  title: string;
  count: number;
  icon: ReactNode;
  people: Person[];
  onMessage: (id: string) => void;
  loadingId: string | null;
  isAgentSection?: boolean;
}) {
  if (count === 0) return null;

  return (
    <section className="border-t border-border first:border-t-0">
      <div className="flex items-center gap-2 bg-bg-elevated px-4 py-2 text-xs font-semibold uppercase text-text-muted">
        {icon}
        <span>{title}</span>
        <span className="ml-auto">{count}</span>
      </div>
      <div>
        {people.map((person) => (
          <PersonRow
            key={person.id}
            person={person}
            onMessage={onMessage}
            loading={loadingId === person.id}
            isAgent={isAgentSection || person.isAgent}
          />
        ))}
      </div>
    </section>
  );
}

function PersonRow({
  person,
  onMessage,
  loading,
  isAgent = false,
}: {
  person: Person;
  onMessage: (id: string) => void;
  loading: boolean;
  isAgent?: boolean;
}) {
  const isOnline = person.status === "online" || isAgent;
  const awaitingFirstSignIn = isAwaitingFirstSignIn(person);
  const recentlyAdded = isRecentlyAdded(person);
  const statusText = isAgent ? "Active" : awaitingFirstSignIn ? "Needs first sign-in" : isOnline ? "Active" : "Away";

  return (
    <div className="group flex items-center gap-3 border-t border-border px-4 py-3 first:border-t-0 hover:bg-bg-hover">
      <ProfilePopover
        user={person}
        className="min-w-0 flex-1"
        triggerClassName="flex min-w-0 flex-1 items-center gap-3 rounded-md text-left focus:outline-none focus:ring-2 focus:ring-accent/40"
      >
        <span className="relative shrink-0">
          <span className={`h-10 w-10 avatar text-sm ${isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"}`}>
            {person.avatarUrl ? (
              <img src={person.avatarUrl} alt="" className="h-full w-full rounded-full object-cover" />
            ) : isAgent ? (
              <Bot size={18} />
            ) : (
              person.displayName[0]?.toUpperCase()
            )}
          </span>
          <span className={`absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full border-2 border-bg ${isOnline ? "bg-teal" : "bg-border"}`} />
        </span>

        <span className="min-w-0 flex-1">
          <span className="flex min-w-0 items-center gap-2">
            <span className={`truncate text-sm font-semibold ${isAgent ? "text-teal" : "text-text-primary"}`}>
              {person.displayName}
            </span>
            {isAgent ? (
              <span className="badge-teal text-2xs">agent</span>
            ) : recentlyAdded ? (
              <span className="rounded-full bg-accent-muted px-1.5 py-0.5 text-2xs font-semibold uppercase text-accent">
                New
              </span>
            ) : (
              <span className="truncate text-2xs text-text-muted">@{person.username}</span>
            )}
            {!isAgent && recentlyAdded && (
              <span className="truncate text-2xs text-text-muted">@{person.username}</span>
            )}
          </span>
          <span className="mt-0.5 flex min-w-0 items-center gap-2 text-xs text-text-muted">
            <span>{statusText}</span>
            {person.location && !isAgent && (
              <>
                <span>/</span>
                <span className="inline-flex min-w-0 items-center gap-1 truncate">
                  <MapPin size={11} className="shrink-0" />
                  <span className="truncate">{person.location}</span>
                </span>
              </>
            )}
            {person.bio && (
              <>
                <span>/</span>
                <span className="truncate">{person.bio}</span>
              </>
            )}
          </span>
          {awaitingFirstSignIn && (
            <span className="mt-1 block break-words text-2xs leading-4 text-text-muted">
              LibreChat sign-in unlocks Messages automatically.
            </span>
          )}
        </span>
      </ProfilePopover>

      <button
        data-testid="start-dm-button"
        data-user-id={person.id}
        onClick={() => onMessage(person.id)}
        disabled={loading}
        className="btn-ghost flex h-8 items-center gap-1.5 px-2 text-xs md:opacity-0 md:group-hover:opacity-100"
        type="button"
      >
        <MessageSquare size={13} />
        <span>{loading ? "Opening" : "Message"}</span>
      </button>
    </div>
  );
}
