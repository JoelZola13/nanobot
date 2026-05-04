"use client";

import { useState } from "react";
import type { ReactNode } from "react";
import { useRouter } from "next/navigation";
import { Bot, MapPin, MessageSquare, Search, Sparkles, UserRound } from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";

interface Person {
  id: string;
  username: string;
  displayName: string;
  avatarUrl: string | null;
  bio: string | null;
  status: string;
  location: string | null;
  isAgent?: boolean;
}

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

  return (
    <div className="group flex items-center gap-3 border-t border-border px-4 py-3 first:border-t-0 hover:bg-bg-hover">
      <div className="relative shrink-0">
        <div className={`h-10 w-10 avatar text-sm ${isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"}`}>
          {person.avatarUrl ? (
            <img src={person.avatarUrl} alt="" className="h-full w-full rounded-full object-cover" />
          ) : isAgent ? (
            <Bot size={18} />
          ) : (
            person.displayName[0]?.toUpperCase()
          )}
        </div>
        <span className={`absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full border-2 border-bg ${isOnline ? "bg-teal" : "bg-border"}`} />
      </div>

      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 items-center gap-2">
          <span className={`truncate text-sm font-semibold ${isAgent ? "text-teal" : "text-text-primary"}`}>
            {person.displayName}
          </span>
          {isAgent ? (
            <span className="badge-teal text-2xs">agent</span>
          ) : (
            <span className="truncate text-2xs text-text-muted">@{person.username}</span>
          )}
        </div>
        <div className="mt-0.5 flex min-w-0 items-center gap-2 text-xs text-text-muted">
          <span>{isOnline ? "Active" : "Away"}</span>
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
        </div>
      </div>

      <button
        onClick={() => onMessage(person.id)}
        disabled={loading}
        className="btn-ghost flex h-8 items-center gap-1.5 px-2 text-xs md:opacity-0 md:group-hover:opacity-100"
      >
        <MessageSquare size={13} />
        <span>{loading ? "Opening" : "Message"}</span>
      </button>
    </div>
  );
}
