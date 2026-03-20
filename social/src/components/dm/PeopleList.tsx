"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { MessageSquare, MapPin, Bot } from "lucide-react";
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

  const online = people.filter((p) => p.status === "online");
  const offline = people.filter((p) => p.status !== "online");

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Online people */}
        {online.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-3 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-teal" />
              Online — {online.length}
            </h3>
            <div className="space-y-1">
              {online.map((person) => (
                <PersonCard key={person.id} person={person} onMessage={startDM} loading={starting === person.id} />
              ))}
            </div>
          </div>
        )}

        {/* Offline people */}
        {offline.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-3">
              Offline — {offline.length}
            </h3>
            <div className="space-y-1">
              {offline.map((person) => (
                <PersonCard key={person.id} person={person} onMessage={startDM} loading={starting === person.id} />
              ))}
            </div>
          </div>
        )}

        {/* AI Agents */}
        {agents.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-3 flex items-center gap-2">
              <Bot size={12} className="text-teal" />
              AI Agents — {agents.length}
            </h3>
            <div className="space-y-1">
              {agents.map((agent) => (
                <PersonCard key={agent.id} person={agent} onMessage={startDM} loading={starting === agent.id} isAgent />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function PersonCard({
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
  return (
    <div className="flex items-center gap-3 px-4 py-3 rounded-xl glass-soft hover:bg-bg-hover transition-colors group">
      <div className="relative">
        <div className={`w-10 h-10 avatar text-sm ${isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"}`}>
          {person.avatarUrl ? (
            <img src={person.avatarUrl} alt="" className="w-full h-full object-cover rounded-full" />
          ) : isAgent ? (
            <Bot size={18} />
          ) : (
            person.displayName[0]?.toUpperCase()
          )}
        </div>
        {!isAgent && person.status === "online" && (
          <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full bg-teal border-2 border-bg" />
        )}
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={`font-medium text-sm ${isAgent ? "text-teal" : "text-text-primary"}`}>
            {person.displayName}
          </span>
          {isAgent && <span className="badge-teal text-2xs">agent</span>}
          {!isAgent && (
            <span className="text-2xs text-text-muted">@{person.username}</span>
          )}
        </div>
        {person.bio && (
          <p className="text-xs text-text-secondary truncate mt-0.5">{person.bio}</p>
        )}
        {person.location && !isAgent && (
          <div className="flex items-center gap-1 mt-0.5 text-2xs text-text-muted">
            <MapPin size={10} />
            <span>{person.location}</span>
          </div>
        )}
      </div>

      <button
        onClick={() => onMessage(person.id)}
        disabled={loading}
        className="btn-ghost text-xs px-3 py-1.5 opacity-0 group-hover:opacity-100 transition-all flex items-center gap-1.5"
      >
        <MessageSquare size={12} />
        <span>{loading ? "Opening..." : "Message"}</span>
      </button>
    </div>
  );
}
