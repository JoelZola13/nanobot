"use client";

import { Bot, Zap, ArrowRight, CheckCircle2, AlertCircle, Loader2, Wrench } from "lucide-react";

export interface ActivityEvent {
  channelId: string;
  agent: { id: string; displayName: string; username: string };
  type: "thinking" | "working" | "delegating" | "tool_call" | "done" | "error";
  text: string;
  delegatedTo?: string;
  toolsUsed?: string[];
  timestamp?: string;
}

interface AgentActivityProps {
  activities: ActivityEvent[];
}

export default function AgentActivity({ activities }: AgentActivityProps) {
  if (activities.length === 0) return null;

  // Group by agent
  const agentMap = new Map<string, ActivityEvent[]>();
  for (const a of activities) {
    const existing = agentMap.get(a.agent.id) || [];
    existing.push(a);
    agentMap.set(a.agent.id, existing);
  }

  return (
    <div className="px-4 py-2 space-y-2">
      {Array.from(agentMap.entries()).map(([agentId, events]) => {
        const latest = events[events.length - 1];
        const isDone = latest.type === "done";
        const isError = latest.type === "error";
        const isActive = !isDone && !isError;

        return (
          <div
            key={agentId}
            className={`rounded-lg border transition-all duration-300 ${
              isActive
                ? "border-teal/30 bg-teal/5"
                : isError
                  ? "border-danger/30 bg-danger/5"
                  : "border-border bg-bg-surface opacity-60"
            }`}
          >
            {/* Header */}
            <div className="flex items-center gap-2 px-3 py-2">
              <div className="w-5 h-5 avatar text-2xs bg-teal-muted text-teal">
                <Bot size={12} />
              </div>
              <span className="text-xs font-medium text-teal">
                {latest.agent.displayName}
              </span>
              {isActive && (
                <Loader2 size={12} className="text-teal animate-spin ml-auto" />
              )}
              {isDone && (
                <CheckCircle2 size={12} className="text-teal ml-auto" />
              )}
              {isError && (
                <AlertCircle size={12} className="text-danger ml-auto" />
              )}
            </div>

            {/* Activity steps */}
            <div className="px-3 pb-2 space-y-1">
              {events.map((event, i) => (
                <ActivityStep key={i} event={event} />
              ))}
            </div>

            {/* Tools used summary */}
            {isDone && latest.toolsUsed && latest.toolsUsed.length > 0 && (
              <div className="px-3 pb-2 flex flex-wrap gap-1">
                {latest.toolsUsed.map((tool, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 text-2xs px-1.5 py-0.5 rounded bg-bg-elevated text-text-muted border border-border"
                  >
                    <Wrench size={8} />
                    {tool}
                  </span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ActivityStep({ event }: { event: ActivityEvent }) {
  const icon = (() => {
    switch (event.type) {
      case "thinking":
        return <Loader2 size={10} className="animate-spin text-teal" />;
      case "delegating":
        return <ArrowRight size={10} className="text-accent" />;
      case "tool_call":
        return <Zap size={10} className="text-yellow-400" />;
      case "working":
        return <Loader2 size={10} className="animate-spin text-text-muted" />;
      case "done":
        return <CheckCircle2 size={10} className="text-teal" />;
      case "error":
        return <AlertCircle size={10} className="text-danger" />;
    }
  })();

  return (
    <div className="flex items-start gap-2 text-2xs">
      <div className="mt-0.5 shrink-0">{icon}</div>
      <span
        className={`${
          event.type === "delegating"
            ? "text-accent font-medium"
            : event.type === "error"
              ? "text-danger"
              : "text-text-secondary"
        }`}
      >
        {event.text}
        {event.delegatedTo && (
          <span className="ml-1 text-accent font-medium">
            → {event.delegatedTo}
          </span>
        )}
      </span>
    </div>
  );
}
