"use client";

import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import { PanelLeftOpen } from "lucide-react";
import type { ChannelInfo } from "@/types";
import { useUnreadStore } from "@/stores/unreadStore";
import Sidebar from "./Sidebar";

type DmChannel = ChannelInfo & {
  otherUser?: {
    id: string;
    displayName: string;
    avatarUrl: string | null;
    isAgent: boolean;
    status: string;
  } | null;
};

interface ResponsiveMessagesShellProps {
  channels: ChannelInfo[];
  dms: DmChannel[];
  userId: string;
  initialUnreadCounts: Record<string, number>;
  activityUnreadCount: number;
  children: ReactNode;
}

export default function ResponsiveMessagesShell({
  channels,
  dms,
  userId,
  initialUnreadCounts,
  activityUnreadCount,
  children,
}: ResponsiveMessagesShellProps) {
  const pathname = usePathname();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isInteractive, setIsInteractive] = useState(false);
  const [isFramed, setIsFramed] = useState(false);
  const hasMountedPathnameRef = useRef(false);
  const hydrateUnreadCounts = useUnreadStore((s) => s.hydrate);

  useEffect(() => {
    hydrateUnreadCounts(initialUnreadCounts);
  }, [hydrateUnreadCounts, initialUnreadCounts]);

  useEffect(() => {
    if (!hasMountedPathnameRef.current) {
      hasMountedPathnameRef.current = true;
      return;
    }

    setSidebarOpen(false);
  }, [pathname]);

  useEffect(() => {
    try {
      setIsFramed(window.self !== window.top);
    } catch {
      setIsFramed(true);
    }

    const interactiveTimer = window.setTimeout(
      () => setIsInteractive(true),
      1000,
    );
    return () => window.clearTimeout(interactiveTimer);
  }, []);

  return (
    <div
      className={`sv-messages-shell flex h-screen overflow-hidden ${isFramed ? "is-framed" : ""} ${isInteractive ? "is-interactive" : ""}`}
    >
      <button
        type="button"
        className="sv-mobile-sidebar-toggle disabled:cursor-not-allowed disabled:opacity-60"
        aria-label="Open messages sidebar"
        aria-expanded={sidebarOpen}
        disabled={!isInteractive}
        onClick={() => setSidebarOpen(true)}
      >
        <PanelLeftOpen size={18} />
      </button>

      <button
        type="button"
        className={`sv-mobile-sidebar-backdrop ${sidebarOpen ? "is-mobile-open" : ""}`}
        aria-label="Close messages sidebar"
        onClick={() => setSidebarOpen(false)}
      />

      <Sidebar
        channels={channels}
        dms={dms}
        userId={userId}
        activityUnreadCount={activityUnreadCount}
        mobileOpen={sidebarOpen}
        onMobileClose={() => setSidebarOpen(false)}
      />
      <main className="sv-messages-main flex flex-1 min-w-0 flex-col overflow-hidden">
        {children}
      </main>
    </div>
  );
}
