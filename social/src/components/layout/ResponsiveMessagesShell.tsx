"use client";

import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { PanelLeftOpen } from "lucide-react";
import type { ChannelInfo } from "@/types";
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
  children: ReactNode;
}

export default function ResponsiveMessagesShell({
  channels,
  dms,
  userId,
  children,
}: ResponsiveMessagesShellProps) {
  const pathname = usePathname();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    setSidebarOpen(false);
  }, [pathname]);

  return (
    <div className="sv-messages-shell flex h-screen overflow-hidden">
      <button
        type="button"
        className="sv-mobile-sidebar-toggle"
        aria-label="Open messages sidebar"
        aria-expanded={sidebarOpen}
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
        mobileOpen={sidebarOpen}
        onMobileClose={() => setSidebarOpen(false)}
      />
      <main className="sv-messages-main flex flex-1 min-w-0 flex-col overflow-hidden">{children}</main>
    </div>
  );
}
