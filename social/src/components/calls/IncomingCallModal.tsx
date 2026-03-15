"use client";

import { Phone, PhoneOff, Video } from "lucide-react";
import { useCallStore } from "@/stores/callStore";
import { useSocket } from "@/components/providers/SocketProvider";
import { acceptCall, rejectOrEndCall } from "@/components/providers/SocketProvider";

export default function IncomingCallModal() {
  const incomingCall = useCallStore((s) => s.incomingCall);
  const socket = useSocket();

  if (!incomingCall || !socket) return null;

  const isVideo = incomingCall.callType === "video";

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-bg-surface border border-border rounded-2xl shadow-2xl p-8 text-center max-w-sm w-full mx-4 animate-in fade-in zoom-in-95 duration-200">
        {/* Caller avatar */}
        <div className="w-20 h-20 avatar text-2xl bg-accent-muted text-accent mx-auto mb-4 ring-4 ring-accent/20 animate-pulse">
          {incomingCall.callerName[0]?.toUpperCase() || "?"}
        </div>

        <h3 className="font-heading text-lg font-semibold text-text-primary mb-1">
          {incomingCall.callerName}
        </h3>
        <p className="text-sm text-text-muted mb-8">
          Incoming {isVideo ? "video" : "voice"} call...
        </p>

        <div className="flex items-center justify-center gap-6">
          {/* Decline */}
          <button
            onClick={() => rejectOrEndCall(socket)}
            className="w-14 h-14 rounded-full bg-danger flex items-center justify-center text-white hover:bg-danger/90 transition-colors shadow-lg"
          >
            <PhoneOff size={24} />
          </button>

          {/* Accept */}
          <button
            onClick={() => acceptCall(socket)}
            className="w-14 h-14 rounded-full bg-teal flex items-center justify-center text-white hover:bg-teal/90 transition-colors shadow-lg"
          >
            {isVideo ? <Video size={24} /> : <Phone size={24} />}
          </button>
        </div>
      </div>
    </div>
  );
}
