"use client";

import { useEffect, useRef, useState } from "react";
import { PhoneOff, Mic, MicOff, Video, VideoOff, Monitor } from "lucide-react";
import { useCallStore } from "@/stores/callStore";
import { useSocket } from "@/components/providers/SocketProvider";
import { rejectOrEndCall } from "@/components/providers/SocketProvider";

export default function CallOverlay() {
  const activeCall = useCallStore((s) => s.activeCall);
  const localStream = useCallStore((s) => s.localStream);
  const remoteStream = useCallStore((s) => s.remoteStream);
  const isMuted = useCallStore((s) => s.isMuted);
  const isCameraOff = useCallStore((s) => s.isCameraOff);
  const toggleMute = useCallStore((s) => s.toggleMute);
  const toggleCamera = useCallStore((s) => s.toggleCamera);
  const socket = useSocket();

  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!activeCall) return;
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - activeCall.startTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [activeCall]);

  useEffect(() => {
    if (localVideoRef.current && localStream) {
      localVideoRef.current.srcObject = localStream;
    }
  }, [localStream]);

  useEffect(() => {
    if (remoteVideoRef.current && remoteStream) {
      remoteVideoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  if (!activeCall) return null;

  const isVideo = activeCall.callType === "video";
  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div className="fixed inset-0 z-[100] bg-[#0A0A0A] flex flex-col">
      {/* Main area */}
      <div className="flex-1 flex items-center justify-center relative">
        {isVideo && remoteStream ? (
          // Video call - remote video fills screen
          <video
            ref={remoteVideoRef}
            autoPlay
            playsInline
            className="w-full h-full object-cover"
          />
        ) : (
          // Audio call - show avatar
          <div className="text-center">
            <div className="w-32 h-32 avatar text-5xl bg-accent-muted text-accent mx-auto mb-6 ring-4 ring-accent/20">
              {activeCall.peerName[0]?.toUpperCase() || "?"}
            </div>
            <h2 className="font-heading text-2xl font-bold text-text-primary mb-2">
              {activeCall.peerName}
            </h2>
            <p className="text-text-muted font-mono tabular-nums">
              {formatTime(elapsed)}
            </p>
            {/* Audio visualization */}
            <div className="flex items-center justify-center gap-1 mt-4">
              {Array.from({ length: 5 }).map((_, i) => (
                <div
                  key={i}
                  className="w-1 bg-accent rounded-full animate-pulse"
                  style={{
                    height: `${12 + Math.random() * 20}px`,
                    animationDelay: `${i * 0.15}s`,
                  }}
                />
              ))}
            </div>
          </div>
        )}

        {/* Local video PiP */}
        {isVideo && localStream && (
          <div className="absolute bottom-4 right-4 w-48 h-36 rounded-xl overflow-hidden border-2 border-border shadow-xl">
            <video
              ref={localVideoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-4 py-8 bg-gradient-to-t from-black/80 to-transparent">
        {/* Duration */}
        <span className="absolute top-4 left-1/2 -translate-x-1/2 text-sm font-mono text-text-muted tabular-nums">
          {formatTime(elapsed)}
        </span>

        <button
          onClick={toggleMute}
          className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${
            isMuted ? "bg-danger text-white" : "bg-bg-elevated text-text-primary hover:bg-bg-hover"
          }`}
          title={isMuted ? "Unmute" : "Mute"}
        >
          {isMuted ? <MicOff size={20} /> : <Mic size={20} />}
        </button>

        {isVideo && (
          <button
            onClick={toggleCamera}
            className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${
              isCameraOff ? "bg-danger text-white" : "bg-bg-elevated text-text-primary hover:bg-bg-hover"
            }`}
            title={isCameraOff ? "Turn on camera" : "Turn off camera"}
          >
            {isCameraOff ? <VideoOff size={20} /> : <Video size={20} />}
          </button>
        )}

        <button
          className="w-12 h-12 rounded-full bg-bg-elevated text-text-primary hover:bg-bg-hover flex items-center justify-center transition-colors"
          title="Share screen"
        >
          <Monitor size={20} />
        </button>

        <button
          onClick={() => socket && rejectOrEndCall(socket)}
          className="w-14 h-14 rounded-full bg-danger text-white flex items-center justify-center hover:bg-danger/90 transition-colors shadow-lg"
          title="End call"
        >
          <PhoneOff size={24} />
        </button>
      </div>
    </div>
  );
}
