"use client";

import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Loader2,
  Mic,
  MicOff,
  Monitor,
  PhoneOff,
  Video,
  VideoOff,
  X,
} from "lucide-react";
import { useCallStore } from "@/stores/callStore";
import { useSocket } from "@/components/providers/SocketProvider";
import {
  rejectOrEndCall,
  replaceActiveCallVideoTrack,
} from "@/components/providers/SocketProvider";

export default function CallOverlay() {
  const activeCall = useCallStore((s) => s.activeCall);
  const localStream = useCallStore((s) => s.localStream);
  const remoteStream = useCallStore((s) => s.remoteStream);
  const isMuted = useCallStore((s) => s.isMuted);
  const isCameraOff = useCallStore((s) => s.isCameraOff);
  const isScreenSharing = useCallStore((s) => s.isScreenSharing);
  const toggleMute = useCallStore((s) => s.toggleMute);
  const toggleCamera = useCallStore((s) => s.toggleCamera);
  const setLocalStream = useCallStore((s) => s.setLocalStream);
  const setScreenSharing = useCallStore((s) => s.setScreenSharing);
  const socket = useSocket();

  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);
  const screenStreamRef = useRef<MediaStream | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [screenShareBusy, setScreenShareBusy] = useState(false);
  const [screenShareError, setScreenShareError] = useState<string | null>(null);

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

  useEffect(() => {
    if (activeCall) return;
    screenStreamRef.current?.getTracks().forEach((track) => track.stop());
    screenStreamRef.current = null;
    cameraStreamRef.current = null;
    setScreenSharing(false);
    setScreenShareBusy(false);
    setScreenShareError(null);
  }, [activeCall, setScreenSharing]);

  if (!activeCall) return null;

  const isVideo = activeCall.callType === "video";

  const stopScreenShare = async () => {
    const screenStream = screenStreamRef.current;
    screenStream?.getTracks().forEach((track) => track.stop());
    screenStreamRef.current = null;

    const cameraStream = cameraStreamRef.current;
    cameraStreamRef.current = null;
    const cameraTrack = cameraStream
      ?.getVideoTracks()
      .find((track) => track.readyState !== "ended");
    if (cameraTrack) {
      await replaceActiveCallVideoTrack(cameraTrack);
    }
    if (cameraStream) {
      setLocalStream(cameraStream);
    }
    setScreenSharing(false);
    setScreenShareBusy(false);
  };

  const startScreenShare = async () => {
    if (!isVideo) return;
    setScreenShareBusy(true);
    setScreenShareError(null);

    try {
      if (!navigator.mediaDevices?.getDisplayMedia) {
        throw new Error("Screen sharing is not supported in this browser.");
      }

      const screenStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: false,
      });
      const screenTrack = screenStream.getVideoTracks()[0];
      if (!screenTrack) {
        screenStream.getTracks().forEach((track) => track.stop());
        throw new Error("No screen video track was available.");
      }

      cameraStreamRef.current = localStream;
      screenStreamRef.current = screenStream;
      screenTrack.onended = () => {
        void stopScreenShare();
      };
      await replaceActiveCallVideoTrack(screenTrack);
      const audioTracks = localStream?.getAudioTracks() || [];
      setLocalStream(new MediaStream([...audioTracks, screenTrack]));
      setScreenSharing(true);
    } catch (error) {
      screenStreamRef.current?.getTracks().forEach((track) => track.stop());
      screenStreamRef.current = null;
      cameraStreamRef.current = null;
      setScreenSharing(false);
      setScreenShareError(
        error instanceof DOMException &&
          ["NotAllowedError", "PermissionDeniedError"].includes(error.name)
          ? "Screen sharing permission is required."
          : error instanceof Error && error.message.trim()
            ? error.message
            : "Screen sharing could not start.",
      );
    } finally {
      setScreenShareBusy(false);
    }
  };

  const toggleScreenShare = () => {
    if (isScreenSharing) {
      void stopScreenShare();
    } else {
      void startScreenShare();
    }
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div
      className="fixed inset-0 z-[100] bg-[#0A0A0A] flex flex-col"
      role="dialog"
      aria-modal="true"
      aria-label={`${isVideo ? "Video" : "Voice"} call with ${activeCall.peerName}`}
      data-testid="call-overlay"
    >
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
            isMuted
              ? "bg-danger text-white"
              : "bg-bg-elevated text-text-primary hover:bg-bg-hover"
          }`}
          type="button"
          title={isMuted ? "Unmute" : "Mute"}
          aria-label={isMuted ? "Unmute" : "Mute"}
        >
          {isMuted ? <MicOff size={20} /> : <Mic size={20} />}
        </button>

        {isVideo && (
          <button
            onClick={toggleCamera}
            className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors ${
              isCameraOff
                ? "bg-danger text-white"
                : "bg-bg-elevated text-text-primary hover:bg-bg-hover"
            }`}
            type="button"
            title={isCameraOff ? "Turn on camera" : "Turn off camera"}
            aria-label={isCameraOff ? "Turn on camera" : "Turn off camera"}
          >
            {isCameraOff ? <VideoOff size={20} /> : <Video size={20} />}
          </button>
        )}

        {isVideo && (
          <button
            onClick={toggleScreenShare}
            className={`w-12 h-12 rounded-full flex items-center justify-center transition-colors disabled:cursor-wait disabled:opacity-60 ${
              isScreenSharing
                ? "bg-accent text-white"
                : "bg-bg-elevated text-text-primary hover:bg-bg-hover"
            }`}
            type="button"
            disabled={screenShareBusy}
            title={
              screenShareBusy
                ? "Starting screen share"
                : isScreenSharing
                  ? "Stop sharing screen"
                  : "Share screen"
            }
            aria-label={
              screenShareBusy
                ? "Starting screen share"
                : isScreenSharing
                  ? "Stop sharing screen"
                  : "Share screen"
            }
          >
            {screenShareBusy ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Monitor size={20} />
            )}
          </button>
        )}

        <button
          onClick={() => socket && rejectOrEndCall(socket)}
          className="w-14 h-14 rounded-full bg-danger text-white flex items-center justify-center hover:bg-danger/90 transition-colors shadow-lg"
          type="button"
          title="End call"
          aria-label="End call"
        >
          <PhoneOff size={24} />
        </button>
      </div>
      {screenShareError && (
        <div
          data-testid="call-screen-share-error"
          role="alert"
          className="absolute bottom-24 left-1/2 flex max-w-md -translate-x-1/2 items-start gap-2 rounded-lg border border-red-900/60 bg-red-950/90 px-3 py-2 text-xs text-red-100 shadow-lg"
        >
          <AlertCircle size={14} className="mt-0.5 shrink-0" />
          <span className="min-w-0 flex-1">{screenShareError}</span>
          <button
            type="button"
            onClick={() => setScreenShareError(null)}
            className="rounded p-0.5 text-red-100 hover:bg-red-900/60"
            aria-label="Dismiss screen share error"
          >
            <X size={12} />
          </button>
        </div>
      )}
    </div>
  );
}
