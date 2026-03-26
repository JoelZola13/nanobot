"use client";

import { createContext, useContext, useEffect, useRef, useState } from "react";
import { Socket } from "socket.io-client";
import { getSocket, disconnectSocket } from "@/lib/socket";
import { usePresenceStore } from "@/stores/presenceStore";
import { useUnreadStore } from "@/stores/unreadStore";
import { useCallStore } from "@/stores/callStore";
import { playMessageSound, playCallRingtone, stopCallRingtone } from "@/lib/sounds";
import { createPeerConnection, getLocalMedia, createOffer, createAnswer } from "@/lib/webrtc";
import type { MessageData } from "@/types";

const SocketContext = createContext<Socket | null>(null);

export function useSocket() {
  return useContext(SocketContext);
}

// Store peer connection globally so call handlers can access it
let peerConnection: RTCPeerConnection | null = null;

export default function SocketProvider({
  userId,
  channelIds,
  userName,
  children,
}: {
  userId: string;
  channelIds: string[];
  userName?: string;
  children: React.ReactNode;
}) {
  const [socketState, setSocketState] = useState<Socket | null>(null);
  const socketRef = useRef<Socket | null>(null);
  const setPresence = usePresenceStore((s) => s.setStatus);
  const setAllPresence = usePresenceStore((s) => s.setAll);
  const incrementUnread = useUnreadStore((s) => s.increment);
  const { setIncomingCall, setLocalStream, setRemoteStream, endCall } = useCallStore.getState();

  useEffect(() => {
    const socket = getSocket(userId);
    socketRef.current = socket;
    setSocketState(socket);

    // Pass userName in auth for call signaling
    if (userName) {
      (socket.auth as Record<string, string>).userName = userName;
    }

    if (!socket.connected) {
      socket.connect();
    }

    socket.on("connect", () => {
      socket.emit("join:channels", channelIds);
    });

    if (socket.connected) {
      socket.emit("join:channels", channelIds);
    }

    // ── Presence ──
    socket.on("presence:list", (entries: { userId: string; status: string }[]) => {
      setAllPresence(entries);
    });

    socket.on("presence:update", ({ userId: uid, status }: { userId: string; status: string }) => {
      setPresence(uid, status as "online" | "away" | "offline");
    });

    // Heartbeat every 30s
    const heartbeatInterval = setInterval(() => {
      socket.emit("heartbeat");
    }, 30000);

    // Focus/blur detection
    const handleFocus = () => socket.emit("presence:active");
    const handleBlur = () => socket.emit("presence:away");
    window.addEventListener("focus", handleFocus);
    window.addEventListener("blur", handleBlur);

    // ── Unread + Notification Sounds ──
    socket.on("message:new", (msg: MessageData) => {
      // Track unread for channels not currently viewed
      incrementUnread(msg.channelId);
      // Play notification sound if window is not focused
      if (document.hidden) {
        playMessageSound();
      }
    });

    // ── WebRTC Call Signaling ──
    socket.on("call:incoming", ({ callerId, callerName, callType, channelId }) => {
      setIncomingCall({ callerId, callerName, callType, channelId });
      playCallRingtone();
    });

    socket.on("call:offer", async ({ callerId, offer }) => {
      try {
        const { activeCall } = useCallStore.getState();
        if (!activeCall) return;

        const stream = await getLocalMedia(activeCall.callType === "video");
        setLocalStream(stream);

        peerConnection = createPeerConnection(
          (remoteStream) => setRemoteStream(remoteStream),
          (candidate) => socket.emit("call:ice-candidate", { targetUserId: callerId, candidate }),
        );

        stream.getTracks().forEach((track) => peerConnection!.addTrack(track, stream));
        await peerConnection.setRemoteDescription(new RTCSessionDescription(offer));
        const answer = await createAnswer(peerConnection);
        socket.emit("call:answer", { targetUserId: callerId, answer });
      } catch (err) {
        console.error("Error handling call offer:", err);
      }
    });

    socket.on("call:answer", async ({ answer }) => {
      try {
        if (peerConnection) {
          await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
        }
      } catch (err) {
        console.error("Error handling call answer:", err);
      }
    });

    socket.on("call:ice-candidate", async ({ candidate }) => {
      try {
        if (peerConnection && candidate) {
          await peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
        }
      } catch (err) {
        console.error("Error handling ICE candidate:", err);
      }
    });

    socket.on("call:ended", () => {
      peerConnection?.close();
      peerConnection = null;
      endCall();
      stopCallRingtone();
    });

    socket.on("call:rejected", () => {
      peerConnection?.close();
      peerConnection = null;
      endCall();
    });

    return () => {
      clearInterval(heartbeatInterval);
      window.removeEventListener("focus", handleFocus);
      window.removeEventListener("blur", handleBlur);
      peerConnection?.close();
      peerConnection = null;
      disconnectSocket();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  return (
    <SocketContext.Provider value={socketState}>
      {children}
    </SocketContext.Provider>
  );
}

// Export function to initiate a call
export async function initiateCall(
  socket: Socket,
  targetUserId: string,
  targetName: string,
  callType: "audio" | "video",
  channelId: string,
): Promise<void> {
  const { setActiveCall, setLocalStream, setRemoteStream } = useCallStore.getState();

  const stream = await getLocalMedia(callType === "video");
  setLocalStream(stream);

  peerConnection = createPeerConnection(
    (remoteStream) => setRemoteStream(remoteStream),
    (candidate) => socket.emit("call:ice-candidate", { targetUserId, candidate }),
  );

  stream.getTracks().forEach((track) => peerConnection!.addTrack(track, stream));

  const offer = await createOffer(peerConnection);
  socket.emit("call:initiate", { targetUserId, callType, channelId });
  socket.emit("call:offer", { targetUserId, offer });

  setActiveCall({
    peerId: targetUserId,
    peerName: targetName,
    callType,
    channelId,
    startTime: Date.now(),
  });
}

// Export function to accept an incoming call
export function acceptCall(_socket: Socket): void {
  const { incomingCall, setActiveCall } = useCallStore.getState();
  if (!incomingCall) return;

  stopCallRingtone();
  setActiveCall({
    peerId: incomingCall.callerId,
    peerName: incomingCall.callerName,
    callType: incomingCall.callType,
    channelId: incomingCall.channelId,
    startTime: Date.now(),
  });
}

// Export function to reject/end a call
export function rejectOrEndCall(socket: Socket): void {
  const { activeCall, incomingCall, endCall } = useCallStore.getState();
  const targetId = activeCall?.peerId || incomingCall?.callerId;

  if (targetId) {
    if (activeCall) {
      socket.emit("call:end", { targetUserId: targetId });
    } else {
      socket.emit("call:reject", { targetUserId: targetId });
    }
  }

  peerConnection?.close();
  peerConnection = null;
  endCall();
  stopCallRingtone();
}
