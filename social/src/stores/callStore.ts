import { create } from "zustand";

export type CallType = "audio" | "video";

interface CallState {
  // Active call
  activeCall: {
    peerId: string;
    peerName: string;
    callType: CallType;
    channelId: string;
    startTime: number;
  } | null;
  // Incoming call
  incomingCall: {
    callerId: string;
    callerName: string;
    callType: CallType;
    channelId: string;
  } | null;
  // Media streams
  localStream: MediaStream | null;
  remoteStream: MediaStream | null;
  // State
  isMuted: boolean;
  isCameraOff: boolean;
  isScreenSharing: boolean;
  // Actions
  setActiveCall: (call: CallState["activeCall"]) => void;
  setIncomingCall: (call: CallState["incomingCall"]) => void;
  setLocalStream: (stream: MediaStream | null) => void;
  setRemoteStream: (stream: MediaStream | null) => void;
  toggleMute: () => void;
  toggleCamera: () => void;
  setScreenSharing: (sharing: boolean) => void;
  endCall: () => void;
}

export const useCallStore = create<CallState>((set, get) => ({
  activeCall: null,
  incomingCall: null,
  localStream: null,
  remoteStream: null,
  isMuted: false,
  isCameraOff: false,
  isScreenSharing: false,

  setActiveCall: (call) => set({ activeCall: call, incomingCall: null }),
  setIncomingCall: (call) => set({ incomingCall: call }),
  setLocalStream: (stream) => set({ localStream: stream }),
  setRemoteStream: (stream) => set({ remoteStream: stream }),
  toggleMute: () => {
    const { localStream, isMuted } = get();
    if (localStream) {
      localStream.getAudioTracks().forEach((t) => (t.enabled = isMuted));
    }
    set({ isMuted: !isMuted });
  },
  toggleCamera: () => {
    const { localStream, isCameraOff } = get();
    if (localStream) {
      localStream.getVideoTracks().forEach((t) => (t.enabled = isCameraOff));
    }
    set({ isCameraOff: !isCameraOff });
  },
  setScreenSharing: (sharing) => set({ isScreenSharing: sharing }),
  endCall: () => {
    const { localStream, remoteStream } = get();
    localStream?.getTracks().forEach((t) => t.stop());
    remoteStream?.getTracks().forEach((t) => t.stop());
    set({
      activeCall: null,
      incomingCall: null,
      localStream: null,
      remoteStream: null,
      isMuted: false,
      isCameraOff: false,
      isScreenSharing: false,
    });
  },
}));
