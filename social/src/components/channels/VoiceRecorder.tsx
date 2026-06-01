"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  AlertCircle,
  Mic,
  RefreshCw,
  Square,
  X,
  Send,
  Loader2,
} from "lucide-react";

interface VoiceRecorderProps {
  onSend: (audioBlob: Blob, duration: number) => Promise<void>;
  onCancel: () => void;
}

export default function VoiceRecorder({
  onSend,
  onCancel,
}: VoiceRecorderProps) {
  const [recording, setRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [levels, setLevels] = useState<number[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval>>();
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number>();
  const streamRef = useRef<MediaStream | null>(null);

  const startRecording = useCallback(async () => {
    let stream: MediaStream | null = null;
    try {
      setError(null);
      setAudioBlob(null);
      setDuration(0);
      setLevels([]);

      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        setError("Voice recording is not supported in this browser.");
        setRecording(false);
        return;
      }

      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Set up audio analyser for waveform
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      source.connect(analyser);
      analyserRef.current = analyser;

      const recorderStream = stream;
      const preferredMimeType = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
        "audio/wav",
      ].find((type) => MediaRecorder.isTypeSupported(type));
      const mediaRecorder = new MediaRecorder(
        recorderStream,
        preferredMimeType ? { mimeType: preferredMimeType } : undefined,
      );

      chunksRef.current = [];
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type:
            mediaRecorder.mimeType ||
            chunksRef.current[0]?.type ||
            preferredMimeType ||
            "audio/webm",
        });
        setAudioBlob(blob);
        recorderStream.getTracks().forEach((t) => t.stop());
        if (streamRef.current === recorderStream) {
          streamRef.current = null;
        }
      };

      mediaRecorder.start(100);
      mediaRecorderRef.current = mediaRecorder;
      setRecording(true);
      setDuration(0);

      // Timer
      timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000);

      // Waveform animation
      const updateLevels = () => {
        if (!analyserRef.current) return;
        const data = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(data);
        const avg = Array.from(data.slice(0, 16)).map((v) => v / 255);
        setLevels(avg);
        animFrameRef.current = requestAnimationFrame(updateLevels);
      };
      updateLevels();
    } catch (err) {
      stream?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      const denied =
        err instanceof DOMException &&
        ["NotAllowedError", "PermissionDeniedError"].includes(err.name);
      setError(
        denied
          ? "Microphone permission is required to record a voice message."
          : "Voice recording could not start.",
      );
      setRecording(false);
    }
  }, []);

  useEffect(() => {
    startRecording();
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [startRecording]);

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    if (timerRef.current) clearInterval(timerRef.current);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    setRecording(false);
  };

  const handleSend = async () => {
    if (!audioBlob) return;
    setSending(true);
    setError(null);
    try {
      await onSend(audioBlob, duration);
    } catch (sendError) {
      setError(
        sendError instanceof Error && sendError.message.trim()
          ? sendError.message
          : "Voice message could not be sent.",
      );
    } finally {
      setSending(false);
    }
  };

  const formatDuration = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div className="px-4 pb-4 pt-2">
      <div
        className="bg-bg-surface border border-danger/30 rounded-xl overflow-hidden"
        data-testid="voice-recorder"
      >
        <div className="flex items-center gap-3 px-4 py-3">
          {/* Recording indicator */}
          <div
            className={`w-3 h-3 rounded-full ${recording ? "bg-danger animate-pulse" : "bg-text-muted"}`}
          />

          <div className="flex-1">
            <div className="flex h-8 items-center gap-0.5">
              {recording ? (
                levels.map((level, i) => (
                  <div
                    key={i}
                    className="w-1 rounded-full bg-danger/70 transition-all duration-75"
                    style={{ height: `${Math.max(4, level * 32)}px` }}
                  />
                ))
              ) : error ? (
                <span
                  role="alert"
                  className="flex items-center gap-1.5 text-sm text-danger"
                  data-testid="voice-recorder-error"
                >
                  <AlertCircle size={14} className="shrink-0" />
                  {error}
                </span>
              ) : audioBlob ? (
                <span className="text-sm text-text-secondary">
                  Voice message ready
                </span>
              ) : null}
            </div>
          </div>

          {/* Duration */}
          <span className="text-sm font-mono text-text-primary tabular-nums">
            {formatDuration(duration)}
          </span>

          {/* Controls */}
          <div className="flex items-center gap-1">
            {recording ? (
              <button
                type="button"
                onClick={stopRecording}
                className="p-2 rounded-lg bg-danger text-white hover:bg-danger/90 transition-colors"
                title="Stop recording"
                aria-label="Stop recording"
              >
                <Square size={14} />
              </button>
            ) : audioBlob ? (
              <>
                <button
                  type="button"
                  onClick={handleSend}
                  disabled={sending}
                  className="p-2 rounded-lg bg-accent text-white hover:bg-accent-hover transition-colors disabled:opacity-50"
                  title={error ? "Retry voice message" : "Send voice message"}
                  aria-label={
                    error ? "Retry voice message" : "Send voice message"
                  }
                >
                  {sending ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Send size={14} />
                  )}
                </button>
              </>
            ) : error ? (
              <button
                type="button"
                onClick={() => void startRecording()}
                className="inline-flex items-center gap-1 rounded-lg bg-accent px-2 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-accent-hover"
                title="Retry voice recording"
                aria-label="Retry voice recording"
              >
                <RefreshCw size={12} />
                Retry
              </button>
            ) : null}
            <button
              type="button"
              onClick={onCancel}
              className="p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-bg-hover transition-colors"
              title="Cancel"
              aria-label="Cancel voice recording"
            >
              <X size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Export a simple button to trigger recording
export function VoiceRecordButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="p-1.5 rounded-lg text-text-muted hover:text-accent hover:bg-bg-hover transition-colors"
      title="Record voice message"
      aria-label="Record voice message"
    >
      <Mic size={16} />
    </button>
  );
}
