"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Mic, Square, X, Send, Loader2 } from "lucide-react";

interface VoiceRecorderProps {
  onSend: (audioBlob: Blob, duration: number) => Promise<void>;
  onCancel: () => void;
}

export default function VoiceRecorder({ onSend, onCancel }: VoiceRecorderProps) {
  const [recording, setRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [sending, setSending] = useState(false);
  const [levels, setLevels] = useState<number[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval>>();
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number>();
  const streamRef = useRef<MediaStream | null>(null);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Set up audio analyser for waveform
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 64;
      source.connect(analyser);
      analyserRef.current = analyser;

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
          ? "audio/webm;codecs=opus"
          : "audio/webm",
      });

      chunksRef.current = [];
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setAudioBlob(blob);
        stream.getTracks().forEach((t) => t.stop());
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
      console.error("Microphone access denied:", err);
      onCancel();
    }
  }, [onCancel]);

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
    await onSend(audioBlob, duration);
    setSending(false);
  };

  const formatDuration = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div className="px-4 pb-4 pt-2">
      <div className="bg-bg-surface border border-danger/30 rounded-xl overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-3">
          {/* Recording indicator */}
          <div className={`w-3 h-3 rounded-full ${recording ? "bg-danger animate-pulse" : "bg-text-muted"}`} />

          {/* Waveform visualization */}
          <div className="flex-1 flex items-center gap-0.5 h-8">
            {recording ? (
              levels.map((level, i) => (
                <div
                  key={i}
                  className="w-1 bg-danger/70 rounded-full transition-all duration-75"
                  style={{ height: `${Math.max(4, level * 32)}px` }}
                />
              ))
            ) : audioBlob ? (
              <span className="text-sm text-text-secondary">Voice message ready</span>
            ) : null}
          </div>

          {/* Duration */}
          <span className="text-sm font-mono text-text-primary tabular-nums">
            {formatDuration(duration)}
          </span>

          {/* Controls */}
          <div className="flex items-center gap-1">
            {recording ? (
              <button
                onClick={stopRecording}
                className="p-2 rounded-lg bg-danger text-white hover:bg-danger/90 transition-colors"
                title="Stop recording"
              >
                <Square size={14} />
              </button>
            ) : audioBlob ? (
              <>
                <button
                  onClick={handleSend}
                  disabled={sending}
                  className="p-2 rounded-lg bg-accent text-white hover:bg-accent-hover transition-colors disabled:opacity-50"
                  title="Send voice message"
                >
                  {sending ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                </button>
              </>
            ) : null}
            <button
              onClick={onCancel}
              className="p-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-bg-hover transition-colors"
              title="Cancel"
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
    >
      <Mic size={16} />
    </button>
  );
}
