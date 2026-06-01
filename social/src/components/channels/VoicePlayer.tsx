"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  AlertCircle,
  ChevronDown,
  Loader2,
  Pause,
  Play,
  RefreshCw,
} from "lucide-react";

interface VoicePlayerProps {
  url: string;
  duration?: number;
  transcription?: string;
  transcriptionStatus?: "pending" | "complete" | "failed";
  transcriptionError?: string;
  isAgent?: boolean;
  onRetryTranscription?: () => void;
}

const WAVEFORM_BARS = 40;

function generateWaveform(seedText: string) {
  let seed = 0;
  for (let i = 0; i < seedText.length; i += 1) {
    seed = (seed * 31 + seedText.charCodeAt(i)) >>> 0;
  }

  return Array.from({ length: WAVEFORM_BARS }, (_, index) => {
    seed = (seed * 1664525 + 1013904223 + index) >>> 0;
    return 0.25 + (seed / 0xffffffff) * 0.7;
  });
}

function formatTime(seconds: number) {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(seconds, 0) : 0;
  const m = Math.floor(safeSeconds / 60);
  const sec = Math.floor(safeSeconds % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export default function VoicePlayer({
  url,
  duration,
  transcription,
  transcriptionStatus,
  transcriptionError,
  isAgent,
  onRetryTranscription,
}: VoicePlayerProps) {
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [totalDuration, setTotalDuration] = useState(duration || 0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [showTranscription, setShowTranscription] = useState(false);
  const [loading, setLoading] = useState(true);
  const [playBusy, setPlayBusy] = useState(false);
  const [playError, setPlayError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [waveformData, setWaveformData] = useState<number[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>();
  const playbackRateRef = useRef(playbackRate);

  useEffect(() => {
    playbackRateRef.current = playbackRate;
  }, [playbackRate]);

  useEffect(() => {
    setPlaying(false);
    setCurrentTime(0);
    setTotalDuration(duration || 0);
    setLoading(true);
    setPlayBusy(false);
    setPlayError(null);
    setWaveformData(generateWaveform(url));

    const audio = new Audio(url);
    audio.crossOrigin = "anonymous";
    audio.preload = "metadata";
    audio.playbackRate = playbackRateRef.current;
    audioRef.current = audio;

    const handleLoadedMetadata = () => {
      if (audio.duration && isFinite(audio.duration)) {
        setTotalDuration(Math.round(audio.duration));
      }
      setLoading(false);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    const handleEnded = () => {
      setPlaying(false);
      setCurrentTime(0);
      audio.currentTime = 0;
    };

    const handleCanPlay = () => {
      setLoading(false);
      setPlayError(null);
    };

    const handlePause = () => setPlaying(false);

    const handlePlay = () => {
      setLoading(false);
      setPlaying(true);
    };

    const handleError = () => {
      setLoading(false);
      setPlayBusy(false);
      setPlaying(false);
      setPlayError("Voice audio could not load.");
    };

    audio.addEventListener("loadedmetadata", handleLoadedMetadata);
    audio.addEventListener("durationchange", handleLoadedMetadata);
    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("canplay", handleCanPlay);
    audio.addEventListener("pause", handlePause);
    audio.addEventListener("play", handlePlay);
    audio.addEventListener("error", handleError);
    audio.load();

    return () => {
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
      audio.removeEventListener("durationchange", handleLoadedMetadata);
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("canplay", handleCanPlay);
      audio.removeEventListener("pause", handlePause);
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("error", handleError);
      audio.pause();
      audio.src = "";
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [duration, reloadKey, url]);

  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || waveformData.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, w, h);

    const barWidth = w / waveformData.length;
    const progress = totalDuration > 0 ? currentTime / totalDuration : 0;

    waveformData.forEach((val, i) => {
      const x = i * barWidth;
      const barH = val * h * 0.8;
      const y = (h - barH) / 2;

      const isPlayed = i / waveformData.length <= progress;
      ctx.fillStyle = isPlayed
        ? isAgent
          ? "#00D4AA"
          : "#FF6B35"
        : isAgent
          ? "rgba(0,212,170,0.25)"
          : "rgba(255,107,53,0.25)";
      ctx.beginPath();
      ctx.roundRect(x + 0.5, y, barWidth - 1, barH, 1);
      ctx.fill();
    });
  }, [waveformData, currentTime, totalDuration, isAgent]);

  useEffect(() => {
    drawWaveform();
  }, [drawWaveform]);

  useEffect(() => {
    if (!playing) return;

    const update = () => {
      if (audioRef.current) {
        setCurrentTime(audioRef.current.currentTime);
      }
      animRef.current = requestAnimationFrame(update);
    };
    update();

    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [playing]);

  const togglePlay = async () => {
    if (!audioRef.current) return;
    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
    } else {
      setPlayBusy(true);
      setPlayError(null);
      try {
        audioRef.current.playbackRate = playbackRate;
        await audioRef.current.play();
        setPlaying(true);
      } catch {
        setPlaying(false);
        setPlayError("Voice audio could not play.");
      } finally {
        setPlayBusy(false);
      }
    }
  };

  const cycleSpeed = () => {
    const speeds = [1, 1.5, 2];
    const next = speeds[(speeds.indexOf(playbackRate) + 1) % speeds.length];
    setPlaybackRate(next);
    if (audioRef.current) audioRef.current.playbackRate = next;
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioRef.current || !totalDuration) return;
    const nextTime = Number(e.target.value);
    audioRef.current.currentTime = Math.min(
      Math.max(nextTime, 0),
      totalDuration,
    );
    setCurrentTime(audioRef.current.currentTime);
  };

  const retryAudio = () => {
    setReloadKey((current) => current + 1);
  };

  const accentColor = isAgent ? "text-teal" : "text-accent";
  const accentBg = isAgent ? "bg-teal" : "bg-accent";
  const canSeek = totalDuration > 0 && !playError;
  const transcriptionId = `voice-transcription-${url.replace(/[^a-z0-9]/gi, "-").slice(0, 48)}`;
  const hasTranscription = Boolean(transcription);
  const transcribing = transcriptionStatus === "pending";
  const transcriptionFailed = transcriptionStatus === "failed";
  const noTranscript = transcriptionStatus === "complete" && !hasTranscription;

  return (
    <div data-testid="voice-player" className="max-w-xs">
      <div className="flex items-center gap-2">
        {/* Play/Pause */}
        <button
          type="button"
          onClick={() => void togglePlay()}
          disabled={Boolean(playError) || playBusy}
          className={`w-8 h-8 rounded-full ${accentBg} text-white flex items-center justify-center shrink-0 hover:opacity-90 transition-opacity disabled:cursor-not-allowed disabled:opacity-60`}
          title={
            playBusy
              ? "Loading voice message"
              : playing
                ? "Pause voice message"
                : "Play voice message"
          }
          aria-label={
            playBusy
              ? "Loading voice message"
              : playing
                ? "Pause voice message"
                : "Play voice message"
          }
        >
          {playBusy ? (
            <Loader2 size={14} className="animate-spin" />
          ) : playing ? (
            <Pause size={14} />
          ) : (
            <Play size={14} className="ml-0.5" />
          )}
        </button>

        {/* Waveform */}
        <div className="flex-1">
          <div className="relative h-8">
            <canvas ref={canvasRef} aria-hidden="true" className="h-8 w-full" />
            <input
              data-testid="voice-seek-slider"
              type="range"
              min="0"
              max={Math.max(totalDuration, 1)}
              step="0.1"
              value={Math.min(currentTime, totalDuration || 0)}
              disabled={!canSeek}
              onChange={handleSeek}
              aria-label="Seek voice message"
              className="absolute inset-0 h-8 w-full cursor-pointer opacity-0 disabled:cursor-not-allowed"
            />
          </div>
          <div className="flex items-center justify-between mt-0.5">
            <span className="text-2xs text-text-muted tabular-nums">
              {loading ? "Loading" : formatTime(currentTime)} /{" "}
              {formatTime(totalDuration)}
            </span>
            <button
              type="button"
              onClick={cycleSpeed}
              disabled={Boolean(playError)}
              className={`text-2xs font-mono font-medium ${accentColor} hover:opacity-80 disabled:cursor-not-allowed disabled:opacity-50`}
              title="Change playback speed"
              aria-label={`Change playback speed, currently ${playbackRate}x`}
            >
              {playbackRate}x
            </button>
          </div>
        </div>
      </div>

      {/* Transcription */}
      {hasTranscription && (
        <div className="mt-1">
          <button
            type="button"
            onClick={() => setShowTranscription(!showTranscription)}
            className="flex items-center gap-1 text-2xs text-text-muted hover:text-text-secondary transition-colors"
            title={
              showTranscription ? "Hide transcription" : "Show transcription"
            }
            aria-label={
              showTranscription ? "Hide transcription" : "Show transcription"
            }
            aria-controls={transcriptionId}
            aria-expanded={showTranscription}
          >
            <ChevronDown
              size={10}
              className={`transition-transform ${showTranscription ? "rotate-180" : ""}`}
            />
            Transcription
          </button>
          {showTranscription && (
            <p
              id={transcriptionId}
              className="mt-1 text-xs text-text-secondary italic bg-bg-elevated rounded-lg px-2.5 py-1.5 border border-border"
            >
              {transcription}
            </p>
          )}
        </div>
      )}

      {!hasTranscription &&
        (transcribing || transcriptionFailed || noTranscript) && (
          <div
            data-testid="voice-transcription-status"
            role={transcriptionFailed ? "alert" : "status"}
            className={`mt-2 flex items-start gap-2 rounded-lg border px-2.5 py-2 text-xs ${
              transcriptionFailed
                ? "border-red-300 bg-red-50 text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
                : "border-border bg-bg-elevated text-text-secondary"
            }`}
          >
            {transcribing ? (
              <Loader2 size={13} className="mt-0.5 shrink-0 animate-spin" />
            ) : transcriptionFailed ? (
              <AlertCircle size={13} className="mt-0.5 shrink-0" />
            ) : (
              <AlertCircle
                size={13}
                className="mt-0.5 shrink-0 text-text-muted"
              />
            )}
            <div className="min-w-0 flex-1">
              {transcribing
                ? "Transcribing voice message..."
                : transcriptionFailed
                  ? transcriptionError || "Voice transcription failed."
                  : "No transcript was generated."}
            </div>
            {transcriptionFailed && onRetryTranscription && (
              <button
                type="button"
                onClick={onRetryTranscription}
                className="inline-flex shrink-0 items-center gap-1 rounded border border-red-300 bg-white px-1.5 py-0.5 text-2xs font-medium hover:bg-red-50 dark:border-red-900/70 dark:bg-red-950/40 dark:hover:bg-red-900/30"
                aria-label="Retry voice transcription"
              >
                <RefreshCw size={10} />
                Retry
              </button>
            )}
          </div>
        )}

      {playError && (
        <div
          data-testid="voice-player-error"
          className="mt-2 flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 px-2.5 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
        >
          <AlertCircle size={13} className="mt-0.5 shrink-0" />
          <div className="min-w-0 flex-1">{playError}</div>
          <button
            type="button"
            onClick={retryAudio}
            className="inline-flex shrink-0 items-center gap-1 rounded border border-red-300 bg-white px-1.5 py-0.5 text-2xs font-medium hover:bg-red-50 dark:border-red-900/70 dark:bg-red-950/40 dark:hover:bg-red-900/30"
            aria-label="Retry voice audio"
          >
            <RefreshCw size={10} />
            Retry
          </button>
        </div>
      )}
    </div>
  );
}
