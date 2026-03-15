"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Play, Pause, ChevronDown } from "lucide-react";

interface VoicePlayerProps {
  url: string;
  duration?: number;
  transcription?: string;
  isAgent?: boolean;
}

export default function VoicePlayer({ url, duration, transcription, isAgent }: VoicePlayerProps) {
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [totalDuration, setTotalDuration] = useState(duration || 0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [showTranscription, setShowTranscription] = useState(false);
  const [waveformData, setWaveformData] = useState<number[]>([]);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>();

  // Generate waveform from audio
  useEffect(() => {
    const audio = new Audio(url);
    audio.crossOrigin = "anonymous";
    audioRef.current = audio;

    audio.addEventListener("loadedmetadata", () => {
      if (audio.duration && isFinite(audio.duration)) {
        setTotalDuration(Math.round(audio.duration));
      }
    });

    audio.addEventListener("ended", () => {
      setPlaying(false);
      setCurrentTime(0);
    });

    // Generate fake waveform data (since decoding cross-origin audio is tricky)
    const bars = 40;
    const data = Array.from({ length: bars }, () => 0.2 + Math.random() * 0.8);
    setWaveformData(data);

    return () => {
      audio.pause();
      audio.src = "";
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [url]);

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
        ? isAgent ? "#00D4AA" : "#FF6B35"
        : isAgent ? "rgba(0,212,170,0.25)" : "rgba(255,107,53,0.25)";
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

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (playing) {
      audioRef.current.pause();
      setPlaying(false);
    } else {
      audioRef.current.playbackRate = playbackRate;
      audioRef.current.play();
      setPlaying(true);
    }
  };

  const cycleSpeed = () => {
    const speeds = [1, 1.5, 2];
    const next = speeds[(speeds.indexOf(playbackRate) + 1) % speeds.length];
    setPlaybackRate(next);
    if (audioRef.current) audioRef.current.playbackRate = next;
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !audioRef.current || !totalDuration) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const progress = x / rect.width;
    audioRef.current.currentTime = progress * totalDuration;
    setCurrentTime(audioRef.current.currentTime);
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  const accentColor = isAgent ? "text-teal" : "text-accent";
  const accentBg = isAgent ? "bg-teal" : "bg-accent";

  return (
    <div className="max-w-xs">
      <div className="flex items-center gap-2">
        {/* Play/Pause */}
        <button
          onClick={togglePlay}
          className={`w-8 h-8 rounded-full ${accentBg} text-white flex items-center justify-center shrink-0 hover:opacity-90 transition-opacity`}
        >
          {playing ? <Pause size={14} /> : <Play size={14} className="ml-0.5" />}
        </button>

        {/* Waveform */}
        <div className="flex-1">
          <canvas
            ref={canvasRef}
            className="w-full h-8 cursor-pointer"
            onClick={handleCanvasClick}
          />
          <div className="flex items-center justify-between mt-0.5">
            <span className="text-2xs text-text-muted tabular-nums">
              {formatTime(playing ? currentTime : 0)} / {formatTime(totalDuration)}
            </span>
            <button
              onClick={cycleSpeed}
              className={`text-2xs font-mono font-medium ${accentColor} hover:opacity-80`}
            >
              {playbackRate}x
            </button>
          </div>
        </div>
      </div>

      {/* Transcription */}
      {transcription && (
        <div className="mt-1">
          <button
            onClick={() => setShowTranscription(!showTranscription)}
            className="flex items-center gap-1 text-2xs text-text-muted hover:text-text-secondary transition-colors"
          >
            <ChevronDown size={10} className={`transition-transform ${showTranscription ? "rotate-180" : ""}`} />
            Transcription
          </button>
          {showTranscription && (
            <p className="mt-1 text-xs text-text-secondary italic bg-bg-elevated rounded-lg px-2.5 py-1.5 border border-border">
              {transcription}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
