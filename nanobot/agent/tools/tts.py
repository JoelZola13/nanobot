"""Qwen3-TTS tool — generates human-quality speech on Apple Silicon via MPS."""

import asyncio
import hashlib
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from loguru import logger

from nanobot.agent.tools.base import Tool

# Lazy-loaded models (heavy — load on first use, cached after)
_custom_voice_model = None
_voice_design_model = None
_voice_clone_model = None

SPEAKERS = (
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
)

LANGUAGES = (
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
)

_YOUTUBE_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w-]+"
)

_REF_AUDIO_CACHE_DIR = Path.home() / ".nanobot" / "workspace" / "voice_clone_cache"


def _get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _get_custom_voice_model():
    """Load Qwen3-TTS CustomVoice model (lazy, cached)."""
    global _custom_voice_model
    if _custom_voice_model is not None:
        return _custom_voice_model

    from qwen_tts import Qwen3TTSModel

    _custom_voice_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        dtype=torch.float32,
        attn_implementation="sdpa",
        device_map=_get_device(),
    )
    return _custom_voice_model


def _get_voice_design_model():
    """Load Qwen3-TTS VoiceDesign model (lazy, cached)."""
    global _voice_design_model
    if _voice_design_model is not None:
        return _voice_design_model

    from qwen_tts import Qwen3TTSModel

    _voice_design_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        dtype=torch.float32,
        attn_implementation="sdpa",
        device_map=_get_device(),
    )
    return _voice_design_model


def _get_voice_clone_model():
    """Load Qwen3-TTS Base model for voice cloning (lazy, cached)."""
    global _voice_clone_model
    if _voice_clone_model is not None:
        return _voice_clone_model

    from qwen_tts import Qwen3TTSModel

    _voice_clone_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        dtype=torch.float32,
        attn_implementation="sdpa",
        device_map=_get_device(),
    )
    return _voice_clone_model


def _sync_device():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _pad_silence(wavs, sr, duration_sec=0.1):
    """Prepend silence to prevent onset clipping in video players."""
    silence = np.zeros(int(sr * duration_sec), dtype=wavs[0].dtype)
    wavs[0] = np.concatenate([silence, wavs[0]])
    return wavs


def _min_expected_duration(text: str) -> float:
    """Rough minimum expected audio duration based on word count."""
    return max(1.5, len(text.split()) / 4.0)


def _is_youtube_url(url: str) -> bool:
    return bool(_YOUTUBE_RE.match(url.strip()))


def _download_youtube_audio(url: str) -> Path:
    """Download audio from YouTube, trim to ~15s, cache result."""
    _REF_AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Cache by URL hash
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    cached = _REF_AUDIO_CACHE_DIR / f"{url_hash}.wav"
    if cached.exists():
        logger.info(f"Using cached YouTube audio: {cached}")
        return cached

    # Download full audio to temp file
    raw_path = _REF_AUDIO_CACHE_DIR / f"{url_hash}_raw.wav"
    logger.info(f"Downloading YouTube audio: {url}")
    result = subprocess.run(
        [
            "/opt/homebrew/bin/yt-dlp",
            "-x", "--audio-format", "wav",
            "--no-playlist",
            "-o", str(raw_path),
            url,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")

    # Trim to ~15 seconds for voice cloning reference
    data, sr = sf.read(str(raw_path))
    max_samples = int(15 * sr)
    if len(data) > max_samples:
        data = data[:max_samples]
    sf.write(str(cached), data, sr)

    # Clean up raw file
    raw_path.unlink(missing_ok=True)
    logger.info(f"Cached trimmed reference audio: {cached} ({len(data)/sr:.1f}s)")
    return cached


def _is_valid_transcript(text: str) -> bool:
    """Check if a transcript looks like real speech (not music/noise garbage).

    Whisper often hallucinates non-Latin scripts when fed music or noise.
    We check that a reasonable fraction of characters are Latin/common punctuation.
    """
    if len(text.strip()) < 5:
        return False
    latin_chars = sum(1 for c in text if c.isascii() or c in "''""—–")
    ratio = latin_chars / len(text)
    # If less than 40% ASCII, it's probably garbage from music
    return ratio > 0.4


async def _auto_transcribe(audio_path: Path) -> str:
    """Transcribe reference audio via Groq Whisper. Returns empty string on failure."""
    try:
        from nanobot.providers.transcription import GroqTranscriptionProvider
        provider = GroqTranscriptionProvider()
        text = await provider.transcribe(audio_path)
        if text:
            logger.info(f"Auto-transcribed reference audio: {text[:100]}...")
        return text
    except Exception as e:
        logger.warning(f"Auto-transcription failed (will use embedding-only mode): {e}")
        return ""


class QwenTTSTool(Tool):
    """Generate human-quality speech audio using Qwen3-TTS."""

    name = "tts"
    description = (
        "Generate human-quality speech audio using Qwen3-TTS (runs locally on Apple Silicon). "
        "Supports 10 languages. THREE modes:\n"
        "\n"
        "**Mode 1 — Preset speaker** (set 'speaker' param): Use one of 9 preset voices. "
        "Speakers: Vivian (bright Chinese female), Serena (warm Chinese female), Uncle_Fu (deep Chinese male), "
        "Dylan (Beijing dialect male), Eric (Sichuan dialect male), Ryan (dynamic English male), "
        "Aiden (sunny American male), Ono_Anna (playful Japanese female), Sohee (warm Korean female). "
        "Use 'instruct' for style control (emotion, speed, pitch).\n"
        "\n"
        "**Mode 2 — Custom voice design** (set 'voice_description' param instead of 'speaker'): "
        "Describe ANY voice in natural language and the model creates it from scratch. "
        "Examples: 'Deep authoritative male, 50s, British accent, news anchor', "
        "'Young energetic woman, 20s, bright and bubbly, podcast host'. "
        "Include age, gender, accent, vocal quality, and personality for best results.\n"
        "\n"
        "**Mode 3 — Voice clone from YouTube** (set 'ref_audio' param): "
        "Clone a voice from a YouTube URL, direct audio URL, or local file path. "
        "Paste a YouTube link and the tool auto-downloads audio, trims to ~15s, "
        "auto-transcribes via Groq Whisper, and clones the voice. "
        "Optionally provide 'ref_text' with the transcript of the reference audio for better quality. "
        "Falls back to embedding-only mode if transcription unavailable.\n"
        "\n"
        "Output WAV saved to Remotion public/audio for use in video compositions "
        "via <Audio src={staticFile('audio/FILENAME.wav')}/>."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to convert to speech.",
            },
            "output_name": {
                "type": "string",
                "description": "Output filename without extension (e.g. 'narration', 'intro-voiceover')",
            },
            "speaker": {
                "type": "string",
                "description": "Preset speaker voice (Mode 1). Use this OR voice_description OR ref_audio.",
                "enum": list(SPEAKERS),
            },
            "voice_description": {
                "type": "string",
                "description": (
                    "Natural language description of the voice to generate (Mode 2). "
                    "E.g. 'Deep male voice, 40s, calm and authoritative, American accent'. "
                    "Use this OR speaker OR ref_audio."
                ),
            },
            "ref_audio": {
                "type": "string",
                "description": (
                    "Reference audio for voice cloning (Mode 3). "
                    "YouTube URL, direct audio URL, or local file path. "
                    "The tool auto-downloads and trims YouTube audio to ~15s."
                ),
            },
            "ref_text": {
                "type": "string",
                "description": (
                    "Transcript of the reference audio (Mode 3, optional). "
                    "If omitted, auto-transcribed via Groq Whisper. "
                    "Providing accurate transcript improves clone quality."
                ),
            },
            "language": {
                "type": "string",
                "description": "Language of the text (default: English)",
                "enum": list(LANGUAGES),
            },
            "instruct": {
                "type": "string",
                "description": (
                    "Style instruction for how to speak. Works with ALL modes (1, 2, and 3). "
                    "E.g. 'Warm and friendly', 'Fast-paced excited', 'Whispered, intimate', "
                    "'Very excited and energetic'. For voice cloning, this controls the emotion/style "
                    "of the cloned voice output."
                ),
            },
        },
        "required": ["text", "output_name"],
    }

    def __init__(self, audio_dir: Path):
        self._audio_dir = audio_dir

    async def execute(
        self,
        text: str,
        output_name: str,
        speaker: str = "",
        voice_description: str = "",
        ref_audio: str = "",
        ref_text: str = "",
        language: str = "English",
        instruct: str = "",
    ) -> str:
        if not text.strip():
            return "Error: text cannot be empty"

        if language not in LANGUAGES:
            return f"Error: unknown language '{language}'. Choose from: {', '.join(LANGUAGES)}"

        use_voice_clone = bool(ref_audio.strip())
        use_voice_design = bool(voice_description.strip()) and not use_voice_clone

        output_path = self._audio_dir / f"{output_name}.wav"

        try:
            if use_voice_clone:
                wavs, sr, voice_label = await self._handle_voice_clone(
                    text, ref_audio.strip(), ref_text.strip(), language, instruct
                )
            elif use_voice_design:
                wavs, sr = await asyncio.to_thread(
                    self._generate_voice_design, text, voice_description, language, instruct
                )
                voice_label = f"custom ({voice_description[:50]}...)" if len(voice_description) > 50 else f"custom ({voice_description})"
            else:
                speaker = speaker or "Ryan"
                if speaker not in SPEAKERS:
                    return f"Error: unknown speaker '{speaker}'. Choose from: {', '.join(SPEAKERS)}"
                wavs, sr = await asyncio.to_thread(
                    self._generate_custom_voice, text, speaker, language, instruct
                )
                voice_label = speaker
        except Exception as e:
            return f"Error generating speech: {e}"

        if wavs is None or len(wavs) == 0 or len(wavs[0]) == 0:
            return "Error: No audio generated."

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), wavs[0], sr)

        duration_secs = len(wavs[0]) / sr
        return (
            f"Generated speech audio: {output_path.name} ({duration_secs:.1f}s, voice: {voice_label}, lang: {language})\n"
            f"File: {output_path}\n"
            f"Use in Remotion: <Audio src={{staticFile('audio/{output_name}.wav')}}/>"
        )

    async def _handle_voice_clone(
        self, text: str, ref_audio: str, ref_text: str, language: str, instruct: str = ""
    ):
        """Handle Mode 3: voice cloning from YouTube/URL/file."""
        # Step 1: Resolve reference audio to a local file
        if _is_youtube_url(ref_audio):
            ref_path = await asyncio.to_thread(_download_youtube_audio, ref_audio)
            voice_label = f"cloned (YouTube)"
        elif ref_audio.startswith(("http://", "https://")):
            # Direct audio URL — download it
            ref_path = await asyncio.to_thread(self._download_audio_url, ref_audio)
            voice_label = f"cloned (URL)"
        else:
            ref_path = Path(ref_audio)
            if not ref_path.exists():
                raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
            voice_label = f"cloned ({ref_path.name})"

        # Step 2: Auto-transcribe if no ref_text provided
        if not ref_text:
            ref_text = await _auto_transcribe(ref_path)

        # Validate transcript quality — garbage transcriptions (e.g. music → wrong script)
        # cause the model to generate endlessly. Fall back to embedding-only if suspicious.
        if ref_text and not _is_valid_transcript(ref_text):
            logger.warning(f"Transcript looks like garbage (music/noise?), falling back to embedding-only: {ref_text[:60]}...")
            ref_text = ""

        use_embedding_only = not ref_text
        if use_embedding_only:
            logger.info("No transcript available — using embedding-only voice clone mode")
            voice_label += " [embedding-only]"
        else:
            logger.info(f"Using ICL voice clone with transcript: {ref_text[:80]}...")

        # Step 3: Generate cloned speech
        wavs, sr = await asyncio.to_thread(
            self._generate_voice_clone, text, str(ref_path), ref_text, language, use_embedding_only, instruct
        )
        return wavs, sr, voice_label

    @staticmethod
    def _download_audio_url(url: str) -> Path:
        """Download audio from a direct URL."""
        _REF_AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        cached = _REF_AUDIO_CACHE_DIR / f"{url_hash}.wav"
        if cached.exists():
            return cached

        import httpx
        with httpx.Client(timeout=60) as client:
            resp = client.get(url)
            resp.raise_for_status()
            cached.write_bytes(resp.content)

        # Trim to 15s like YouTube
        data, sr = sf.read(str(cached))
        max_samples = int(15 * sr)
        if len(data) > max_samples:
            data = data[:max_samples]
            sf.write(str(cached), data, sr)

        return cached

    @staticmethod
    def _generate_voice_clone(
        text: str, ref_audio_path: str, ref_text: str, language: str,
        use_embedding_only: bool, instruct: str = ""
    ):
        """Voice clone mode with retry for short outputs. Supports instruct for style control."""
        model = _get_voice_clone_model()
        min_dur = _min_expected_duration(text)

        # Build instruct_ids if style instruction provided
        # The Base model's generate() accepts instruct_ids just like CustomVoice
        instruct_ids_list = None
        if instruct:
            instruct_text = model._build_instruct_text(instruct)
            instruct_tok = model._tokenize_texts([instruct_text])[0]
            instruct_ids_list = [instruct_tok]
            logger.info(f"Voice clone with instruct: {instruct}")

        best_wav, best_sr, best_dur = None, None, 0.0

        for _ in range(3):
            # Build the clone prompt manually so we can inject instruct_ids
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text=ref_text if not use_embedding_only else None,
                x_vector_only_mode=use_embedding_only,
            )
            voice_clone_prompt_dict = model._prompt_items_to_voice_clone_prompt(prompt_items)

            # Build input_ids and ref_ids like the library does internally
            input_texts = [model._build_assistant_text(text)]
            input_ids = model._tokenize_texts(input_texts)

            ref_ids = None
            if not use_embedding_only and ref_text:
                ref_tok = model._tokenize_texts([model._build_ref_text(ref_text)])[0]
                ref_ids = [ref_tok]

            gen_kwargs = model._merge_generate_kwargs(max_new_tokens=512)

            generate_kwargs = dict(
                input_ids=input_ids,
                ref_ids=ref_ids,
                voice_clone_prompt=voice_clone_prompt_dict,
                languages=[language],
                non_streaming_mode=False,
                **gen_kwargs,
            )
            if instruct_ids_list:
                generate_kwargs["instruct_ids"] = instruct_ids_list

            talker_codes_list, _ = model.model.generate(**generate_kwargs)

            # Decode audio from talker codes
            codes_for_decode = []
            for i, codes in enumerate(talker_codes_list):
                ref_code = voice_clone_prompt_dict.get("ref_code", [None])[i]
                if ref_code is not None:
                    codes = codes[ref_code.shape[-1]:]
                codes_for_decode.append(codes)

            wavs_all, sr = model.model.speech_tokenizer.decode(
                [{"audio_codes": c} for c in codes_for_decode]
            )
            wavs = [w.cpu().numpy() if hasattr(w, 'numpy') else w for w in wavs_all]
            _sync_device()

            if wavs is None or len(wavs) == 0 or len(wavs[0]) == 0:
                continue

            dur = len(wavs[0]) / sr
            if dur > best_dur:
                best_wav, best_sr, best_dur = wavs, sr, dur
            if dur >= min_dur:
                break

        if best_wav is None:
            return None, 0
        return _pad_silence(best_wav, best_sr), best_sr

    @staticmethod
    def _generate_custom_voice(text: str, speaker: str, language: str, instruct: str):
        """Preset speaker mode with retry for short outputs."""
        model = _get_custom_voice_model()
        min_dur = _min_expected_duration(text)

        best_wav, best_sr, best_dur = None, None, 0.0

        for _ in range(3):
            wavs, sr = model.generate_custom_voice(
                text=text, language=language, speaker=speaker, instruct=instruct or "",
            )
            _sync_device()

            if wavs is None or len(wavs) == 0 or len(wavs[0]) == 0:
                continue

            dur = len(wavs[0]) / sr
            if dur > best_dur:
                best_wav, best_sr, best_dur = wavs, sr, dur
            if dur >= min_dur:
                break

        if best_wav is None:
            return None, 0
        return _pad_silence(best_wav, best_sr), best_sr

    @staticmethod
    def _generate_voice_design(text: str, voice_description: str, language: str, instruct: str):
        """Custom voice design mode with retry for short outputs."""
        model = _get_voice_design_model()
        min_dur = _min_expected_duration(text)

        # Combine voice description with style instruction
        full_instruct = voice_description
        if instruct:
            full_instruct = f"{voice_description}. {instruct}"

        best_wav, best_sr, best_dur = None, None, 0.0

        for _ in range(3):
            wavs, sr = model.generate_voice_design(
                text=text, language=language, instruct=full_instruct,
            )
            _sync_device()

            if wavs is None or len(wavs) == 0 or len(wavs[0]) == 0:
                continue

            dur = len(wavs[0]) / sr
            if dur > best_dur:
                best_wav, best_sr, best_dur = wavs, sr, dur
            if dur >= min_dur:
                break

        if best_wav is None:
            return None, 0
        return _pad_silence(best_wav, best_sr), best_sr
