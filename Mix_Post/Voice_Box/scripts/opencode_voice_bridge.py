#!/usr/bin/env python3
"""OpenCode-compatible bridge for Voicebox STT/TTS endpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


DEFAULT_BASE_URL = os.getenv("VOICEBOX_API_URL", "http://127.0.0.1:17493")
WAIT_INTERVAL_SECONDS = 3
MCP_PROTOCOL_VERSION = "2024-11-05"


class BridgeError(Exception):
    """Error returned from the backend API."""

    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(json.dumps({"status": status, "detail": detail}))
        self.status = status
        self.detail = detail


def _ensure_base_url(url: str) -> str:
    return url.rstrip("/")


def _to_json(payload: bytes) -> Any:
    if not payload:
        return {}
    return json.loads(payload.decode("utf-8"))


def _read_http_error(exc: urllib_error.HTTPError) -> BridgeError:
    body = exc.read()
    try:
        detail = _to_json(body).get("detail")
    except Exception:
        detail = body.decode("utf-8", errors="replace")
    return BridgeError(status=exc.code, detail=detail)


def _request_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        data = body

    request = urllib_request.Request(url, method=method, data=data, headers=headers)

    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            return _to_json(response.read())
    except urllib_error.HTTPError as exc:
        raise _read_http_error(exc)
    except urllib_error.URLError as exc:
        raise BridgeError(status=0, detail=str(exc))


def _post_form(
    url: str,
    fields: Dict[str, str],
    file_path: Path,
    timeout: int = 30,
) -> Any:
    boundary = uuid.uuid4().hex
    parts: List[bytes] = []

    def add_field(name: str, value: str) -> None:
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        parts.append(f"{value}\r\n".encode("utf-8"))

    for name, value in fields.items():
        add_field(name, value)

    file_data = file_path.read_bytes()
    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode(
            "utf-8"
        )
    )
    parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
    parts.append(file_data)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    body = b"".join(parts)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
        "Accept": "application/json",
    }
    request = urllib_request.Request(url, method="POST", data=body, headers=headers)

    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            return _to_json(response.read())
    except urllib_error.HTTPError as exc:
        raise _read_http_error(exc)
    except urllib_error.URLError as exc:
        raise BridgeError(status=0, detail=str(exc))


def _download_file(url: str, destination: Path, timeout: int = 30) -> Path:
    request = urllib_request.Request(url, method="GET", headers={"Accept": "audio/wav, */*"})
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            destination.write_bytes(response.read())
        return destination
    except urllib_error.HTTPError as exc:
        raise _read_http_error(exc)
    except urllib_error.URLError as exc:
        raise BridgeError(status=0, detail=str(exc))


def _find_profile_by_name(base_url: str, name: str) -> str:
    profiles = _request_json("GET", f"{base_url}/profiles")
    if not isinstance(profiles, list):
        raise BridgeError(status=500, detail="Unexpected profile response format")

    typed_profiles: List[Dict[str, Any]] = []
    for profile in profiles:
        if isinstance(profile, dict):
            if not isinstance(profile.get("id"), str) or not isinstance(profile.get("name"), str):
                continue
            typed_profiles.append(profile)

    exact = [p for p in typed_profiles if p.get("name", "").lower() == name.lower()]
    if exact:
        return exact[0]["id"]

    matches = [p for p in typed_profiles if name.lower() in p.get("name", "").lower()]
    if len(matches) == 1:
        return matches[0]["id"]
    if len(matches) > 1:
        raise BridgeError(
            status=409,
            detail={
                "error": "multiple_profiles",
                "message": f"Profile name '{name}' matches multiple profiles.",
                "matches": [p.get("name") for p in matches],
            },
        )

    raise BridgeError(
        status=404,
        detail={"error": "profile_not_found", "message": f"Profile '{name}' not found."},
    )


def _request_with_retry(
    method: str, url: str, payload: Dict[str, Any], wait: bool, timeout: int
) -> Any:
    if not wait:
        return _request_json(method, url, payload=payload)

    deadline = time.time() + timeout

    while True:
        try:
            return _request_json(method=method, url=url, payload=payload)
        except BridgeError as exc:
            if exc.status != 202:
                raise

            detail = exc.detail
            if not isinstance(detail, dict) or not detail.get("downloading"):
                raise

            if time.time() >= deadline:
                raise BridgeError(
                    status=408,
                    detail={
                        "error": "timeout",
                        "message": "Timed out while waiting for required model download.",
                        "detail": detail,
                    },
                )
            time.sleep(WAIT_INTERVAL_SECONDS)


def _post_form_with_retry(
    url: str, fields: Dict[str, str], file_path: Path, wait: bool, timeout: int
) -> Any:
    if not wait:
        return _post_form(url, fields, file_path)

    deadline = time.time() + timeout

    while True:
        try:
            return _post_form(url, fields, file_path)
        except BridgeError as exc:
            if exc.status != 202:
                raise

            detail = exc.detail
            if not isinstance(detail, dict) or not detail.get("downloading"):
                raise

            if time.time() >= deadline:
                raise BridgeError(
                    status=408,
                    detail={
                        "error": "timeout",
                        "message": "Timed out while waiting for Whisper model download.",
                        "detail": detail,
                    },
                )

            time.sleep(WAIT_INTERVAL_SECONDS)


def run_profiles(base_url: str, name_filter: Optional[str]) -> List[Dict[str, Any]]:
    profiles = _request_json("GET", f"{base_url}/profiles")
    if not isinstance(profiles, list):
        raise BridgeError(status=500, detail="Unexpected profile response format")

    filtered: List[Dict[str, Any]] = [
        p
        for p in profiles
        if isinstance(p, dict)
        and (name_filter is None or name_filter.lower() in str(p.get("name", "")).lower())
    ]

    return filtered


def run_stt(
    base_url: str, audio_file: Path, language: Optional[str], wait: bool, timeout: int
) -> Dict[str, Any]:
    if not audio_file.exists():
        raise BridgeError(
            status=404, detail={"error": "file_not_found", "message": str(audio_file)}
        )

    payload: Dict[str, str] = {}
    if language:
        payload["language"] = language

    return _post_form_with_retry(
        f"{base_url}/transcribe",
        payload,
        audio_file,
        wait=wait,
        timeout=timeout,
    )


def run_tts(
    base_url: str,
    text: str,
    profile_id: Optional[str],
    profile_name: Optional[str],
    language: str,
    seed: Optional[int],
    model_size: Optional[str],
    instruct: Optional[str],
    out_path: Optional[Path],
    wait: bool,
    timeout: int,
) -> Dict[str, Any]:
    if not profile_id:
        if not profile_name:
            raise BridgeError(
                status=400,
                detail={
                    "error": "missing_profile",
                    "message": "Provide --profile-id or --profile-name.",
                },
            )
        profile_id = _find_profile_by_name(base_url, profile_name)

    payload: Dict[str, Any] = {
        "profile_id": profile_id,
        "text": text,
        "language": language,
    }
    if seed is not None:
        payload["seed"] = seed
    if model_size is not None:
        payload["model_size"] = model_size
    if instruct:
        payload["instruct"] = instruct

    generation = _request_with_retry(
        "POST",
        f"{base_url}/generate",
        payload,
        wait=wait,
        timeout=timeout,
    )

    if not isinstance(generation, dict):
        raise BridgeError(
            status=500,
            detail={"error": "invalid_response", "message": "Unexpected /generate response format"},
        )

    generation_id = generation.get("id")
    if not generation_id:
        raise BridgeError(
            status=500,
            detail={"error": "invalid_response", "message": "No generation id returned"},
        )

    audio_path = out_path or Path.cwd() / f"voicebox_{datetime.utcnow():%Y%m%d_%H%M%S}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    _download_file(f"{base_url}/audio/{generation_id}", audio_path)

    output = {
        "id": generation_id,
        "text": generation.get("text"),
        "duration": generation.get("duration"),
        "profile_id": generation.get("profile_id"),
        "audio_id": generation_id,
        "audio_path": str(audio_path),
        "audio_url": f"{base_url}/audio/{generation_id}",
    }
    return output


def _mcp_tool_specs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "profiles",
            "description": "List available Voicebox voice profiles.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Optional case-insensitive name substring filter.",
                    }
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "stt",
            "description": "Transcribe an audio file to text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Path to an audio file.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional language hint for STT (for example en or zh).",
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "Wait if model download is in progress.",
                        "default": False,
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Max seconds to wait when wait is set.",
                        "minimum": 1,
                        "default": 240,
                    },
                },
                "required": ["audio_path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "tts",
            "description": "Generate speech from text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to synthesize.",
                    },
                    "profile_id": {
                        "type": "string",
                        "description": "Voice profile UUID.",
                    },
                    "profile_name": {
                        "type": "string",
                        "description": "Voice profile name when id is unknown.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code for generation.",
                        "default": "en",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Optional generation seed.",
                    },
                    "model_size": {
                        "type": "string",
                        "description": "Model size to use.",
                        "enum": ["1.7B", "0.6B"],
                        "default": "1.7B",
                    },
                    "instruct": {
                        "type": "string",
                        "description": "Optional generation instruction.",
                    },
                    "out": {
                        "type": "string",
                        "description": "Optional output WAV path.",
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "Wait if model download is in progress.",
                        "default": False,
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Max seconds to wait when wait is set.",
                        "minimum": 1,
                        "default": 240,
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    ]


def _mcp_send(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload), file=sys.stdout)
    sys.stdout.flush()


def _mcp_ok(request_id: Any, result: Dict[str, Any]) -> None:
    _mcp_send({"jsonrpc": "2.0", "id": request_id, "result": result})


def _mcp_err(request_id: Any, code: int, message: str, detail: Any = None) -> None:
    error: Dict[str, Any] = {"code": code, "message": message}
    if detail is not None:
        error["data"] = detail
    _mcp_send({"jsonrpc": "2.0", "id": request_id, "error": error})


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"1", "true", "yes", "on"}
    return default


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _handle_tool_call(
    base_url: str,
    tool: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    if tool == "profiles":
        name = arguments.get("name")
        if name is not None and not isinstance(name, str):
            raise BridgeError(
                status=400, detail={"error": "invalid_name", "message": "name must be a string."}
            )
        return {"profiles": run_profiles(base_url=base_url, name_filter=name)}

    if tool == "stt":
        audio_path = arguments.get("audio_path")
        if not isinstance(audio_path, str):
            raise BridgeError(
                status=400,
                detail={
                    "error": "invalid_audio_path",
                    "message": "audio_path must be provided as a string.",
                },
            )
        language = arguments.get("language")
        if language is not None and not isinstance(language, str):
            raise BridgeError(
                status=400,
                detail={"error": "invalid_language", "message": "language must be a string."},
            )

        return run_stt(
            base_url=base_url,
            audio_file=Path(audio_path),
            language=language,
            wait=_coerce_bool(arguments.get("wait")),
            timeout=_coerce_int(arguments.get("timeout"), 240),
        )

    if tool == "tts":
        text = arguments.get("text")
        if not isinstance(text, str) or not text.strip():
            raise BridgeError(
                status=400,
                detail={"error": "invalid_text", "message": "text must be a non-empty string."},
            )

        language = arguments.get("language")
        if language is not None and not isinstance(language, str):
            raise BridgeError(
                status=400,
                detail={"error": "invalid_language", "message": "language must be a string."},
            )

        model_size = arguments.get("model_size")
        if model_size is not None and model_size not in {"1.7B", "0.6B"}:
            raise BridgeError(
                status=400,
                detail={
                    "error": "invalid_model_size",
                    "message": "model_size must be one of 1.7B or 0.6B.",
                },
            )

        seed = arguments.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except Exception:
                raise BridgeError(
                    status=400,
                    detail={"error": "invalid_seed", "message": "seed must be an integer."},
                )

        profile_id = arguments.get("profile_id")
        if profile_id is not None and not isinstance(profile_id, str):
            raise BridgeError(
                status=400,
                detail={"error": "invalid_profile_id", "message": "profile_id must be a string."},
            )

        profile_name = arguments.get("profile_name")
        if profile_name is not None and not isinstance(profile_name, str):
            raise BridgeError(
                status=400,
                detail={
                    "error": "invalid_profile_name",
                    "message": "profile_name must be a string.",
                },
            )

        return run_tts(
            base_url=base_url,
            text=text,
            profile_id=profile_id,
            profile_name=profile_name,
            language=language or "en",
            seed=seed,
            model_size=model_size,
            instruct=arguments.get("instruct"),
            out_path=Path(arguments["out"]).resolve()
            if isinstance(arguments.get("out"), str)
            else None,
            wait=_coerce_bool(arguments.get("wait")),
            timeout=_coerce_int(arguments.get("timeout"), 240),
        )

    raise BridgeError(
        status=404, detail={"error": "unknown_tool", "message": f"Unknown tool '{tool}'."}
    )


def _run_mcp_server(base_url: str) -> None:
    for raw in sys.stdin:
        if not raw.strip():
            continue

        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            _mcp_send(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            )
            continue

        if not isinstance(message, dict):
            _mcp_send(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid request"},
                }
            )
            continue

        method = message.get("method")
        request_id = message.get("id")

        if method == "initialize":
            _mcp_ok(
                request_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "voicebox-opencode-bridge", "version": "1.0.0"},
                },
            )
            continue

        if method == "tools/list":
            _mcp_ok(request_id, {"tools": _mcp_tool_specs()})
            continue

        if method == "tools/call":
            params = message.get("params")
            if not isinstance(params, dict):
                _mcp_err(request_id, -32602, "Invalid params", "params must be an object")
                continue

            name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(name, str):
                _mcp_err(
                    request_id,
                    -32602,
                    "Invalid params",
                    "tool call requires name",
                )
                continue

            if arguments is None:
                arguments = {}

            if not isinstance(arguments, dict):
                _mcp_err(request_id, -32602, "Invalid params", "arguments must be an object")
                continue

            try:
                tool_result = _handle_tool_call(base_url=base_url, tool=name, arguments=arguments)
            except BridgeError as exc:
                _mcp_err(request_id, -32000, "BridgeError", exc.detail)
                continue

            _mcp_ok(
                request_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(tool_result),
                        }
                    ],
                    "structuredContent": tool_result,
                },
            )
            continue

        if method == "notifications/initialized" or method == "shutdown":
            # No response expected for notifications.
            continue

        _mcp_err(request_id, -32601, "Method not found", f"Unknown method '{method}'")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge Voicebox APIs for OpenCode tools")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL for voicebox API")
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Run as MCP stdio server for local integration",
    )
    parser.add_argument(
        "--wait", action="store_true", help="Wait and retry on model download responses"
    )
    parser.add_argument(
        "--timeout", type=int, default=240, help="Max seconds to wait when --wait is set"
    )

    subparsers = parser.add_subparsers(dest="command")

    stt_parser = subparsers.add_parser("stt", help="Transcribe audio to text")
    stt_parser.add_argument("file", help="Audio file to transcribe")
    stt_parser.add_argument("--language", help="Language hint (en or zh)")

    tts_parser = subparsers.add_parser("tts", help="Synthesize text to speech")
    tts_parser.add_argument("text", nargs="?", help="Text to synthesize")
    tts_parser.add_argument("--profile-id", help="Voice profile UUID")
    tts_parser.add_argument("--profile-name", help="Voice profile name")
    tts_parser.add_argument("--language", default="en", help="Language code")
    tts_parser.add_argument("--seed", type=int, help="Generation seed")
    tts_parser.add_argument(
        "--model-size", default="1.7B", choices=["1.7B", "0.6B"], help="Model size"
    )
    tts_parser.add_argument("--instruct", help="Optional generation instruction")
    tts_parser.add_argument("--out", help="Output WAV path")

    profiles_parser = subparsers.add_parser("profiles", help="List voice profiles")
    profiles_parser.add_argument("--name", help="Filter profiles by name")

    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()

    base_url = _ensure_base_url(args.base_url)

    if args.mcp:
        _run_mcp_server(base_url=base_url)
        return

    if args.command is None:
        parser.error("the following arguments are required: command")

    if args.command == "stt":
        output = run_stt(
            base_url=base_url,
            audio_file=Path(args.file),
            language=args.language,
            wait=args.wait,
            timeout=args.timeout,
        )
        print(json.dumps(output))
        return

    if args.command == "profiles":
        print(json.dumps(run_profiles(base_url=base_url, name_filter=args.name)))
        return

    text = args.text
    if text is None:
        text = sys.stdin.read().strip()
    if not text:
        raise RuntimeError(
            json.dumps({"error": "missing_text", "message": "No text provided for tts."})
        )

    out_path = Path(args.out).expanduser() if args.out else None
    if out_path is not None:
        out_path = out_path.resolve()

    output = run_tts(
        base_url=base_url,
        text=text,
        profile_id=args.profile_id,
        profile_name=args.profile_name,
        language=args.language,
        seed=args.seed,
        model_size=args.model_size,
        instruct=args.instruct,
        out_path=out_path,
        wait=args.wait,
        timeout=args.timeout,
    )
    print(json.dumps(output))


if __name__ == "__main__":
    try:
        main()
    except BridgeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
