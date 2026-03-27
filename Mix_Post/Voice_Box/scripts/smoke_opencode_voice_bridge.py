#!/usr/bin/env python3
"""Lightweight smoke test for the OpenCode Voicebox bridge."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import math
import struct
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import subprocess
import tempfile
import wave


DEFAULT_BASE_URL = "http://127.0.0.1:17493"
ROOT_DIR = Path(__file__).resolve().parents[1]
BRIDGE_SCRIPT = ROOT_DIR / "scripts" / "opencode_voice_bridge.py"


class _MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: ARG002, A002
        return

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length) if length else b"{}"
        return json.loads(payload.decode("utf-8") or "{}")

    def _write_json(self, status_code: int, data: Any) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_binary(self, status_code: int, data: bytes) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/profiles":
            payload = [
                {
                    "id": "11111111-1111-4111-8111-111111111111",
                    "name": "Demo Voice",
                    "language": "en",
                    "description": "Mock profile",
                    "avatar_path": None,
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                }
            ]
            self._write_json(200, payload)
            return

        if path.startswith("/audio/") and path.count("/") == 2:
            audio = b"RIFF\x00\x00\x00\x00WAVE"
            self._write_binary(200, audio)
            return

        self._write_json(404, {"detail": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path

        if path == "/transcribe":
            payload = {"text": "mock transcription", "duration": 1.23}
            self._write_json(200, payload)
            return

        if path == "/generate":
            request = self._read_json()
            generation_id = str(uuid.uuid4())
            payload = {
                "id": generation_id,
                "text": request.get("text", ""),
                "language": request.get("language", "en"),
                "duration": 1.0,
                "profile_id": request.get("profile_id", ""),
                "seed": request.get("seed"),
                "instruct": request.get("instruct"),
                "created_at": "2026-01-01T00:00:00Z",
            }
            self._write_json(200, payload)
            return

        self._write_json(404, {"detail": "Not found"})


def _run_bridge(
    base_url: str,
    args: List[str],
    timeout: int = 5,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(BRIDGE_SCRIPT),
        "--base-url",
        base_url,
        "--timeout",
        str(timeout),
    ] + args

    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT_DIR),
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"bridge failed with code {process.returncode}: {process.stderr.strip()}"
        )

    try:
        return json.loads(process.stdout.strip() or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON output: {process.stdout.strip()}") from exc


def _read_profiles(base_url: str) -> List[Dict[str, Any]]:
    result = _run_bridge(base_url, ["profiles"])
    if not isinstance(result, list):
        raise RuntimeError(f"expected list from profiles, got: {result}")

    typed: List[Dict[str, Any]] = [
        item for item in result if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]

    if not typed:
        raise RuntimeError("no profiles returned")
    return typed


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _run_mock_server() -> tuple[HTTPServer, int]:
    server = HTTPServer(("127.0.0.1", 0), _MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_port


def _run_bridge_mcp(base_url: str, audio_input: Path, profile_id: str, mock_backend: bool) -> None:
    command = [
        sys.executable,
        str(BRIDGE_SCRIPT),
        "--base-url",
        base_url,
        "--mcp",
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdin = process.stdin
        stdout = process.stdout
        if stdin is None or stdout is None:
            raise RuntimeError("MCP process streams unavailable")

        def send(payload: Dict[str, Any]) -> None:
            stdin.write(json.dumps(payload))
            stdin.write("\n")
            stdin.flush()

        def recv() -> Dict[str, Any]:
            raw = stdout.readline()
            if not raw:
                raise RuntimeError("No response from MCP bridge")
            return json.loads(raw)

        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "smoke-test", "version": "1.0.0"},
                },
            }
        )

        init = recv()
        if "error" in init:
            raise RuntimeError(f"MCP initialize failed: {init}")

        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        listed = recv()
        if "error" in listed:
            raise RuntimeError(f"MCP tools/list failed: {listed}")
        tool_names = [
            tool.get("name")
            for tool in listed.get("result", {}).get("tools", [])
            if isinstance(tool, dict)
        ]
        for required in ("profiles", "stt", "tts"):
            if required not in tool_names:
                raise RuntimeError(f"MCP missing tool: {required}")

        send(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "stt", "arguments": {"audio_path": str(audio_input)}},
            }
        )
        stt_result = recv()
        if "error" in stt_result:
            raise RuntimeError(f"MCP stt failed: {stt_result}")
        stt_payload = stt_result.get("result", {}).get("structuredContent", {})
        if not isinstance(stt_payload, dict) or "text" not in stt_payload:
            raise RuntimeError(f"Unexpected MCP stt output: {stt_payload}")
        if mock_backend and stt_payload.get("text") != "mock transcription":
            raise RuntimeError(f"Unexpected mock MCP stt output: {stt_payload}")

        output_path = audio_input.with_name("mcp-output.wav")
        send(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "tts",
                    "arguments": {
                        "text": "Hello from MCP smoke",
                        "profile_id": profile_id,
                        "out": str(output_path),
                    },
                },
            }
        )
        tts_result = recv()
        if "error" in tts_result:
            raise RuntimeError(f"MCP tts failed: {tts_result}")

        tts_payload = tts_result.get("result", {}).get("structuredContent", {})
        if not isinstance(tts_payload, dict):
            raise RuntimeError(f"Unexpected MCP tts payload: {tts_payload}")
        if Path(tts_payload.get("audio_path", "")).resolve() != output_path.resolve():
            raise RuntimeError(f"Unexpected MCP tts audio path: {tts_payload.get('audio_path')}")
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        stderr_text = process.stderr.read() if process.stderr is not None else ""
        if stderr_text:
            print(stderr_text, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for opencode_voice_bridge.py")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Bridge base URL")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Start an in-process mock server instead of calling a real backend",
    )

    args = parser.parse_args()

    base_url = args.base_url
    server: Optional[HTTPServer] = None

    if args.mock:
        server, port = _run_mock_server()
        base_url = f"http://127.0.0.1:{port}"

    try:
        profiles = _read_profiles(base_url)
        profile_id = profiles[0]["id"]

        filtered = _run_bridge(base_url, ["profiles", "--name", "Demo"])
        _assert(isinstance(filtered, list), "profiles --name should return a list")

        with tempfile.TemporaryDirectory(prefix="voicebox-bridge-smoke-") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            audio_input = temp_dir / "input.wav"
            sample_rate = 16000
            duration_seconds = 0.3
            frequency = 440.0
            frame_count = int(sample_rate * duration_seconds)
            samples = bytearray()

            for i in range(frame_count):
                value = int(4_000 * math.sin(2 * math.pi * frequency * i / sample_rate))
                samples.extend(struct.pack("<h", value))

            with wave.open(str(audio_input), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples)

            transcription = _run_bridge(base_url, ["stt", str(audio_input)])
            _assert(isinstance(transcription, dict), "stt output should be an object")
            _assert("text" in transcription, "stt output missing text")

            output_wav = (temp_dir / "output.wav").resolve()
            generation = _run_bridge(
                base_url,
                [
                    "tts",
                    "Hello from smoke test",
                    "--profile-id",
                    profile_id,
                    "--out",
                    str(output_wav),
                ],
            )

            _assert(isinstance(generation, dict), "tts output should be an object")
            _assert("audio_path" in generation, "tts output missing audio_path")
            _assert(
                Path(generation["audio_path"]).exists(), "tts output audio file was not written"
            )
            _assert(
                output_wav == Path(str(generation["audio_path"])).resolve(),
                "tts output path does not match requested path",
            )

            _run_bridge_mcp(
                base_url=base_url,
                audio_input=audio_input,
                profile_id=profile_id,
                mock_backend=args.mock,
            )

            _assert(
                generation.get("profile_id") == profile_id,
                "tts output profile_id does not match requested profile",
            )

        print("Bridge smoke test passed")

    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
