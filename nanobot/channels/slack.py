"""Slack channel implementation using Socket Mode."""

import asyncio
from pathlib import Path
import re
import time
import mimetypes
import tempfile
from typing import Any

from loguru import logger
from slack_sdk.socket_mode.websockets import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient

from slackify_markdown import slackify_markdown

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import SlackConfig


# ---------------------------------------------------------------------------
# Message splitting
# ---------------------------------------------------------------------------

def _split_message(content: str, max_len: int = 3900) -> list[str]:
    """Split content into chunks within *max_len*, preferring paragraph then line breaks."""
    if len(content) <= max_len:
        return [content]
    chunks: list[str] = []
    while content:
        if len(content) <= max_len:
            chunks.append(content)
            break
        cut = content[:max_len]
        # Prefer splitting at double-newline (paragraph boundary)
        pos = cut.rfind("\n\n")
        if pos <= 0:
            pos = cut.rfind("\n")
        if pos <= 0:
            pos = cut.rfind(" ")
        if pos <= 0:
            pos = max_len
        chunks.append(content[:pos])
        content = content[pos:].lstrip()
    return chunks


# ---------------------------------------------------------------------------
# Block Kit builder
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_HR_RE = re.compile(r"^-{3,}$|^\*{3,}$|^_{3,}$", re.MULTILINE)
_BULLET_RE = re.compile(r"^[ \t]*[-*]\s+", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^[ \t]*\d+[.)]\s+", re.MULTILINE)
_TABLE_RE = re.compile(r"(?m)^\|.*\|$(?:\n\|[\s:|-]*\|$)(?:\n\|.*\|$)*")

_MAX_BLOCKS = 50  # Slack Block Kit limit per message
_MAX_TEXT_IN_BLOCK = 3000  # Slack text element limit


def _mrkdwn_section(text: str) -> dict:
    """Build a section block with mrkdwn text."""
    return {"type": "section", "text": {"type": "mrkdwn", "text": text[:_MAX_TEXT_IN_BLOCK]}}


def _convert_table(match: re.Match) -> str:
    """Convert a Markdown table to a Slack-readable list."""
    lines = [ln.strip() for ln in match.group(0).strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return match.group(0)
    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    start = 2 if re.fullmatch(r"[|\s:\-]+", lines[1]) else 1
    rows: list[str] = []
    for line in lines[start:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        cells = (cells + [""] * len(headers))[: len(headers)]
        parts = [f"*{headers[i]}*: {cells[i]}" for i in range(len(headers)) if cells[i]]
        if parts:
            rows.append(" · ".join(parts))
    return "\n".join(rows)


def _build_blocks(content: str) -> list[dict]:
    """Parse agent Markdown into Slack Block Kit blocks.

    Returns a list of block dicts ready for ``chat_postMessage(blocks=...)``.
    Falls back gracefully — if parsing fails, returns a single section block.
    """
    if not content:
        return []

    # Pre-process tables into text before block parsing
    content = _TABLE_RE.sub(_convert_table, content)

    blocks: list[dict] = []

    # Extract code blocks first to avoid parsing their contents
    code_blocks: dict[str, str] = {}
    code_counter = 0

    def _stash_code(m: re.Match) -> str:
        nonlocal code_counter
        key = f"\x00CODE{code_counter}\x00"
        code_counter += 1
        code_blocks[key] = m.group(2)
        return key

    content = _CODE_BLOCK_RE.sub(_stash_code, content)

    # Split into segments by headings and horizontal rules
    segments = re.split(r"(\n*^#{1,3}\s+.+$\n*|^-{3,}$|^\*{3,}$|^_{3,}$)", content, flags=re.MULTILINE)

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Check for heading
        heading_match = _HEADING_RE.match(segment)
        if heading_match:
            heading_text = heading_match.group(2).strip()
            # Slack header blocks only support plain_text, max 150 chars
            blocks.append({
                "type": "header",
                "text": {"type": "plain_text", "text": heading_text[:150], "emoji": True},
            })
            continue

        # Check for horizontal rule
        if _HR_RE.match(segment):
            blocks.append({"type": "divider"})
            continue

        # Check if segment is a stashed code block
        if segment in code_blocks:
            code_text = code_blocks[segment]
            blocks.append({
                "type": "rich_text",
                "elements": [{
                    "type": "rich_text_preformatted",
                    "elements": [{"type": "text", "text": code_text[:_MAX_TEXT_IN_BLOCK]}],
                }],
            })
            continue

        # Check if segment contains a stashed code block mixed with text
        if any(key in segment for key in code_blocks):
            # Split around code placeholders and handle each part
            parts = re.split(r"(\x00CODE\d+\x00)", segment)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part in code_blocks:
                    blocks.append({
                        "type": "rich_text",
                        "elements": [{
                            "type": "rich_text_preformatted",
                            "elements": [{"type": "text", "text": code_blocks[part][:_MAX_TEXT_IN_BLOCK]}],
                        }],
                    })
                else:
                    mrkdwn = slackify_markdown(part)
                    if mrkdwn.strip():
                        blocks.append(_mrkdwn_section(mrkdwn))
            continue

        # Check for bullet or numbered list
        lines = segment.split("\n")
        is_list = all(
            _BULLET_RE.match(ln) or _NUMBERED_RE.match(ln) or not ln.strip()
            for ln in lines
        ) and any(_BULLET_RE.match(ln) or _NUMBERED_RE.match(ln) for ln in lines)

        if is_list:
            list_style = "bullet"
            if any(_NUMBERED_RE.match(ln) for ln in lines if ln.strip()):
                list_style = "ordered"
            elements = []
            for ln in lines:
                text = re.sub(r"^[ \t]*[-*]\s+|^[ \t]*\d+[.)]\s+", "", ln).strip()
                if text:
                    elements.append({
                        "type": "rich_text_section",
                        "elements": [{"type": "text", "text": text}],
                    })
            if elements:
                blocks.append({
                    "type": "rich_text",
                    "elements": [{
                        "type": "rich_text_list",
                        "style": list_style,
                        "elements": elements,
                    }],
                })
            continue

        # Default: regular text paragraph → section with mrkdwn
        mrkdwn = slackify_markdown(segment)
        if mrkdwn.strip():
            blocks.append(_mrkdwn_section(mrkdwn))

    return blocks


# ---------------------------------------------------------------------------
# Channel implementation
# ---------------------------------------------------------------------------

class SlackChannel(BaseChannel):
    """Slack channel using Socket Mode with auto-reconnection and Block Kit formatting."""

    name = "slack"

    def __init__(self, config: SlackConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: SlackConfig = config
        self._web_client: AsyncWebClient | None = None
        self._socket_client: SocketModeClient | None = None
        self._bot_user_id: str | None = None
        # Health / reconnection state
        self._connected: bool = False
        self._reconnect_count: int = 0
        self._last_event_time: float = 0.0
        self._health_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the Slack Socket Mode client with auto-reconnection."""
        if not self.config.bot_token or not self.config.app_token:
            logger.error("Slack bot/app token not configured")
            return
        if self.config.mode != "socket":
            logger.error(f"Unsupported Slack mode: {self.config.mode}")
            return

        self._running = True
        self._web_client = AsyncWebClient(token=self.config.bot_token)

        delay = self.config.reconnect_delay_s

        while self._running:
            try:
                # Auth test — validates tokens before connecting
                auth = await self._web_client.auth_test()
                self._bot_user_id = auth.get("user_id")
                logger.info(f"Slack bot authenticated as {self._bot_user_id}")

                # Create fresh socket client each attempt
                self._socket_client = SocketModeClient(
                    app_token=self.config.app_token,
                    web_client=self._web_client,
                )
                self._socket_client.socket_mode_request_listeners.append(self._on_socket_request)

                await self._socket_client.connect()
                self._connected = True
                self._last_event_time = time.monotonic()
                delay = self.config.reconnect_delay_s  # reset backoff
                logger.info("Slack Socket Mode connected")

                # Start health monitor
                if self._health_task is None or self._health_task.done():
                    self._health_task = asyncio.create_task(self._health_loop())

                # Hold connection open
                while self._running and self._connected:
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                self._reconnect_count += 1
                logger.warning(
                    f"Slack connection failed (attempt {self._reconnect_count}): {e}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.max_reconnect_delay_s)

    async def stop(self) -> None:
        """Stop the Slack client."""
        self._running = False
        self._connected = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        if self._socket_client:
            try:
                await self._socket_client.close()
            except Exception as e:
                logger.warning(f"Slack socket close failed: {e}")
            self._socket_client = None

    # ------------------------------------------------------------------
    # Health monitoring
    # ------------------------------------------------------------------

    async def _health_loop(self) -> None:
        """Periodically check connection health."""
        while self._running:
            await asyncio.sleep(60)
            if not self._connected:
                continue
            idle = time.monotonic() - self._last_event_time
            if idle > self.config.health_warn_idle_s:
                logger.warning(f"Slack: no events received for {idle:.0f}s")

    def health(self) -> dict[str, Any]:
        """Return connection health info."""
        idle = time.monotonic() - self._last_event_time if self._last_event_time else 0
        return {
            "connected": self._connected,
            "last_event_age_s": round(idle, 1),
            "reconnect_count": self._reconnect_count,
        }

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Slack with Block Kit formatting."""
        if not self._web_client:
            logger.warning("Slack client not running")
            return

        slack_meta = msg.metadata.get("slack", {}) if msg.metadata else {}
        thread_ts = slack_meta.get("thread_ts")
        channel_type = slack_meta.get("channel_type")
        event_ts = slack_meta.get("event", {}).get("ts") if slack_meta.get("event") else None
        use_thread = thread_ts and channel_type != "im"

        try:
            blocks = _build_blocks(msg.content)
            fallback_text = _to_mrkdwn(msg.content)

            if blocks and len(blocks) <= _MAX_BLOCKS:
                await self._web_client.chat_postMessage(
                    channel=msg.chat_id,
                    blocks=blocks,
                    text=fallback_text,
                    thread_ts=thread_ts if use_thread else None,
                )
            elif blocks:
                # Too many blocks — split into multiple messages
                for i in range(0, len(blocks), _MAX_BLOCKS):
                    chunk_blocks = blocks[i : i + _MAX_BLOCKS]
                    await self._web_client.chat_postMessage(
                        channel=msg.chat_id,
                        blocks=chunk_blocks,
                        text=fallback_text if i == 0 else "...",
                        thread_ts=thread_ts if use_thread else None,
                    )
                    if i + _MAX_BLOCKS < len(blocks):
                        await asyncio.sleep(0.3)
            else:
                # Fallback to plain text with chunking
                for i, chunk in enumerate(_split_message(fallback_text, self.config.max_message_length)):
                    await self._web_client.chat_postMessage(
                        channel=msg.chat_id,
                        text=chunk,
                        thread_ts=thread_ts if use_thread else None,
                    )
                    if i > 0:
                        await asyncio.sleep(0.3)

            # Upload media files if present
            if msg.media:
                await self._upload_media(msg.chat_id, msg.media, thread_ts if use_thread else None)

            # Done emoji reaction
            if self.config.done_emoji and event_ts:
                try:
                    await self._web_client.reactions_add(
                        channel=msg.chat_id,
                        name=self.config.done_emoji,
                        timestamp=event_ts,
                    )
                except Exception:
                    pass

            # Remove eyes reaction after response
            if self.config.react_emoji and event_ts:
                try:
                    await self._web_client.reactions_remove(
                        channel=msg.chat_id,
                        name=self.config.react_emoji,
                        timestamp=event_ts,
                    )
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")

    async def _upload_media(
        self, channel: str, media: list[str], thread_ts: str | None
    ) -> None:
        """Upload media files to the channel (best-effort per file)."""
        if not self._web_client:
            return
        for url_or_path in media:
            try:
                if url_or_path.startswith(("http://", "https://")):
                    # Download then upload
                    import aiohttp

                    async with aiohttp.ClientSession() as session:
                        async with session.get(url_or_path) as resp:
                            if resp.status == 200:
                                data = await resp.read()
                                filename = url_or_path.rsplit("/", 1)[-1] or "file"
                                await self._web_client.files_upload_v2(
                                    channel=channel,
                                    content=data,
                                    filename=filename,
                                    thread_ts=thread_ts,
                                )
                else:
                    await self._web_client.files_upload_v2(
                        channel=channel,
                        file=url_or_path,
                        thread_ts=thread_ts,
                    )
            except Exception as e:
                logger.warning(f"Slack file upload failed for {url_or_path}: {e}")

    # ------------------------------------------------------------------
    # Inbound
    # ------------------------------------------------------------------

    async def _on_socket_request(
        self,
        client: SocketModeClient,
        req: SocketModeRequest,
    ) -> None:
        """Handle incoming Socket Mode requests."""
        self._last_event_time = time.monotonic()

        if req.type != "events_api":
            return

        await client.send_socket_mode_response(
            SocketModeResponse(envelope_id=req.envelope_id)
        )

        payload = req.payload or {}
        event = payload.get("event") or {}
        event_type = event.get("type")

        if event_type not in ("message", "app_mention"):
            return

        sender_id = event.get("user")
        chat_id = event.get("channel")

        # Ignore bot/system messages
        if event.get("subtype"):
            return
        if self._bot_user_id and sender_id == self._bot_user_id:
            return

        # Avoid double-processing: prefer app_mention over message for mentions
        text = event.get("text") or ""
        if event_type == "message" and self._bot_user_id and f"<@{self._bot_user_id}>" in text:
            return

        logger.debug(
            "Slack event: type={} user={} channel={} channel_type={} text={}",
            event_type, sender_id, chat_id, event.get("channel_type"), text[:80],
        )
        if not sender_id or not chat_id:
            return

        channel_type = event.get("channel_type") or ""

        if not self._is_allowed(sender_id, chat_id, channel_type):
            return

        if channel_type != "im" and not self._should_respond_in_channel(event_type, text, chat_id):
            return

        text = self._strip_bot_mention(text)
        file_snippets = await self._transcribe_slack_attachments(event.get("files"))
        if file_snippets:
            if text:
                text = f"{text}\n\n" + "\n".join(file_snippets)
            else:
                text = "\n".join(file_snippets)

        thread_ts = event.get("thread_ts")
        if self.config.reply_in_thread and not thread_ts:
            thread_ts = event.get("ts")

        # Add eyes reaction (best-effort, retry once)
        if self._web_client and event.get("ts"):
            for attempt in range(2):
                try:
                    await self._web_client.reactions_add(
                        channel=chat_id,
                        name=self.config.react_emoji,
                        timestamp=event.get("ts"),
                    )
                    break
                except Exception as e:
                    if attempt == 0:
                        await asyncio.sleep(0.5)
                    else:
                        logger.debug(f"Slack reactions_add failed: {e}")

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=text,
            metadata={
                "slack": {
                    "event": event,
                    "thread_ts": thread_ts,
                    "channel_type": channel_type,
                }
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_allowed(self, sender_id: str, chat_id: str, channel_type: str) -> bool:
        if channel_type == "im":
            if not self.config.dm.enabled:
                return False
            if self.config.dm.policy == "allowlist":
                return sender_id in self.config.dm.allow_from
            return True
        if self.config.group_policy == "allowlist":
            return chat_id in self.config.group_allow_from
        return True

    def _should_respond_in_channel(self, event_type: str, text: str, chat_id: str) -> bool:
        if self.config.group_policy == "open":
            return True
        if self.config.group_policy == "mention":
            if event_type == "app_mention":
                return True
            return self._bot_user_id is not None and f"<@{self._bot_user_id}>" in text
        if self.config.group_policy == "allowlist":
            return chat_id in self.config.group_allow_from
        return False

    def _strip_bot_mention(self, text: str) -> str:
        if not text or not self._bot_user_id:
            return text
        return re.sub(rf"<@{re.escape(self._bot_user_id)}>\s*", "", text).strip()

    @staticmethod
    def _is_audio_file(file_info: dict[str, Any]) -> bool:
        """Return True when a Slack file attachment looks like audio."""
        mimetype = (file_info.get("mimetype") or "").lower()
        filetype = (file_info.get("filetype") or "").lower()
        return mimetype.startswith("audio/") or filetype.startswith("audio")

    @staticmethod
    def _audio_extension(file_info: dict[str, Any]) -> str:
        """Infer a file extension from Slack attachment metadata."""
        mimetype = file_info.get("mimetype")
        if isinstance(mimetype, str):
            guess = mimetypes.guess_extension(mimetype)
            if guess:
                return guess
        name = file_info.get("name")
        if isinstance(name, str):
            suffixes = Path(name).suffixes
            if suffixes:
                return suffixes[-1]
        return ".mp3"

    async def _transcribe_slack_attachments(self, files: Any) -> list[str]:
        """Download and transcribe Slack audio files from the event payload."""
        if not isinstance(files, list):
            return []

        snippets: list[str] = []
        for file_info in files:
            if not isinstance(file_info, dict) or not self._is_audio_file(file_info):
                continue

            url = (
                file_info.get("url_private_download")
                or file_info.get("url_private")
            )
            if not isinstance(url, str):
                snippets.append("[Voice message: missing attachment URL]")
                continue

            transcript = await self._transcribe_slack_audio_file(url, file_info)
            if transcript:
                snippets.append(f"[Voice message: {transcript}]")
            else:
                snippets.append("[Voice message: could not transcribe]")
        return snippets

    async def _transcribe_slack_audio_file(self, url: str, file_info: dict[str, Any] | None = None) -> str:
        """Download and transcribe a Slack audio attachment URL."""
        if not self._web_client or not self.config.bot_token:
            return ""

        try:
            import httpx
            from nanobot.providers.transcription import GroqTranscriptionProvider
        except Exception:
            return ""

        provider = GroqTranscriptionProvider()
        if not provider.api_key:
            return ""

        suffix = self._audio_extension(file_info or {})
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name

        try:
            async with httpx.AsyncClient(
                timeout=60.0,
                headers={"Authorization": f"Bearer {self.config.bot_token}"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                Path(tmp_path).write_bytes(response.content)

            return await provider.transcribe(tmp_path)
        except Exception as e:
            logger.warning(f"Slack audio transcription failed for {url}: {e}")
            return ""
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Plain mrkdwn fallback (module-level for reuse)
# ---------------------------------------------------------------------------

def _to_mrkdwn(text: str) -> str:
    """Convert Markdown to Slack mrkdwn (plain text fallback)."""
    if not text:
        return ""
    text = _TABLE_RE.sub(_convert_table, text)
    return slackify_markdown(text)
