"""Document tools: document_editor, calendar_read."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


def _resolve_path(path: str, allowed_dir: Path | None = None) -> Path:
    """Resolve path and optionally enforce directory restriction."""
    resolved = Path(path).expanduser().resolve()
    if allowed_dir and not str(resolved).startswith(str(allowed_dir.resolve())):
        raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


class DocumentEditorTool(Tool):
    """Create or edit text/markdown documents."""

    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "document_editor"

    @property
    def description(self) -> str:
        return (
            "Create or edit documents (text, markdown, etc.). "
            "Actions: 'create' (new file), 'edit' (overwrite), 'append' (add to end). "
            "Provide path and content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action: 'create', 'edit', or 'append'",
                },
                "path": {"type": "string", "description": "Document file path"},
                "content": {"type": "string", "description": "Content to write or append"},
            },
            "required": ["action", "path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "").strip().lower()
        path_str = kwargs.get("path", "").strip()
        content = kwargs.get("content", "")

        if not action:
            return "Error: No action specified. Use 'create', 'edit', or 'append'."
        if action not in ("create", "edit", "append"):
            return f"Error: Unknown action '{action}'. Use 'create', 'edit', or 'append'."
        if not path_str:
            return "Error: No file path provided."

        try:
            file_path = _resolve_path(path_str, self._allowed_dir)
        except PermissionError as e:
            return f"Error: {e}"

        try:
            if action == "create":
                if file_path.exists():
                    return f"Error: File already exists: {path_str}. Use 'edit' to overwrite."
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
                return f"Created: {file_path.name} ({len(content)} chars)"

            elif action == "edit":
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")
                return f"Updated: {file_path.name} ({len(content)} chars)"

            elif action == "append":
                if not file_path.exists():
                    return f"Error: File not found: {path_str}. Use 'create' first."
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(content)
                return f"Appended {len(content)} chars to {file_path.name}"

        except OSError as e:
            return f"Error: {e}"

        return "Error: Unexpected state."


class CalendarReadTool(Tool):
    """Read calendar events from .ics files."""

    @property
    def name(self) -> str:
        return "calendar_read"

    @property
    def description(self) -> str:
        return (
            "Read calendar events from ICS (iCalendar) files. "
            "Provide a file path to an .ics file or a directory to scan for .ics files. "
            "Optionally filter by date range."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to .ics file or directory containing .ics files",
                },
                "start_date": {
                    "type": "string",
                    "description": "Filter: only events on or after this date (YYYY-MM-DD)",
                },
                "end_date": {
                    "type": "string",
                    "description": "Filter: only events on or before this date (YYYY-MM-DD)",
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        path_str = kwargs.get("path", "").strip()
        start_str = kwargs.get("start_date", "").strip()
        end_str = kwargs.get("end_date", "").strip()

        if not path_str:
            return "Error: No path provided. Provide a path to an .ics file or directory."

        target = Path(path_str).expanduser().resolve()

        ics_files = []
        if target.is_file() and target.suffix.lower() == ".ics":
            ics_files = [target]
        elif target.is_dir():
            ics_files = sorted(target.glob("*.ics"))
        else:
            return f"Error: '{path_str}' is not an .ics file or directory."

        if not ics_files:
            return f"No .ics files found in: {path_str}"

        start_date = datetime.strptime(start_str, "%Y-%m-%d") if start_str else None
        end_date = datetime.strptime(end_str, "%Y-%m-%d") if end_str else None

        all_events = []
        for ics_path in ics_files:
            try:
                content = ics_path.read_text(encoding="utf-8")
                events = self._parse_ics(content)
                all_events.extend(events)
            except Exception as e:
                all_events.append({"error": f"Failed to parse {ics_path.name}: {e}"})

        # Filter by date
        if start_date or end_date:
            filtered = []
            for ev in all_events:
                if "error" in ev:
                    continue
                dt_str = ev.get("dtstart", "")
                try:
                    dt = datetime.strptime(dt_str[:10], "%Y-%m-%d") if len(dt_str) >= 10 else None
                except ValueError:
                    dt = None
                if dt:
                    if start_date and dt < start_date:
                        continue
                    if end_date and dt > end_date:
                        continue
                filtered.append(ev)
            all_events = filtered

        if not all_events:
            return "No events found matching criteria."

        # Format output
        lines = [f"Found {len(all_events)} event(s):\n"]
        for i, ev in enumerate(all_events[:50], 1):
            if "error" in ev:
                lines.append(f"{i}. {ev['error']}")
                continue
            summary = ev.get("summary", "Untitled")
            dtstart = ev.get("dtstart", "?")
            dtend = ev.get("dtend", "")
            location = ev.get("location", "")
            desc = ev.get("description", "")

            lines.append(f"{i}. {summary}")
            lines.append(f"   When: {dtstart}" + (f" to {dtend}" if dtend else ""))
            if location:
                lines.append(f"   Where: {location}")
            if desc:
                lines.append(f"   Details: {desc[:200]}")
            lines.append("")

        if len(all_events) > 50:
            lines.append(f"... and {len(all_events) - 50} more events")

        return "\n".join(lines)

    @staticmethod
    def _parse_ics(content: str) -> list[dict[str, str]]:
        """Parse VEVENT blocks from ICS content using regex."""
        events = []
        # Split into VEVENT blocks
        vevent_pattern = re.compile(r'BEGIN:VEVENT(.*?)END:VEVENT', re.DOTALL)
        for match in vevent_pattern.finditer(content):
            block = match.group(1)
            event: dict[str, str] = {}
            # Unfold lines (RFC 5545: continuation lines start with space/tab)
            block = re.sub(r'\r?\n[ \t]', '', block)
            for line in block.strip().split('\n'):
                line = line.strip()
                if ':' in line:
                    key, _, value = line.partition(':')
                    # Strip parameters (e.g., DTSTART;VALUE=DATE:20240101 -> DTSTART)
                    key = key.split(';')[0].upper()
                    if key == "SUMMARY":
                        event["summary"] = value.replace('\\n', ' ').replace('\\,', ',')
                    elif key == "DTSTART":
                        event["dtstart"] = _format_ics_date(value)
                    elif key == "DTEND":
                        event["dtend"] = _format_ics_date(value)
                    elif key == "LOCATION":
                        event["location"] = value.replace('\\n', ' ').replace('\\,', ',')
                    elif key == "DESCRIPTION":
                        event["description"] = value.replace('\\n', '\n').replace('\\,', ',')
            if event:
                events.append(event)
        return events


def _format_ics_date(value: str) -> str:
    """Format an ICS date/datetime string into readable form."""
    value = value.strip().rstrip('Z')
    if len(value) == 8:  # YYYYMMDD
        try:
            return datetime.strptime(value, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return value
    elif len(value) >= 15:  # YYYYMMDDTHHmmss
        try:
            return datetime.strptime(value[:15], "%Y%m%dT%H%M%S").strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return value
    return value
