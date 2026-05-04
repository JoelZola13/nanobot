"""Server-side privacy filter — defense in depth alongside the client filter.

Mirrors the rules in client-sdk/src/privacy.ts. Anything matching is removed
from `properties` and a schema_violation row is logged so we can chase down the
upstream caller.
"""

from __future__ import annotations

import re
from typing import Any


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\s().-]?){7,}\d")
TOKEN_RE = re.compile(r"\b(?:sk|pk|phc|phx|ghp|xox[baprs])[-_][A-Za-z0-9]{16,}\b")
ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b",
    re.IGNORECASE,
)

FORBIDDEN_KEYS = {
    "message", "text", "body", "content", "prompt", "query", "q", "search", "search_text",
    "note", "notes", "caption", "description", "bio",
    "cover_letter", "cover_letter_text",
    "resume", "resume_text", "cv", "cv_text",
    "case_note", "case_notes", "case_narrative", "case_summary",
    "email", "email_address",
    "phone", "phone_number", "tel",
    "address", "street", "street_address", "address_line", "address_line_1", "address_line_2",
    "name", "full_name", "first_name", "last_name", "display_name", "username_text",
    "file_name", "filename", "original_name", "attachment_name",
    "card", "card_number", "cvv", "cvc", "iban", "routing_number", "account_number",
    "token", "access_token", "refresh_token", "api_key", "secret", "password",
}

STRING_VALUE_LIMIT = 200


def scrub(value: Any) -> tuple[Any, list[str]]:
    """Recursively scrub PII. Returns (cleaned_value, redacted_paths)."""
    redacted: list[str] = []
    cleaned = _scrub_inner(value, "", redacted)
    return cleaned, redacted


def _scrub_inner(value: Any, path: str, redacted: list[str]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            sub_path = f"{path}.{k}" if path else k
            if k.lower() in FORBIDDEN_KEYS:
                redacted.append(sub_path)
                continue
            cleaned = _scrub_inner(v, sub_path, redacted)
            out[k] = cleaned
        return out
    if isinstance(value, list):
        return [_scrub_inner(v, f"{path}[]", redacted) for v in value]
    if isinstance(value, str):
        if len(value) > STRING_VALUE_LIMIT:
            redacted.append(path or "<root>")
            return None
        if EMAIL_RE.search(value) or PHONE_RE.search(value) or TOKEN_RE.search(value) or ADDRESS_RE.search(value):
            redacted.append(path or "<root>")
            return None
    return value
