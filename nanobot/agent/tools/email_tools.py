"""Email tools: read and send email via IMAP/SMTP."""

import asyncio
import imaplib
import re
import smtplib
import ssl
from email import policy
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parseaddr
from typing import Any

from nanobot.agent.tools.base import Tool


class EmailReadTool(Tool):
    """Read recent emails from inbox via IMAP."""

    name = "email_read"
    description = (
        "Read recent emails from any mail folder. "
        "Returns sender, subject, date, and body for each message. "
        "Use count to limit results (default 10, max 25). "
        "Use folder to specify which folder: INBOX (default), "
        "\"[Gmail]/Sent Mail\" for sent, \"[Gmail]/Drafts\" for drafts, "
        "\"[Gmail]/Spam\" for spam, \"[Gmail]/Trash\" for trash, "
        "\"[Gmail]/All Mail\" for all mail."
    )
    parameters = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "Number of recent emails to fetch (default 10, max 25)",
                "minimum": 1,
                "maximum": 25,
            },
            "unread_only": {
                "type": "boolean",
                "description": "Only fetch unread emails (default false)",
            },
            "search": {
                "type": "string",
                "description": "Optional search term to filter by subject or sender",
            },
            "folder": {
                "type": "string",
                "description": "Mail folder to read from. Default is INBOX. Use \"[Gmail]/Sent Mail\" for sent emails, \"[Gmail]/Drafts\" for drafts, etc.",
            },
        },
        "required": [],
    }

    def __init__(self, imap_host: str, imap_port: int, username: str, password: str, use_ssl: bool = True, mailbox: str = "INBOX"):
        self._host = imap_host
        self._port = imap_port
        self._user = username
        self._pass = password
        self._ssl = use_ssl
        self._mailbox = mailbox

    async def execute(self, count: int = 10, unread_only: bool = False, search: str = "", folder: str = "") -> str:
        count = min(max(1, count), 25)
        return await asyncio.to_thread(self._fetch, count, unread_only, search, folder)

    def _fetch(self, count: int, unread_only: bool, search: str, folder: str) -> str:
        if self._ssl:
            client = imaplib.IMAP4_SSL(self._host, self._port)
        else:
            client = imaplib.IMAP4(self._host, self._port)

        try:
            client.login(self._user, self._pass)
            mailbox = folder if folder else self._mailbox
            # IMAP requires quoted folder names with spaces/special chars
            if ' ' in mailbox or '/' in mailbox:
                mailbox = f'"{mailbox}"'
            status, _ = client.select(mailbox, readonly=True)
            if status != "OK":
                return "Error: Could not open mailbox"

            criteria = "UNSEEN" if unread_only else "ALL"
            status, data = client.search(None, criteria)
            if status != "OK" or not data or not data[0]:
                return "No emails found."

            ids = data[0].split()
            ids = ids[-count:]  # Most recent
            ids.reverse()

            results = []
            for imap_id in ids:
                status, fetched = client.fetch(imap_id, "(BODY.PEEK[])")
                if status != "OK" or not fetched:
                    continue

                raw = self._extract_bytes(fetched)
                if not raw:
                    continue

                parsed = BytesParser(policy=policy.default).parsebytes(raw)
                sender = parseaddr(parsed.get("From", ""))[1]
                sender_name = parseaddr(parsed.get("From", ""))[0]
                recipient = self._decode_header(parsed.get("To", ""))
                cc = self._decode_header(parsed.get("Cc", ""))
                subject = self._decode_header(parsed.get("Subject", ""))
                date_val = parsed.get("Date", "")
                body = self._extract_body(parsed)[:3000]

                if search:
                    term = search.lower()
                    searchable = f"{subject} {sender} {sender_name} {recipient} {cc}".lower()
                    if term not in searchable:
                        continue

                display_from = f"{sender_name} <{sender}>" if sender_name else sender
                entry = f"From: {display_from}\n"
                if recipient:
                    entry += f"To: {recipient}\n"
                if cc:
                    entry += f"Cc: {cc}\n"
                entry += (
                    f"Subject: {subject}\n"
                    f"Date: {date_val}\n"
                    f"Body:\n{body}\n"
                    f"---"
                )
                results.append(entry)

            if not results:
                return "No emails found matching your criteria."
            return f"Found {len(results)} email(s):\n\n" + "\n\n".join(results)
        finally:
            try:
                client.logout()
            except Exception:
                pass

    @staticmethod
    def _extract_bytes(fetched) -> bytes | None:
        for item in fetched:
            if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], (bytes, bytearray)):
                return bytes(item[1])
        return None

    @staticmethod
    def _decode_header(value: str) -> str:
        if not value:
            return ""
        try:
            return str(make_header(decode_header(value)))
        except Exception:
            return value

    @classmethod
    def _extract_body(cls, msg) -> str:
        import html as html_mod

        if msg.is_multipart():
            plain = []
            for part in msg.walk():
                if part.get_content_disposition() == "attachment":
                    continue
                if part.get_content_type() == "text/plain":
                    try:
                        plain.append(part.get_content())
                    except Exception:
                        payload = part.get_payload(decode=True) or b""
                        plain.append(payload.decode("utf-8", errors="replace"))
            if plain:
                return "\n\n".join(p for p in plain if isinstance(p, str)).strip()
        try:
            payload = msg.get_content()
        except Exception:
            payload = (msg.get_payload(decode=True) or b"").decode("utf-8", errors="replace")
        if isinstance(payload, str):
            if msg.get_content_type() == "text/html":
                payload = re.sub(r"<[^>]+>", "", payload)
                payload = html_mod.unescape(payload)
            return payload.strip()
        return ""


class EmailSendTool(Tool):
    """Send an email via SMTP."""

    name = "email_send"
    description = (
        "Send an email. Requires recipient address, subject, and body. "
        "Only use when the user explicitly asks you to send an email."
    )
    parameters = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject line"},
            "body": {"type": "string", "description": "Email body text"},
        },
        "required": ["to", "subject", "body"],
    }

    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str, from_addr: str, use_tls: bool = True, use_ssl: bool = False):
        self._host = smtp_host
        self._port = smtp_port
        self._user = username
        self._pass = password
        self._from = from_addr
        self._tls = use_tls
        self._ssl = use_ssl

    async def execute(self, to: str, subject: str, body: str) -> str:
        return await asyncio.to_thread(self._send, to, subject, body)

    def _send(self, to: str, subject: str, body: str) -> str:
        msg = EmailMessage()
        msg["From"] = self._from
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)

        try:
            if self._ssl:
                with smtplib.SMTP_SSL(self._host, self._port, timeout=30) as smtp:
                    smtp.login(self._user, self._pass)
                    smtp.send_message(msg)
            else:
                with smtplib.SMTP(self._host, self._port, timeout=30) as smtp:
                    if self._tls:
                        smtp.starttls(context=ssl.create_default_context())
                    smtp.login(self._user, self._pass)
                    smtp.send_message(msg)
            return f"Email sent to {to} with subject: {subject}"
        except Exception as e:
            return f"Error sending email: {e}"
