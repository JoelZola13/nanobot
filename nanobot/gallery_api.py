"""Street Gallery API – route handlers for the gallery feature.

All handlers are async Starlette request handlers that return JSONResponse.
Database: PostgreSQL via asyncpg, using the social-postgres container.
"""
from __future__ import annotations

import asyncio
import os
import smtplib
import ssl
import uuid
import json
from datetime import datetime
from decimal import Decimal
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import asyncpg
from loguru import logger
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse


def _load_smtp_config() -> dict[str, Any] | None:
    """Read SMTP settings from ~/.nanobot/config.json (channels.email)."""
    try:
        cfg_path = Path.home() / ".nanobot" / "config.json"
        if not cfg_path.is_file():
            return None
        cfg = json.loads(cfg_path.read_text())
        email_cfg = (cfg.get("channels", {}) or {}).get("email") or cfg.get("email") or {}
        host = email_cfg.get("smtpHost") or email_cfg.get("smtp_host")
        port = email_cfg.get("smtpPort") or email_cfg.get("smtp_port") or 587
        user = email_cfg.get("smtpUsername") or email_cfg.get("smtp_username")
        pwd = email_cfg.get("smtpPassword") or email_cfg.get("smtp_password")
        from_addr = email_cfg.get("fromAddress") or user
        if not (host and user and pwd and from_addr):
            return None
        return {
            "host": host,
            "port": int(port),
            "user": user,
            "password": pwd,
            "from_addr": from_addr,
            "use_tls": bool(email_cfg.get("smtpUseTls", email_cfg.get("smtp_use_tls", True))),
            "use_ssl": bool(email_cfg.get("smtpUseSsl", email_cfg.get("smtp_use_ssl", False))),
        }
    except Exception as e:
        logger.warning(f"SMTP config load failed: {e}")
        return None


def _send_email_sync(to_addr: str, subject: str, body_text: str, body_html: str | None = None) -> bool:
    """Blocking SMTP send. Returns True on success."""
    smtp_cfg = _load_smtp_config()
    if smtp_cfg is None:
        logger.warning("SMTP not configured; skipping email send")
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_cfg["from_addr"]
    msg["To"] = to_addr
    msg.set_content(body_text)
    if body_html:
        msg.add_alternative(body_html, subtype="html")
    try:
        if smtp_cfg["use_ssl"]:
            with smtplib.SMTP_SSL(smtp_cfg["host"], smtp_cfg["port"], timeout=30) as smtp:
                smtp.login(smtp_cfg["user"], smtp_cfg["password"])
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"], timeout=30) as smtp:
                if smtp_cfg["use_tls"]:
                    smtp.starttls(context=ssl.create_default_context())
                smtp.login(smtp_cfg["user"], smtp_cfg["password"])
                smtp.send_message(msg)
        return True
    except Exception as e:
        logger.error(f"SMTP send to {to_addr} failed: {e}")
        return False


async def _send_email(to_addr: str, subject: str, body_text: str, body_html: str | None = None) -> bool:
    """Async wrapper — runs SMTP in a thread so the event loop stays free."""
    return await asyncio.to_thread(_send_email_sync, to_addr, subject, body_text, body_html)

# Persistent upload directory (volume-mounted in docker-compose via ~/.nanobot).
GALLERY_UPLOAD_DIR = Path(
    os.environ.get(
        "GALLERY_UPLOAD_DIR",
        str(Path.home() / ".nanobot" / "workspace" / "gallery" / "images"),
    )
)

# ── Database pool ────────────────────────────────────────────────────────────

_pool: asyncpg.Pool | None = None

# Try multiple connection strings – Docker internal first, then localhost fallback
_DB_URLS = [
    "postgresql://social:social_password@social-postgres:5432/social",
    "postgresql://lobehub:lobehub_password@localhost:5433/social",
]


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        last_err = None
        for url in _DB_URLS:
            try:
                _pool = await asyncpg.create_pool(url, min_size=1, max_size=5)
                return _pool
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Cannot connect to gallery DB: {last_err}")
    return _pool


# ── Serialisation helper ─────────────────────────────────────────────────────

def _serialise(row: asyncpg.Record) -> dict[str, Any]:
    """Convert an asyncpg Record to a JSON-safe dict."""
    d: dict[str, Any] = {}
    for key, val in dict(row).items():
        if isinstance(val, datetime):
            d[key] = val.isoformat()
        elif isinstance(val, Decimal):
            d[key] = float(val)
        elif isinstance(val, list):
            d[key] = list(val)
        else:
            d[key] = val
    return d


# ── Route handlers ───────────────────────────────────────────────────────────

async def gallery_list_artworks(request: Request) -> JSONResponse:
    """GET /gallery/artworks — list artworks with optional filters."""
    pool = await _get_pool()
    params = request.query_params

    conditions = ["is_public = true", "is_approved = true"]
    args: list[Any] = []
    idx = 1

    if params.get("medium"):
        conditions.append(f"medium = ${idx}")
        args.append(params["medium"])
        idx += 1

    if params.get("style"):
        conditions.append(f"style = ${idx}")
        args.append(params["style"])
        idx += 1

    if params.get("tags"):
        tag_list = [t.strip() for t in params["tags"].split(",") if t.strip()]
        if tag_list:
            conditions.append(f"tags @> ${idx}::text[]")
            args.append(tag_list)
            idx += 1

    if params.get("is_for_sale") == "true":
        conditions.append("is_for_sale = true")

    if params.get("search"):
        search = f"%{params['search']}%"
        conditions.append(f"(title ILIKE ${idx} OR description ILIKE ${idx})")
        args.append(search)
        idx += 1

    where = " AND ".join(conditions)
    query = f"""
        SELECT * FROM gallery_artworks
        WHERE {where}
        ORDER BY display_order ASC NULLS LAST, created_at DESC
        LIMIT 100
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *args)

    return JSONResponse([_serialise(r) for r in rows])


async def gallery_list_mediums(request: Request) -> JSONResponse:
    """GET /gallery/artworks/mediums — distinct mediums."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT medium FROM gallery_artworks "
            "WHERE medium IS NOT NULL AND is_public = true "
            "ORDER BY medium"
        )
    return JSONResponse([{"value": r["medium"], "label": r["medium"]} for r in rows])


async def gallery_get_artwork(request: Request) -> JSONResponse:
    """GET /gallery/artworks/{artwork_id} — single artwork."""
    artwork_id = request.path_params["artwork_id"]
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM gallery_artworks WHERE id = $1", artwork_id
        )
    if row is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(_serialise(row))


async def gallery_update_artwork(request: Request) -> JSONResponse:
    """PATCH /gallery/artworks/{artwork_id}?user_id=... — owner-only update.

    Accepts JSON body with any of: price, is_for_sale, currency, is_sold.
    """
    artwork_id = request.path_params["artwork_id"]
    user_id = request.query_params.get("user_id", "").strip()
    if not user_id:
        return JSONResponse({"error": "user_id required"}, status_code=400)

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be an object"}, status_code=400)

    updates: list[tuple[str, Any]] = []
    if "price" in body:
        price_raw = body["price"]
        if price_raw in (None, ""):
            updates.append(("price", None))
        else:
            try:
                price_val = Decimal(str(price_raw))
            except Exception:
                return JSONResponse({"error": "invalid price"}, status_code=400)
            if price_val < 0:
                return JSONResponse({"error": "price must be >= 0"}, status_code=400)
            updates.append(("price", price_val))
    if "is_for_sale" in body:
        updates.append(("is_for_sale", bool(body["is_for_sale"])))
    if "is_sold" in body:
        updates.append(("is_sold", bool(body["is_sold"])))
    if "currency" in body and isinstance(body["currency"], str) and body["currency"].strip():
        updates.append(("currency", body["currency"].strip().upper()[:3]))

    if not updates:
        return JSONResponse({"error": "no updatable fields provided"}, status_code=400)

    pool = await _get_pool()
    async with pool.acquire() as conn:
        owner_row = await conn.fetchrow(
            "SELECT artist_id FROM gallery_artworks WHERE id = $1", artwork_id
        )
        if owner_row is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        if owner_row["artist_id"] != user_id:
            return JSONResponse({"error": "not owner"}, status_code=403)

        set_clauses = ", ".join(f"{col} = ${i + 2}" for i, (col, _) in enumerate(updates))
        args: list[Any] = [artwork_id] + [val for _, val in updates]
        row = await conn.fetchrow(
            f"UPDATE gallery_artworks SET {set_clauses}, updated_at = NOW() "
            f"WHERE id = $1 RETURNING *",
            *args,
        )

    return JSONResponse(_serialise(row))


async def gallery_delete_artwork(request: Request) -> JSONResponse:
    """DELETE /gallery/artworks/{artwork_id}?user_id=... — owner-only delete."""
    artwork_id = request.path_params["artwork_id"]
    user_id = request.query_params.get("user_id", "").strip()
    if not user_id:
        return JSONResponse({"error": "user_id required"}, status_code=400)

    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT artist_id, image_url FROM gallery_artworks WHERE id = $1",
            artwork_id,
        )
        if row is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        if row["artist_id"] != user_id:
            return JSONResponse({"error": "not owner"}, status_code=403)

        await conn.execute("DELETE FROM gallery_favorites WHERE artwork_id = $1", artwork_id)
        await conn.execute("DELETE FROM gallery_comments WHERE artwork_id = $1", artwork_id)
        await conn.execute("DELETE FROM gallery_artworks WHERE id = $1", artwork_id)

    image_url = row["image_url"] or ""
    prefix = "/uploads/gallery/"
    if image_url.startswith(prefix):
        filename = image_url[len(prefix):]
        if filename and "/" not in filename and ".." not in filename:
            try:
                (GALLERY_UPLOAD_DIR / filename).unlink(missing_ok=True)
            except OSError:
                pass

    return JSONResponse({"ok": True})


async def gallery_add_favorite(request: Request) -> JSONResponse:
    """POST /gallery/artworks/{artwork_id}/favorites — add favourite."""
    artwork_id = request.path_params["artwork_id"]
    user_id = request.query_params.get("user_id", "")
    if not user_id:
        return JSONResponse({"error": "user_id required"}, status_code=400)

    pool = await _get_pool()
    fav_id = str(uuid.uuid4())
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO gallery_favorites (id, artwork_id, user_id) "
            "VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
            fav_id, artwork_id, user_id,
        )
        await conn.execute(
            "UPDATE gallery_artworks SET favorite_count = favorite_count + 1 WHERE id = $1",
            artwork_id,
        )
    return JSONResponse({"ok": True})


async def gallery_remove_favorite(request: Request) -> JSONResponse:
    """DELETE /gallery/artworks/{artwork_id}/favorites — remove favourite."""
    artwork_id = request.path_params["artwork_id"]
    user_id = request.query_params.get("user_id", "")
    if not user_id:
        return JSONResponse({"error": "user_id required"}, status_code=400)

    pool = await _get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM gallery_favorites WHERE artwork_id = $1 AND user_id = $2",
            artwork_id, user_id,
        )
        if result and result.split()[-1] != "0":
            await conn.execute(
                "UPDATE gallery_artworks SET favorite_count = GREATEST(favorite_count - 1, 0) WHERE id = $1",
                artwork_id,
            )
    return JSONResponse({"ok": True})


async def gallery_user_favorites(request: Request) -> JSONResponse:
    """GET /gallery/user/{user_id}/artwork-favorites — user's favourites."""
    user_id = request.path_params["user_id"]
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT artwork_id, created_at FROM gallery_favorites "
            "WHERE user_id = $1 ORDER BY created_at DESC",
            user_id,
        )
    return JSONResponse([_serialise(r) for r in rows])


async def gallery_user_favorites_legacy(request: Request) -> JSONResponse:
    """GET /gallery/users/{user_id}/favorites — legacy endpoint."""
    user_id = request.path_params["user_id"]
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT artwork_id, created_at FROM gallery_favorites "
            "WHERE user_id = $1 ORDER BY created_at DESC",
            user_id,
        )
    return JSONResponse([_serialise(r) for r in rows])


async def gallery_tags(request: Request) -> JSONResponse:
    """GET /gallery/tags — popular tags."""
    limit = int(request.query_params.get("limit", "20"))
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT tag, COUNT(*) as count FROM ("
            "  SELECT unnest(tags) as tag FROM gallery_artworks "
            "  WHERE is_public = true AND is_approved = true"
            ") t GROUP BY tag ORDER BY count DESC LIMIT $1",
            limit,
        )
    return JSONResponse([{"tag": r["tag"], "count": r["count"]} for r in rows])


async def gallery_uploads(request: Request) -> JSONResponse:
    """GET /gallery/uploads — user's uploads."""
    user_id = request.query_params.get("user_id", "")
    if not user_id:
        return JSONResponse({"error": "user_id required"}, status_code=400)

    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM gallery_uploads WHERE user_id = $1 ORDER BY created_at DESC",
            user_id,
        )
    return JSONResponse([_serialise(r) for r in rows])


async def gallery_create_artwork(request: Request) -> JSONResponse:
    """POST /gallery/artworks — create a new artwork (upload submission)."""
    pool = await _get_pool()

    # Support both JSON and form-data
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        data: dict[str, Any] = {}
        for key in form:
            val = form[key]
            if hasattr(val, "read"):
                file_bytes = await val.read()
                orig = getattr(val, "filename", "") or ""
                ext = Path(orig).suffix.lower() or ".png"
                if ext not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
                    ext = ".png"
                filename = f"{uuid.uuid4().hex}{ext}"
                GALLERY_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                (GALLERY_UPLOAD_DIR / filename).write_bytes(file_bytes)
                data["image_url"] = f"/uploads/gallery/{filename}"
            else:
                data[key] = str(val)
    else:
        data = await request.json()

    title = data.get("title", "").strip()
    if not title:
        return JSONResponse({"error": "title is required"}, status_code=400)

    artwork_id = str(uuid.uuid4())
    tags_raw = data.get("tags", "")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if isinstance(tags_raw, str) else tags_raw

    year_created = None
    if data.get("year_created"):
        try:
            year_created = int(data["year_created"])
        except (ValueError, TypeError):
            pass

    price = None
    if data.get("price"):
        try:
            price = Decimal(str(data["price"]))
        except Exception:
            pass

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO gallery_artworks (
                id, artist_id, artist_name, title, description,
                medium, style, year_created, image_url,
                is_for_sale, price, currency,
                accepts_commissions, commission_info,
                tags, is_public, is_approved
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11, $12,
                $13, $14,
                $15, true, true
            )""",
            artwork_id,
            data.get("artist_id"),
            data.get("artist_name"),
            title,
            data.get("description"),
            data.get("medium"),
            data.get("style"),
            year_created,
            data.get("image_url", ""),
            data.get("is_for_sale", "false").lower() == "true",
            price,
            data.get("currency", "CAD"),
            data.get("accepts_commissions", "false").lower() == "true",
            data.get("commission_info"),
            tags,
        )

    # Also create an entry in gallery_uploads for the user's upload history
    if data.get("artist_id"):
        upload_id = str(uuid.uuid4())
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO gallery_uploads (id, user_id, title, image_url) VALUES ($1, $2, $3, $4)",
                upload_id, data["artist_id"], title, data.get("image_url", ""),
            )

    return JSONResponse({"ok": True, "id": artwork_id}, status_code=201)


async def gallery_serve_upload(request: Request):
    """GET /uploads/gallery/{filename} — serve an uploaded artwork image."""
    filename = request.path_params.get("filename", "")
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    path = GALLERY_UPLOAD_DIR / filename
    try:
        if not path.resolve().is_relative_to(GALLERY_UPLOAD_DIR.resolve()):
            return JSONResponse({"error": "forbidden"}, status_code=403)
    except (OSError, ValueError):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


async def gallery_saved_collections(request: Request) -> JSONResponse:
    """GET /gallery/collections/saved — get user's saved collections."""
    user_id = request.query_params.get("user_id", "")
    if not user_id:
        return JSONResponse({"error": "user_id required"}, status_code=400)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT collection_id, created_at FROM gallery_saved_collections "
            "WHERE user_id = $1 ORDER BY created_at DESC",
            user_id,
        )
    return JSONResponse([_serialise(r) for r in rows])


async def gallery_save_collection(request: Request) -> JSONResponse:
    """POST /gallery/collections/save — save a collection."""
    user_id = request.query_params.get("user_id", "")
    collection_id = request.query_params.get("collection_id", "")
    if not user_id or not collection_id:
        return JSONResponse({"error": "user_id and collection_id required"}, status_code=400)
    pool = await _get_pool()
    save_id = str(uuid.uuid4())
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO gallery_saved_collections (id, user_id, collection_id) "
            "VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
            save_id, user_id, collection_id,
        )
    return JSONResponse({"ok": True})


async def gallery_unsave_collection(request: Request) -> JSONResponse:
    """DELETE /gallery/collections/save — unsave a collection."""
    user_id = request.query_params.get("user_id", "")
    collection_id = request.query_params.get("collection_id", "")
    if not user_id or not collection_id:
        return JSONResponse({"error": "user_id and collection_id required"}, status_code=400)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM gallery_saved_collections WHERE user_id = $1 AND collection_id = $2",
            user_id, collection_id,
        )
    return JSONResponse({"ok": True})


async def gallery_list_comments(request: Request) -> JSONResponse:
    """GET /gallery/comments?artwork_id=... — list comments for an artwork."""
    artwork_id = request.query_params.get("artwork_id", "")
    if not artwork_id:
        return JSONResponse({"error": "artwork_id required"}, status_code=400)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM gallery_comments WHERE artwork_id = $1 ORDER BY created_at ASC",
            artwork_id,
        )
    return JSONResponse([_serialise(r) for r in rows])


async def gallery_post_comment(request: Request) -> JSONResponse:
    """POST /gallery/comments — post a comment on an artwork."""
    data = await request.json()
    artwork_id = data.get("artwork_id", "").strip()
    user_id = data.get("user_id", "").strip()
    user_name = data.get("user_name", "Anonymous")
    user_avatar = data.get("user_avatar", "")
    body = data.get("body", "").strip()
    parent_id = data.get("parent_id") or None
    if not artwork_id or not body:
        return JSONResponse({"error": "artwork_id and body required"}, status_code=400)
    comment_id = str(uuid.uuid4())
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO gallery_comments (id, artwork_id, user_id, user_name, user_avatar, body, parent_id)
               VALUES ($1, $2, $3, $4, $5, $6, $7)""",
            comment_id, artwork_id, user_id, user_name, user_avatar, body, parent_id,
        )
        # Increment comment count on artwork
        await conn.execute(
            "UPDATE gallery_artworks SET comments = comments + 1 WHERE id = $1",
            artwork_id,
        )
    return JSONResponse({"ok": True, "id": comment_id}, status_code=201)


async def gallery_edit_comment(request: Request) -> JSONResponse:
    """PUT /gallery/comments — edit own comment."""
    data = await request.json()
    comment_id = data.get("id", "").strip()
    user_id = data.get("user_id", "").strip()
    body = data.get("body", "").strip()
    if not comment_id or not user_id or not body:
        return JSONResponse({"error": "id, user_id, and body required"}, status_code=400)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM gallery_comments WHERE id = $1 AND user_id = $2",
            comment_id, user_id,
        )
        if not row:
            return JSONResponse({"error": "not found or not owner"}, status_code=404)
        await conn.execute(
            "UPDATE gallery_comments SET body = $1, edited = true, updated_at = CURRENT_TIMESTAMP WHERE id = $2",
            body, comment_id,
        )
    return JSONResponse({"ok": True})


async def gallery_delete_comment(request: Request) -> JSONResponse:
    """DELETE /gallery/comments?id=...&user_id=... — delete own comment."""
    comment_id = request.query_params.get("id", "")
    user_id = request.query_params.get("user_id", "")
    if not comment_id or not user_id:
        return JSONResponse({"error": "id and user_id required"}, status_code=400)
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT artwork_id FROM gallery_comments WHERE id = $1 AND user_id = $2",
            comment_id, user_id,
        )
        if not row:
            return JSONResponse({"error": "not found or not owner"}, status_code=404)
        await conn.execute("DELETE FROM gallery_comments WHERE id = $1", comment_id)
        await conn.execute(
            "UPDATE gallery_artworks SET comments = GREATEST(comments - 1, 0) WHERE id = $1",
            row["artwork_id"],
        )
    return JSONResponse({"ok": True})


AVATAR_UPLOAD_DIR = Path(
    os.environ.get(
        "STREET_PROFILE_AVATAR_DIR",
        str(Path.home() / ".nanobot" / "workspace" / "avatars"),
    )
)

PORTFOLIO_IMAGE_DIR = Path(
    os.environ.get(
        "STREET_PROFILE_PORTFOLIO_DIR",
        str(Path.home() / ".nanobot" / "workspace" / "portfolio"),
    )
)

BANNER_IMAGE_DIR = Path(
    os.environ.get(
        "STREET_PROFILE_BANNER_DIR",
        str(Path.home() / ".nanobot" / "workspace" / "banners"),
    )
)


async def street_profile_upload_banner(request: Request) -> JSONResponse:
    """POST /street-profiles/banner?user_id=... — upload and persist a banner image."""
    user_id = request.query_params.get("user_id", "").strip()
    if not user_id:
        return JSONResponse({"error": "user_id required"}, status_code=400)
    ct = request.headers.get("content-type", "")
    if "multipart/form-data" not in ct:
        return JSONResponse({"error": "multipart/form-data required"}, status_code=400)
    form = await request.form()
    file = form.get("image") or form.get("banner") or form.get("file")
    if not file or not hasattr(file, "read"):
        return JSONResponse({"error": "image file required"}, status_code=400)
    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        return JSONResponse({"error": "image too large (max 20MB)"}, status_code=413)
    orig = getattr(file, "filename", "") or ""
    ext = Path(orig).suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        ext = ".png"
    BANNER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{user_id}-{uuid.uuid4().hex}{ext}"
    (BANNER_IMAGE_DIR / filename).write_bytes(file_bytes)
    banner_url = f"/uploads/banners/{filename}"

    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE users SET banner_url = $1, updated_at = NOW() "
            "WHERE id = $2 RETURNING id, banner_url",
            banner_url,
            user_id,
        )
    if row is None:
        return JSONResponse({"error": "user not found"}, status_code=404)
    return JSONResponse({"ok": True, **dict(row)})


async def street_profile_serve_banner(request: Request):
    """GET /uploads/banners/{filename}"""
    filename = request.path_params.get("filename", "")
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    path = BANNER_IMAGE_DIR / filename
    try:
        if not path.resolve().is_relative_to(BANNER_IMAGE_DIR.resolve()):
            return JSONResponse({"error": "forbidden"}, status_code=403)
    except (OSError, ValueError):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


async def street_profile_upload_portfolio_image(request: Request) -> JSONResponse:
    """POST /street-profiles/portfolio-image?user_id=... — upload a portfolio image."""
    user_id = request.query_params.get("user_id", "").strip()
    if not user_id:
        return JSONResponse({"error": "user_id required"}, status_code=400)
    ct = request.headers.get("content-type", "")
    if "multipart/form-data" not in ct:
        return JSONResponse({"error": "multipart/form-data required"}, status_code=400)
    form = await request.form()
    file = form.get("image") or form.get("file")
    if not file or not hasattr(file, "read"):
        return JSONResponse({"error": "image file required"}, status_code=400)
    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        return JSONResponse({"error": "image too large (max 20MB)"}, status_code=413)
    orig = getattr(file, "filename", "") or ""
    ext = Path(orig).suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        ext = ".png"
    PORTFOLIO_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{user_id}-{uuid.uuid4().hex}{ext}"
    (PORTFOLIO_IMAGE_DIR / filename).write_bytes(file_bytes)
    url = f"/uploads/portfolio/{filename}"
    return JSONResponse({"ok": True, "url": url})


async def street_profile_serve_portfolio_image(request: Request):
    """GET /uploads/portfolio/{filename}"""
    filename = request.path_params.get("filename", "")
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    path = PORTFOLIO_IMAGE_DIR / filename
    try:
        if not path.resolve().is_relative_to(PORTFOLIO_IMAGE_DIR.resolve()):
            return JSONResponse({"error": "forbidden"}, status_code=403)
    except (OSError, ValueError):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


async def street_profile_save_portfolio(request: Request) -> JSONResponse:
    """PUT /street-profiles/portfolio?user_id=... — replace portfolio items list."""
    user_id = request.query_params.get("user_id", "").strip()
    if not user_id:
        return JSONResponse({"error": "user_id required"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    items = body.get("items") if isinstance(body, dict) else body
    if not isinstance(items, list):
        return JSONResponse({"error": "items must be an array"}, status_code=400)
    if len(items) > 200:
        return JSONResponse({"error": "too many items (max 200)"}, status_code=400)

    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE users SET portfolio_items = $1::jsonb, updated_at = NOW() "
            "WHERE id = $2 RETURNING id, portfolio_items",
            json.dumps(items),
            user_id,
        )
    if row is None:
        return JSONResponse({"error": "user not found"}, status_code=404)
    saved = row["portfolio_items"]
    if isinstance(saved, str):
        try:
            saved = json.loads(saved)
        except (json.JSONDecodeError, TypeError):
            saved = []
    return JSONResponse({"ok": True, "items": saved})


async def street_profile_upload_avatar(request: Request) -> JSONResponse:
    """POST /street-profiles/avatar?user_id=... — upload and persist an avatar."""
    user_id = request.query_params.get("user_id", "").strip()
    if not user_id:
        return JSONResponse({"error": "user_id required"}, status_code=400)

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        return JSONResponse({"error": "multipart/form-data required"}, status_code=400)

    form = await request.form()
    file = form.get("image") or form.get("avatar") or form.get("file")
    if not file or not hasattr(file, "read"):
        return JSONResponse({"error": "image file required"}, status_code=400)

    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        return JSONResponse({"error": "image too large (max 10MB)"}, status_code=413)

    orig = getattr(file, "filename", "") or ""
    ext = Path(orig).suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        ext = ".png"

    AVATAR_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{user_id}-{uuid.uuid4().hex}{ext}"
    (AVATAR_UPLOAD_DIR / filename).write_bytes(file_bytes)
    avatar_url = f"/uploads/avatars/{filename}"

    hint_display_name = (form.get("display_name") or "").strip() if hasattr(form, "get") else ""
    hint_email = (form.get("email") or "").strip() if hasattr(form, "get") else ""
    hint_username = (form.get("username") or "").strip() if hasattr(form, "get") else ""

    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "UPDATE users SET avatar_url = $1, updated_at = NOW() "
            "WHERE id = $2 RETURNING id, username, display_name, avatar_url",
            avatar_url,
            user_id,
        )
        if row is None:
            # Auto-create a minimal social row so freshly-registered users can
            # upload an avatar before completing their Street Profile setup.
            short = user_id[:8]
            fallback_username = hint_username or f"user_{short}"
            fallback_email = hint_email or f"{user_id}@local.streetvoices"
            fallback_display_name = hint_display_name or "New User"
            fallback_casdoor_id = f"local_{user_id}"
            try:
                row = await conn.fetchrow(
                    """INSERT INTO users (
                        id, casdoor_id, username, display_name, email,
                        avatar_url, profile_complete, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, false, NOW())
                    RETURNING id, username, display_name, avatar_url""",
                    user_id,
                    fallback_casdoor_id,
                    fallback_username,
                    fallback_display_name,
                    fallback_email,
                    avatar_url,
                )
            except asyncpg.UniqueViolationError:
                # Username or email collision — retry with a disambiguator.
                unique_suffix = uuid.uuid4().hex[:6]
                row = await conn.fetchrow(
                    """INSERT INTO users (
                        id, casdoor_id, username, display_name, email,
                        avatar_url, profile_complete, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, false, NOW())
                    RETURNING id, username, display_name, avatar_url""",
                    user_id,
                    fallback_casdoor_id,
                    f"{fallback_username}_{unique_suffix}",
                    fallback_display_name,
                    f"{unique_suffix}-{fallback_email}",
                    avatar_url,
                )

    return JSONResponse({"ok": True, **dict(row)})


async def street_profile_serve_avatar(request: Request):
    """GET /uploads/avatars/{filename}"""
    filename = request.path_params.get("filename", "")
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "invalid filename"}, status_code=400)
    path = AVATAR_UPLOAD_DIR / filename
    try:
        if not path.resolve().is_relative_to(AVATAR_UPLOAD_DIR.resolve()):
            return JSONResponse({"error": "forbidden"}, status_code=403)
    except (OSError, ValueError):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


async def street_profile_by_username(request: Request) -> JSONResponse:
    """GET /street-profiles/{username} — fetch a profile for the Creative Profile page."""
    username = request.path_params.get("username", "").strip()
    if not username:
        return JSONResponse({"error": "username required"}, status_code=400)

    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, username, display_name, email, avatar_url, banner_url, bio, "
            "location, website, creative_types, social_links, portfolio_items, "
            "profile_views, is_public, created_at, updated_at "
            "FROM users WHERE username = $1",
            username,
        )
        if row is None:
            return JSONResponse({"error": "profile not found"}, status_code=404)
        uid = row["id"]
        follower_count = await conn.fetchval(
            "SELECT count(*)::int FROM follows WHERE following_id = $1", uid
        )
        following_count = await conn.fetchval(
            "SELECT count(*)::int FROM follows WHERE follower_id = $1", uid
        )

    data = _serialise(row)
    uid = data.get("id") or ""
    social_links = data.get("social_links") or {}
    if isinstance(social_links, str):
        try:
            social_links = json.loads(social_links)
        except (json.JSONDecodeError, TypeError):
            social_links = {}

    profile_views = int(data.get("profile_views") or 0)

    portfolio_items = data.get("portfolio_items") or []
    if isinstance(portfolio_items, str):
        try:
            portfolio_items = json.loads(portfolio_items)
        except (json.JSONDecodeError, TypeError):
            portfolio_items = []
    if not isinstance(portfolio_items, list):
        portfolio_items = []

    profile = {
        "id": uid,
        "user_id": uid,
        "username": data.get("username") or "",
        "display_name": data.get("display_name") or data.get("username") or "",
        "primary_roles": data.get("creative_types") or [],
        "secondary_skills": [],
        "bio": data.get("bio"),
        "tagline": None,
        "avatar_url": data.get("avatar_url"),
        "cover_url": data.get("banner_url"),
        "banner_url": data.get("banner_url"),
        "city": data.get("location"),
        "country": None,
        "location_display": data.get("location"),
        "website": data.get("website"),
        "social_links": social_links,
        "availability_status": "open",
        "open_to": [],
        "is_public": data.get("is_public", True),
        "is_featured": False,
        "is_verified": False,
        "creative_types": data.get("creative_types") or [],
        "portfolio_items": portfolio_items,
        "profile_views": profile_views,
        "follower_count": follower_count or 0,
        "following_count": following_count or 0,
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
    }
    return JSONResponse(profile)


async def booking_confirm(request: Request) -> JSONResponse:
    """POST /bookings/confirm — send booking confirmation emails.

    JSON body:
      artist_username    required — looked up in users table to get artist email/display name
      client_name        required
      client_email       required
      service_name       required (e.g. "Mural Commission")
      service_type       optional (e.g. "Video Call", "In Person")
      booking_date       required (human-readable string like "Thursday, April 23, 2026")
      booking_time       required (e.g. "11:00 AM")
      message            optional — client's message to the artist
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be an object"}, status_code=400)

    required = ["artist_username", "client_name", "client_email", "service_name",
                "booking_date", "booking_time"]
    missing = [f for f in required if not str(body.get(f) or "").strip()]
    if missing:
        return JSONResponse(
            {"error": f"missing fields: {', '.join(missing)}"}, status_code=400
        )

    artist_username = str(body["artist_username"]).strip()
    client_name = str(body["client_name"]).strip()
    client_email = str(body["client_email"]).strip()
    service_name = str(body["service_name"]).strip()
    service_type = str(body.get("service_type") or "Consultation").strip()
    booking_date = str(body["booking_date"]).strip()
    booking_time = str(body["booking_time"]).strip()
    client_message = str(body.get("message") or "").strip()

    # Look up the artist so we can email them and include their display name.
    pool = await _get_pool()
    async with pool.acquire() as conn:
        artist_row = await conn.fetchrow(
            "SELECT id, username, display_name, email FROM users WHERE username = $1",
            artist_username,
        )
    artist_display = (artist_row and artist_row["display_name"]) or artist_username
    artist_email = artist_row and artist_row["email"]

    # ── Client confirmation ──
    client_subject = f"Your booking with {artist_display} is confirmed"
    client_text = (
        f"Hi {client_name},\n\n"
        f"Your consultation with {artist_display} has been scheduled.\n\n"
        f"Service: {service_name}\n"
        f"Type: {service_type}\n"
        f"Date: {booking_date}\n"
        f"Time: {booking_time}\n\n"
        f"{artist_display} will confirm your booking within 48 hours. You can "
        f"cancel up to 24 hours before your appointment for a full refund.\n\n"
        f"— Street Voices"
    )
    client_html = f"""<!doctype html><html><body style="font-family:Inter,Arial,sans-serif;background:#0f0f14;color:#f1f1f4;padding:32px;">
      <div style="max-width:520px;margin:0 auto;background:#1a1a22;border-radius:16px;padding:32px;border:1px solid rgba(255,255,255,0.08);">
        <h1 style="color:#FFD700;margin:0 0 8px;">Booking Confirmed</h1>
        <p style="margin:0 0 24px;color:#c5c5cc;">Your consultation with <strong style="color:#fff;">{artist_display}</strong> has been scheduled.</p>
        <table style="width:100%;border-collapse:collapse;">
          <tr><td style="padding:8px 0;color:#9a9aa1;">Service</td><td style="padding:8px 0;text-align:right;color:#fff;">{service_name}</td></tr>
          <tr><td style="padding:8px 0;color:#9a9aa1;">Type</td><td style="padding:8px 0;text-align:right;color:#fff;">{service_type}</td></tr>
          <tr><td style="padding:8px 0;color:#9a9aa1;">Date</td><td style="padding:8px 0;text-align:right;color:#fff;">{booking_date}</td></tr>
          <tr><td style="padding:8px 0;color:#9a9aa1;">Time</td><td style="padding:8px 0;text-align:right;color:#fff;">{booking_time}</td></tr>
        </table>
        <p style="margin:24px 0 0;color:#9a9aa1;font-size:13px;">{artist_display} will confirm within 48 hours. Cancel up to 24 hours before for a full refund.</p>
      </div>
    </body></html>"""

    client_sent = await _send_email(client_email, client_subject, client_text, client_html)

    # ── Artist notification (if we have their email) ──
    artist_sent = False
    if artist_email:
        artist_subject = f"New booking request — {service_name} with {client_name}"
        artist_text = (
            f"Hi {artist_display},\n\n"
            f"{client_name} ({client_email}) requested a {service_name} "
            f"({service_type}) on {booking_date} at {booking_time}.\n\n"
        )
        if client_message:
            artist_text += f"Their message:\n{client_message}\n\n"
        artist_text += (
            "Log in to Street Voices to confirm or reschedule.\n\n"
            "— Street Voices"
        )
        artist_sent = await _send_email(artist_email, artist_subject, artist_text)

    return JSONResponse({
        "ok": bool(client_sent),
        "client_email_sent": client_sent,
        "artist_email_sent": artist_sent,
    })


async def street_profile_record_view(request: Request) -> JSONResponse:
    """POST /street-profiles/{username}/view — increment view counter.

    Skipped if the caller is the profile owner (query param viewer_id matches
    the profile's user id).  No-op for unknown usernames.
    """
    username = request.path_params.get("username", "").strip()
    viewer_id = request.query_params.get("viewer_id", "").strip()
    if not username:
        return JSONResponse({"error": "username required"}, status_code=400)

    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, profile_views FROM users WHERE username = $1", username
        )
        if row is None:
            return JSONResponse({"error": "profile not found"}, status_code=404)
        if viewer_id and viewer_id == row["id"]:
            return JSONResponse({"ok": True, "profile_views": row["profile_views"], "self": True})
        updated = await conn.fetchval(
            "UPDATE users SET profile_views = profile_views + 1 "
            "WHERE id = $1 RETURNING profile_views",
            row["id"],
        )
    return JSONResponse({"ok": True, "profile_views": updated or 0, "self": False})


async def street_profiles_batch_lookup(request: Request) -> JSONResponse:
    """POST /street-profiles/batch-lookup — batch user profile lookup."""
    body = await request.json()
    user_ids = body.get("user_ids", [])
    if not user_ids:
        return JSONResponse({})

    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id as user_id, username, display_name, avatar_url "
            "FROM users WHERE id = ANY($1::text[])",
            user_ids,
        )

    result: dict[str, Any] = {}
    for r in rows:
        profile = _serialise(r)
        profile.setdefault("is_verified", False)
        profile.setdefault("is_featured", False)
        profile.setdefault("primary_roles", [])
        profile.setdefault("city", None)
        profile.setdefault("country", None)
        result[profile["user_id"]] = profile

    return JSONResponse(result)
