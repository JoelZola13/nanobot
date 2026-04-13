"""Street Gallery API – route handlers for the gallery feature.

All handlers are async Starlette request handlers that return JSONResponse.
Database: PostgreSQL via asyncpg, using the social-postgres container.
"""
from __future__ import annotations

import uuid
import json
from datetime import datetime
from decimal import Decimal
from typing import Any

import asyncpg
from starlette.requests import Request
from starlette.responses import JSONResponse

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
                # File upload — for now store a placeholder; real file storage TBD
                file_bytes = await val.read()
                # In production, upload to S3/storage. For now, save locally.
                filename = f"{uuid.uuid4()}.png"
                data["image_url"] = f"/uploads/gallery/{filename}"
                import os
                upload_dir = "/app/uploads/gallery" if os.path.exists("/app/uploads") else "/tmp/gallery_uploads"
                os.makedirs(upload_dir, exist_ok=True)
                with open(os.path.join(upload_dir, filename), "wb") as f:
                    f.write(file_bytes)
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
