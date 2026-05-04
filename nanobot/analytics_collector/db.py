"""Postgres pool — mirrors the pattern used by nanobot/gallery_api.py."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import asyncpg


_pool: asyncpg.Pool | None = None


async def get_pool(db_url: str) -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            db_url,
            min_size=1,
            max_size=10,
            command_timeout=15,
            server_settings={"search_path": "analytics, public"},
        )
        async with _pool.acquire() as conn:
            await conn.set_type_codec(
                "jsonb",
                encoder=json.dumps,
                decoder=json.loads,
                schema="pg_catalog",
            )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def serialise(row: asyncpg.Record | dict[str, Any]) -> dict[str, Any]:
    """JSON-safe serialiser for asyncpg records — handles datetime + Decimal."""
    out: dict[str, Any] = {}
    src: dict[str, Any] = dict(row) if isinstance(row, asyncpg.Record) else row
    for key, val in src.items():
        if isinstance(val, datetime):
            out[key] = val.isoformat()
        elif isinstance(val, date):
            out[key] = val.isoformat()
        elif isinstance(val, UUID):
            out[key] = str(val)
        elif isinstance(val, Decimal):
            out[key] = float(val)
        elif isinstance(val, (set, frozenset)):
            out[key] = list(val)
        else:
            out[key] = val
    return out
