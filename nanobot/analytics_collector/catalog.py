"""Loaders for the static catalog JSONs (event-catalog.json, product-areas.json).
The collector validates incoming events against the catalog so unknown event
names or missing required props are caught at ingest time.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


_event_catalog: dict[str, Any] | None = None
_product_areas: dict[str, Any] | None = None


def event_catalog() -> dict[str, Any]:
    global _event_catalog
    if _event_catalog is None:
        raw = files(__package__).joinpath("catalog/event-catalog.json").read_text(encoding="utf-8")
        _event_catalog = json.loads(raw)
    return _event_catalog


def product_areas() -> dict[str, Any]:
    global _product_areas
    if _product_areas is None:
        raw = files(__package__).joinpath("catalog/product-areas.json").read_text(encoding="utf-8")
        _product_areas = json.loads(raw)
    return _product_areas


def event_index() -> dict[str, dict[str, Any]]:
    """{event_name: definition}"""
    return {e["name"]: e for e in event_catalog()["events"]}
