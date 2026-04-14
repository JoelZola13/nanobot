-- Street Gallery tables
-- Migration: 20260327_gallery

-- ── gallery_artworks ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS "gallery_artworks" (
    "id"                  TEXT PRIMARY KEY,
    "artist_id"           TEXT,
    "artist_name"         TEXT,
    "title"               TEXT NOT NULL,
    "description"         TEXT,
    "medium"              TEXT,
    "style"               TEXT,
    "year_created"        INTEGER,
    "image_url"           TEXT NOT NULL,
    "thumbnail_url"       TEXT,
    "full_resolution_url" TEXT,
    "is_featured"         BOOLEAN NOT NULL DEFAULT false,
    "is_public"           BOOLEAN NOT NULL DEFAULT true,
    "display_order"       INTEGER DEFAULT 0,
    "is_for_sale"         BOOLEAN NOT NULL DEFAULT false,
    "price"               DECIMAL(10,2),
    "currency"            TEXT NOT NULL DEFAULT 'USD',
    "is_sold"             BOOLEAN NOT NULL DEFAULT false,
    "sold_at"             TIMESTAMP(3),
    "accepts_commissions" BOOLEAN NOT NULL DEFAULT false,
    "commission_info"     TEXT,
    "tags"                TEXT[] DEFAULT '{}',
    "collection_name"     TEXT,
    "view_count"          INTEGER NOT NULL DEFAULT 0,
    "favorite_count"      INTEGER NOT NULL DEFAULT 0,
    "comment_count"       INTEGER NOT NULL DEFAULT 0,
    "share_count"         INTEGER NOT NULL DEFAULT 0,
    "is_nsfw"             BOOLEAN NOT NULL DEFAULT false,
    "is_approved"         BOOLEAN NOT NULL DEFAULT true,
    "created_at"          TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at"          TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS "gallery_artworks_medium_idx" ON "gallery_artworks" ("medium");
CREATE INDEX IF NOT EXISTS "gallery_artworks_style_idx" ON "gallery_artworks" ("style");
CREATE INDEX IF NOT EXISTS "gallery_artworks_public_idx" ON "gallery_artworks" ("is_public", "is_approved", "created_at" DESC);
CREATE INDEX IF NOT EXISTS "gallery_artworks_tags_idx" ON "gallery_artworks" USING GIN ("tags");
CREATE INDEX IF NOT EXISTS "gallery_artworks_artist_idx" ON "gallery_artworks" ("artist_id");

-- ── gallery_favorites ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS "gallery_favorites" (
    "id"          TEXT PRIMARY KEY,
    "artwork_id"  TEXT NOT NULL REFERENCES "gallery_artworks"("id") ON DELETE CASCADE,
    "user_id"     TEXT NOT NULL,
    "created_at"  TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS "gallery_favorites_unique" ON "gallery_favorites" ("artwork_id", "user_id");
CREATE INDEX IF NOT EXISTS "gallery_favorites_user_idx" ON "gallery_favorites" ("user_id", "created_at" DESC);

-- ── gallery_uploads ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS "gallery_uploads" (
    "id"          TEXT PRIMARY KEY,
    "user_id"     TEXT NOT NULL,
    "title"       TEXT NOT NULL,
    "image_url"   TEXT NOT NULL,
    "created_at"  TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS "gallery_uploads_user_idx" ON "gallery_uploads" ("user_id", "created_at" DESC);
