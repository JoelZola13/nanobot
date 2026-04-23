#!/usr/bin/env bash
# Seed Street Gallery sample artworks (ga-001..ga-015).
# Idempotent: safe to re-run (INSERT ... ON CONFLICT DO NOTHING).
#
# Requires the stack to already be up (social-postgres container running).
# The social DB schema must exist first — spin up the stack once so Prisma
# migrations create `gallery_artworks`, then run this.
#
# Usage:
#   bash deploy/seeds/seed-gallery.sh

set -euo pipefail

CONTAINER="${SOCIAL_POSTGRES_CONTAINER:-nanobot-social-postgres}"
SEED_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gallery-samples.sql"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "error: container '${CONTAINER}' is not running" >&2
  echo "start the stack first (docker compose up -d) and retry." >&2
  exit 1
fi

if ! docker exec "${CONTAINER}" psql -U social -d social -tA \
     -c "SELECT 1 FROM information_schema.tables WHERE table_name='gallery_artworks' LIMIT 1" | grep -q 1; then
  echo "error: gallery_artworks table does not exist yet" >&2
  echo "start the stack and wait for the social app to run its migrations, then retry." >&2
  exit 1
fi

echo "seeding sample artworks..."
docker exec -i "${CONTAINER}" psql -U social -d social -v ON_ERROR_STOP=1 < "${SEED_FILE}"

COUNT=$(docker exec "${CONTAINER}" psql -U social -d social -tA \
        -c "SELECT count(*) FROM gallery_artworks WHERE id LIKE 'ga-%'")
echo "done — ${COUNT} sample artworks now in gallery_artworks."
