#!/bin/sh
# Runs AFTER 10-initial-state.sql.gz restores the snapshot.
# Substitutes the __SV_ROOT_URL__ placeholder with whatever SV_ROOT_URL
# the teammate has in their environment (defaults to http://localhost:9001).
# Runs only once, on a fresh DB volume.

set -e

URL="${SV_ROOT_URL:-http://localhost:9001}"
echo "[sv-seed] replacing __SV_ROOT_URL__ -> $URL"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOF
UPDATE templates
   SET body = REPLACE(body, '__SV_ROOT_URL__', '$URL')
 WHERE body LIKE '%__SV_ROOT_URL__%';

UPDATE settings
   SET value = REPLACE(value::text, '__SV_ROOT_URL__', '$URL')::jsonb
 WHERE value::text LIKE '%__SV_ROOT_URL__%';

UPDATE campaigns
   SET body = REPLACE(body, '__SV_ROOT_URL__', '$URL')
 WHERE body LIKE '%__SV_ROOT_URL__%';
EOF

echo "[sv-seed] done"
