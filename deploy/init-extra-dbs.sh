#!/bin/bash
# Auto-create extra databases needed by services (e.g. paperclip)
# This runs once when the vectordb container is first initialized.
set -e

for db in ${POSTGRES_DBS//,/ }; do
  echo "Creating database: $db"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-SQL
    SELECT 'CREATE DATABASE "$db"' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\gexec
SQL
done
