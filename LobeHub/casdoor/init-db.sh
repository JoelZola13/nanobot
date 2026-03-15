#!/bin/bash
# Create the casdoor database alongside the lobehub database
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE casdoor;
    GRANT ALL PRIVILEGES ON DATABASE casdoor TO lobehub;
EOSQL
