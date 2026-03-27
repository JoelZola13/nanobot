#!/bin/bash
# Create the casdoor and social databases in the shared vectordb
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE casdoor;
    GRANT ALL PRIVILEGES ON DATABASE casdoor TO myuser;
    CREATE DATABASE social;
    GRANT ALL PRIVILEGES ON DATABASE social TO myuser;
EOSQL
