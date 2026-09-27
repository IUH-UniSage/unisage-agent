#!/bin/sh
# Runs once, automatically, via the postgres image's docker-entrypoint-initdb.d
# hook - the compose file's postgres service already creates $POSTGRES_DB
# (backend-java's database) from its own env vars; this adds the second,
# separate database unisage-agent uses (no shared schema/tables, no cross-service
# FK - see SPEC-ingestion-resume.md).
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE unisage_agent_db;
EOSQL
