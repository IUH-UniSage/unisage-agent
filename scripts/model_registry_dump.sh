#!/bin/sh
# Prints SQL that copies the AI model registry (chat_models: chat/embedding/rerank models with their
# encrypted API keys, and model_prices) of backend-java database "$1" into another backend-java
# database. Runs INSIDE the Postgres container (`task db:models:export` pipes it in), so the host
# needs no psql and the same command works on Windows, macOS and Linux.
#
# The generated SQL:
# - copies through temp tables with an explicit column list (column order may differ between DBs);
# - sets created_by/updated_by (whichever the table has) to NULL: user UUIDs differ per database;
# - inserts with ON CONFLICT DO NOTHING: rows already there (same id, same provider/model, or a
#   second ACTIVE embedding model) are skipped, so re-running is safe;
# - bumps model_registry_version so a running agent reloads the registry.
set -eu
src="$1"

q() { psql -U postgres -d "$src" -v ON_ERROR_STOP=1 -Atc "$1"; }

echo "-- Model registry copied from database $src at $(date -u +%Y-%m-%dT%H:%M:%SZ) (task db:models:export)."
echo "-- Holds API keys encrypted with APP_ENCRYPTION_KEY: never commit this file."
echo "BEGIN;"
for table in chat_models model_prices; do
    columns=$(q "SELECT string_agg(quote_ident(column_name), ', ' ORDER BY ordinal_position)
                 FROM information_schema.columns
                 WHERE table_schema = 'public' AND table_name = '$table'")
    echo "CREATE TEMP TABLE src_$table (LIKE public.$table) ON COMMIT DROP;"
    echo "COPY src_$table ($columns) FROM STDIN;"
    q "COPY public.$table ($columns) TO STDOUT"
    printf '%s\n' '\.'
    audit=$(q "SELECT string_agg(quote_ident(column_name) || ' = NULL', ', ')
               FROM information_schema.columns
               WHERE table_schema = 'public' AND table_name = '$table'
                 AND column_name IN ('created_by', 'updated_by')")
    if [ -n "$audit" ]; then
        echo "UPDATE src_$table SET $audit;"
    fi
    echo "INSERT INTO public.$table ($columns) SELECT $columns FROM src_$table ON CONFLICT DO NOTHING;"
done
echo "UPDATE public.model_registry_version SET version = version + 1, updated_at = now() WHERE id = 1;"
echo "COMMIT;"
