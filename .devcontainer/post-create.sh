#!/usr/bin/env bash
set -euo pipefail

echo "Setting up UniSage DevContainer..."

# Install python dependencies via uv or pip
if command -v uv >/dev/null 2>&1; then
    uv sync
else
    pip install -e .[dev]
fi

echo "DevContainer setup completed successfully!"
