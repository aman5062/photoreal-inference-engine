#!/usr/bin/env bash
# entrypoint.sh — Container entrypoint
# Handles two modes:
#   server  (default) — start FastAPI API server on $PORT
#   <anything else>   — pass through to the shell / python directly
#
# SD 1.5 weights are downloaded automatically by diffusers on first inference.
set -euo pipefail

PORT="${PORT:-8000}"

# ── Dispatch ─────────────────────────────────────────────────────────────────
MODE="${1:-server}"

if [ "${MODE}" = "server" ]; then
    echo "[entrypoint] Starting FastAPI server on port ${PORT} …"
    exec python /app/src/app.py
else
    # Pass all arguments straight through (e.g. python src/main.py "prompt")
    exec "$@"
fi
