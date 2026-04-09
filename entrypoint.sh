#!/usr/bin/env bash
# entrypoint.sh — Container entrypoint
# Handles two modes:
#   server  (default) — start FastAPI API server on $PORT
#   <anything else>   — pass through to the shell / python directly
#
# If the TinyLlama GGUF is missing, auto-download it before starting.
set -euo pipefail

LLAMA_PATH="${LLAMA_MODEL_PATH:-/app/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
PORT="${PORT:-8000}"

# ── Auto-download TinyLlama if needed ────────────────────────────────────────
if [ ! -f "${LLAMA_PATH}" ]; then
    echo "[entrypoint] TinyLlama GGUF not found at ${LLAMA_PATH}"
    echo "[entrypoint] Downloading TinyLlama 1.1B Q4_K_M (~670 MB) …"
    python /app/download_models.py --llama-only
fi

# ── Dispatch ─────────────────────────────────────────────────────────────────
MODE="${1:-server}"

if [ "${MODE}" = "server" ]; then
    echo "[entrypoint] Starting FastAPI server on port ${PORT} …"
    exec python /app/src/app.py
else
    # Pass all arguments straight through (e.g. python src/main.py "prompt")
    exec "$@"
fi
