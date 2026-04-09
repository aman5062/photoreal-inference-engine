#!/usr/bin/env python3
# app.py — FastAPI web interface for the photorealistic inference engine
"""
Starts an HTTP server so you can generate images via a simple REST API:

  POST /generate
  Content-Type: application/json
  Body: {"prompt": "a stormy ocean at night with lightning"}

  → Returns: PNG image bytes  (Content-Type: image/png)

  GET /health  → {"status": "ok", "device": "cuda|cpu"}
"""
import io
import os
import sys
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from PIL import Image

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import load_models, DEVICE
from model import get_params
from prompt import build_prompts
from generate import generate
from filters import apply_filters

# --------------------------------------------------------------------------- #
# App & model state                                                              #
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Photoreal Inference Engine (Lightweight)",
    description="SD 1.5 photorealistic image generation — no LLM required",
    version="3.0",
)

_pipe = None

# Absolute, resolved root that all output files must reside within.
# Evaluated once at startup so it is not dependent on the working directory.
_OUTPUT_ROOT = Path("/app/outputs").resolve()


def _safe_output_dir(user_dir: str) -> Path:
    """
    Resolve a user-supplied output directory to a path that is strictly
    inside _OUTPUT_ROOT, preventing path-traversal attacks.

    Any path that would escape the output root is silently replaced with
    _OUTPUT_ROOT itself.
    """
    try:
        candidate = (_OUTPUT_ROOT / user_dir).resolve()
        candidate.relative_to(_OUTPUT_ROOT)   # raises ValueError if outside root
        return candidate
    except (ValueError, Exception):
        return _OUTPUT_ROOT


@app.on_event("startup")
async def startup_event():
    """Load the model once at startup so every request is fast."""
    global _pipe
    print("[startup] Loading model …")
    t = time.time()
    _pipe = load_models()
    print(f"[startup] Model ready in {time.time() - t:.1f}s")


# --------------------------------------------------------------------------- #
# Request / response schemas                                                     #
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    prompt: str
    output_dir: str = "outputs"


# --------------------------------------------------------------------------- #
# Routes                                                                         #
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "device": DEVICE})


@app.post("/generate")
async def generate_image(req: GenerateRequest):
    """
    Generate a 512×512 photorealistic PNG from a text prompt.

    Returns the PNG file as binary response (Content-Type: image/png).
    Also saves it to the outputs/ directory for persistence.
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="'prompt' must not be empty.")

    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        t_total = time.time()

        # ------------------------------------------------- Rule-based params
        params = get_params(req.prompt)

        # ------------------------------------------------- Build SD prompts
        positive, negative = build_prompts(params)

        # ------------------------------------------------- SD 1.5 inference
        arr = generate(_pipe, positive, negative, params)

        # ------------------------------------------------- Post-processing
        arr = apply_filters(arr, params)

        elapsed = time.time() - t_total
        print(f"[generate] '{req.prompt[:60]}' → {elapsed:.1f}s")

        # ------------------------------------------------- Save to disk
        out_dir = _safe_output_dir(req.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = req.prompt[:40].replace(" ", "_").replace("/", "-")
        out_path = out_dir / f"{safe_name}.png"
        Image.fromarray(arr).save(str(out_path), format="PNG", optimize=False)

        # ------------------------------------------------- Return PNG bytes
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", optimize=False)
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="image/png",
            headers={
                "X-Generation-Time": f"{elapsed:.2f}s",
                "X-Params": json.dumps(params, separators=(",", ":")),
                "X-Output-Path": str(out_path),
            },
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
