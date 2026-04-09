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
    title="Photoreal Inference Engine",
    description="TinyLlama + SDXL Base + Refiner photorealistic image generation",
    version="2.0",
)

_llm = None
_base = None
_refiner = None


@app.on_event("startup")
async def startup_event():
    """Load all models once at startup so every request is fast."""
    global _llm, _base, _refiner
    print("[startup] Loading models …")
    t = time.time()
    _llm, _base, _refiner = load_models()
    print(f"[startup] Models ready in {time.time() - t:.1f}s")


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
    Generate a 1024×1024 photorealistic PNG from a text prompt.

    Returns the PNG file as binary response (Content-Type: image/png).
    Also saves it to the outputs/ directory for persistence.
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="'prompt' must not be empty.")

    if _llm is None or _base is None or _refiner is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    try:
        t_total = time.time()

        # ------------------------------------------------- TinyLlama → JSON
        params = get_params(_llm, req.prompt)

        # ------------------------------------------------- Build SDXL prompts
        positive, negative = build_prompts(params)

        # ------------------------------------------------- SDXL inference
        arr = generate(_base, _refiner, positive, negative, params)

        # ------------------------------------------------- Post-processing
        arr = apply_filters(arr, params)

        elapsed = time.time() - t_total
        print(f"[generate] '{req.prompt[:60]}' → {elapsed:.1f}s")

        # ------------------------------------------------- Save to disk
        out_dir = Path(req.output_dir)
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

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
