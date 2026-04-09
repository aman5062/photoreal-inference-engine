# Photoreal Inference Engine — Lightweight Edition

A fully **offline**, **Docker-runnable** photorealistic image generation pipeline.  
**No API key required.** Generates 512 × 512 PNG images from plain-text prompts.

> **Lightweight branch**: This branch uses **Stable Diffusion 1.5** (single model, ~4 GB)
> and a rule-based prompt parser — no TinyLlama LLM and no SDXL refiner.  
> Requirements are dramatically reduced: 4 GB VRAM minimum, no C++ build toolchain needed.

## Pipeline Architecture

```
User Prompt
    │
    ▼
┌─────────────────┐
│  Rule-based     │  ~0ms   →  params: scene, style, mood, intensity, noise, blur
│  Prompt Parser  │           (keyword matching — no LLM required)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Prompt Builder │  ~0ms   →  positive + negative SD prompt strings
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  SD 1.5         │  ~5-10s →  512×512 RGB image (20 inference steps)
│  20 steps       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  NumPy Filters  │  ~0.1s  →  color grade, sharpen, vignette, grain, contrast
│  (byte-level)   │
└─────────────────┘
    │
    ▼
output.png  (512×512, lossless PNG, ~300–600 KB)
```

## Project Structure

```
photoreal-inference-engine/
├── src/
│   ├── pipeline.py       model loading (SD 1.5)
│   ├── model.py          rule-based prompt parser → structured params
│   ├── prompt.py         params → positive + negative SD prompt strings
│   ├── generate.py       SD 1.5 single-stage inference
│   ├── filters.py        byte-level NumPy/SciPy post-processing
│   ├── main.py           CLI entry point
│   └── app.py            FastAPI HTTP server
├── outputs/              Generated PNGs saved here (volume-mounted)
├── Dockerfile            GPU / CUDA 12.1 build
├── Dockerfile.cpu        CPU-only build
├── docker-compose.yml    GPU deployment
├── docker-compose.cpu.yml  CPU deployment
├── entrypoint.sh         Container entrypoint
├── download_models.py    Optional: pre-cache SD 1.5 weights
└── requirements.txt      Python dependencies
```

## Machine Requirements

| Config | RAM | VRAM | Storage | Speed |
|--------|-----|------|---------|-------|
| CPU only | 8 GB | — | 8 GB | ~5-15 min/image |
| GPU (min) | 8 GB | 4 GB | 8 GB | ~5-10s/image |
| GPU (recommended) | 16 GB | 6 GB+ | 8 GB | ~3-6s/image |

## Quick Start (Docker — GPU)

### 1. Build & start the API server

```bash
docker compose up --build
```

SD 1.5 weights (~4 GB) download automatically from HuggingFace on first startup.

### 2. Generate an image

```bash
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "a stormy ocean at night with lightning and a sinking ship"}' \
     --output result.png
```

The image is also saved to `./outputs/` on the host.

### 3. CLI mode (alternative)

```bash
docker compose run --rm app python src/main.py \
    "a stormy ocean at night with lightning and a sinking ship"
```

## Quick Start (Docker — CPU only)

```bash
# Build CPU image
docker compose -f docker-compose.cpu.yml up --build

# Generate
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "golden hour portrait of a woman in a sunflower field"}' \
     --output result.png
```

## Quick Start (No Docker)

```bash
# 1. Install dependencies (GPU)
pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Run CLI
python src/main.py "a misty mountain valley at sunrise"

# 3. Or start the web server
python src/app.py
```

## API Reference

### `POST /generate`

Generate a 512×512 PNG from a text prompt.

**Request body:**
```json
{
  "prompt": "a cinematic close-up of a wolf in a snowstorm",
  "output_dir": "outputs"
}
```

**Response:** `image/png` binary data (PNG bytes)

**Response headers:**
```
X-Generation-Time: 7.21s
X-Params: {"scene":"...","style":"...","mood":"...","intensity":0.75,...}
X-Output-Path: outputs/a_cinematic_close-up_of_a_wolf_in_a.png
```

### `GET /health`

```json
{"status": "ok", "device": "cuda"}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SD_MODEL` | `runwayml/stable-diffusion-v1-5` | HF model ID or local path |
| `HF_HOME` | `/app/.cache/huggingface` | HuggingFace cache directory |
| `PORT` | `8000` | FastAPI server port |
| `HUGGING_FACE_HUB_TOKEN` | _(unset)_ | Optional HF token for gated models |

## Model Sizes

| Model | Size | Source |
|-------|------|--------|
| Stable Diffusion 1.5 | ~4 GB | runwayml/stable-diffusion-v1-5 |

## Comparison with Full Version

| Feature | Lightweight (this branch) | Full version |
|---------|--------------------------|--------------|
| Image size | 512×512 | 1024×1024 |
| Models | SD 1.5 (~4 GB) | SDXL Base + Refiner (~13 GB) + TinyLlama (~670 MB) |
| VRAM required | 4 GB | 8–12 GB |
| Inference time (GPU) | ~5-10s | ~15-20s |
| Prompt parsing | Rule-based (instant) | TinyLlama LLM (~0.8s) |
| C++ build tools needed | No | Yes (llama-cpp-python) |
