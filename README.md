# Photoreal Inference Engine

A fully **offline**, **Docker-runnable** photorealistic image generation pipeline.  
**No API key required.** Generates 1024 × 1024 HD PNG images from plain-text prompts.

## Pipeline Architecture

```
User Prompt
    │
    ▼
┌─────────────────┐
│  TinyLlama 1.1B │  ~0.8s  →  JSON: scene, style, mood, intensity, noise, blur
│  (GGUF Q4_K_M)  │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Prompt Builder │  ~0.001s → positive + negative SDXL strings
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  SDXL Base 1.0  │  ~6-8s  →  128×128×4 latent tensor (denoising_end=0.8)
│  30 steps       │
│  DPM++ 2M Karras│
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ SDXL Refiner 1.0│  ~3-4s  →  high-frequency detail pass (denoising_start=0.8)
│  20 steps       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  VAE Decoder    │  ~0.5s  →  1024×1024×3 uint8 array
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  NumPy Filters  │  ~0.1s  →  color grade, sharpen, vignette, grain, contrast
│  (byte-level)   │
└─────────────────┘
    │
    ▼
output.png  (1024×1024, lossless PNG, ~800 KB–2 MB)
```

## Project Structure

```
photoreal-inference-engine/
├── src/
│   ├── pipeline.py       model loading & shared resource management
│   ├── model.py          TinyLlama inference → structured JSON params
│   ├── prompt.py         JSON → positive + negative SDXL prompt strings
│   ├── generate.py       SDXL Base + Refiner two-stage inference
│   ├── filters.py        byte-level NumPy/SciPy post-processing
│   ├── main.py           CLI entry point
│   └── app.py            FastAPI HTTP server
├── models/               TinyLlama GGUF goes here (volume-mounted)
├── outputs/              Generated PNGs saved here (volume-mounted)
├── Dockerfile            GPU / CUDA 12.1 build
├── Dockerfile.cpu        CPU-only build
├── docker-compose.yml    GPU deployment
├── docker-compose.cpu.yml  CPU deployment
├── entrypoint.sh         Container entrypoint (auto-downloads TinyLlama)
├── download_models.py    Model download helper script
└── requirements.txt      Python dependencies
```

## Machine Requirements

| Config | RAM | VRAM | Storage | Speed |
|--------|-----|------|---------|-------|
| CPU only | 16 GB | — | 30 GB | ~5-15 min/image |
| GPU (min) | 16 GB | 8 GB | 30 GB | ~15-20s/image |
| GPU (recommended) | 32 GB | 12 GB+ | 30 GB | ~10-15s/image |

## Quick Start (Docker — GPU)

### 1. Download TinyLlama

```bash
python download_models.py
```

This saves `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (~670 MB) into `models/`.

### 2. Build & start the API server

```bash
docker compose up --build
```

SDXL Base + Refiner (~12 GB) download automatically from HuggingFace on first startup.

### 3. Generate an image

```bash
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "a stormy ocean at night with lightning and a sinking ship"}' \
     --output result.png
```

The image is also saved to `./outputs/` on the host.

### 4. CLI mode (alternative)

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

# 2. Download TinyLlama
python download_models.py

# 3. Run CLI
python src/main.py "a misty mountain valley at sunrise"

# 4. Or start the web server
python src/app.py
```

## API Reference

### `POST /generate`

Generate a 1024×1024 PNG from a text prompt.

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
X-Generation-Time: 14.32s
X-Params: {"scene":"...","style":"...","mood":"...","intensity":0.85,...}
X-Output-Path: outputs/a_cinematic_close-up_of_a_wolf_in_a.png
```

### `GET /health`

```json
{"status": "ok", "device": "cuda"}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_MODEL_PATH` | `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | Path to TinyLlama GGUF |
| `SDXL_BASE_MODEL` | `stabilityai/stable-diffusion-xl-base-1.0` | HF model ID or local path |
| `SDXL_REFINER_MODEL` | `stabilityai/stable-diffusion-xl-refiner-1.0` | HF model ID or local path |
| `HF_HOME` | `/app/.cache/huggingface` | HuggingFace cache directory |
| `PORT` | `8000` | FastAPI server port |
| `HUGGING_FACE_HUB_TOKEN` | _(unset)_ | Optional HF token for gated models |

## Model Sizes

| Model | Size | Source |
|-------|------|--------|
| TinyLlama 1.1B Q4_K_M | 670 MB | TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF |
| SDXL Base 1.0 | 6.5 GB | stabilityai/stable-diffusion-xl-base-1.0 |
| SDXL Refiner 1.0 | 6.1 GB | stabilityai/stable-diffusion-xl-refiner-1.0 |

> **Tip:** Base and Refiner share `text_encoder_2` and `vae` — this saves ~3 GB VRAM at runtime.