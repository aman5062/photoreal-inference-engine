# ============================================================
# Dockerfile — GPU (CUDA 12.1) build
# ============================================================
# Requirements:
#   - NVIDIA GPU with ≥ 4 GB VRAM (6 GB+ recommended)
#   - NVIDIA Container Toolkit installed on host
#   - Docker 20.10+
#
# Build:
#   docker build -t photoreal-engine:gpu .
#
# Run (CLI):
#   docker run --gpus all -v $(pwd)/outputs:/app/outputs \
#     photoreal-engine:gpu python src/main.py "your prompt here"
#
# Run (API server):
#   docker run --gpus all -p 8000:8000 \
#     -v $(pwd)/outputs:/app/outputs \
#     photoreal-engine:gpu
# ============================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# ── System setup ──────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        git \
        wget \
        curl \
        libgomp1 \
        libstdc++6 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ───────────────────────────────────────────
# Install PyTorch first (CUDA 12.1 wheel)
RUN pip install --upgrade pip && \
    pip install torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install \
        diffusers==0.27.2 \
        transformers==4.40.1 \
        accelerate==0.29.3 \
        safetensors==0.4.3 \
        Pillow==10.3.0 \
        "numpy==1.26.4" \
        scipy==1.13.0 \
        fastapi==0.111.0 \
        "uvicorn[standard]==0.29.0" \
        pydantic==2.7.1 \
        huggingface-hub==0.23.0

# ── Application code ──────────────────────────────────────────
WORKDIR /app
COPY src/ ./src/
COPY entrypoint.sh ./entrypoint.sh

RUN chmod +x entrypoint.sh && \
    mkdir -p outputs

# ── Runtime configuration ─────────────────────────────────────
# HuggingFace cache inside the container (override with -v for persistence)
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/.cache/huggingface \
    SD_MODEL=runwayml/stable-diffusion-v1-5 \
    PORT=8000

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
# Default: start the FastAPI server
CMD ["server"]

