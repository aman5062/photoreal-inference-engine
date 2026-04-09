#!/usr/bin/env python3
"""
download_models.py — Pre-cache the Stable Diffusion 1.5 model weights.

Usage:
    python download_models.py

SD 1.5 weights (~4 GB) are normally downloaded automatically by diffusers
on the first inference request.  Run this script beforehand to pre-fetch
them so the first request isn't slow.
"""
import os
import torch
from diffusers import StableDiffusionPipeline


def precache_sd15() -> None:
    """Pre-download SD 1.5 into the HuggingFace cache."""
    model_id = os.environ.get("SD_MODEL", "runwayml/stable-diffusion-v1-5")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[download] Stable Diffusion 1.5 ({model_id}) …")
    StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    print("[done] SD 1.5 weights cached.")


if __name__ == "__main__":
    precache_sd15()

