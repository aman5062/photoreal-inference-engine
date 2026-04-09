#!/usr/bin/env python3
"""
download_models.py — Download required model files before first run.

Usage:
    python download_models.py              # download TinyLlama GGUF only
    python download_models.py --llama-only # same as above
    python download_models.py --all        # TinyLlama + pre-cache SDXL weights

The SDXL Base and Refiner weights (~12 GB total) are normally downloaded
automatically by diffusers on first inference.  Use --all to pre-fetch them
so the first request isn't slow.
"""
import argparse
import os
import sys
from pathlib import Path


def download_tinyllama(dest_dir: str = "models") -> Path:
    """Download TinyLlama 1.1B Chat Q4_K_M GGUF from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    out_path = dest / filename

    if out_path.exists():
        print(f"[skip] TinyLlama already present: {out_path}")
        return out_path

    print(f"[download] TinyLlama 1.1B Q4_K_M → {out_path} (~670 MB) …")
    hf_hub_download(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename=filename,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"[done] Saved to {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")
    return out_path


def precache_sdxl() -> None:
    """
    Pre-download SDXL Base and Refiner into the HuggingFace cache.
    This avoids a large download during the first inference request.
    """
    import torch
    from diffusers import DiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    base_id = os.environ.get(
        "SDXL_BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"
    )
    refiner_id = os.environ.get(
        "SDXL_REFINER_MODEL", "stabilityai/stable-diffusion-xl-refiner-1.0"
    )

    print(f"[download] SDXL Base ({base_id}) …")
    base = DiffusionPipeline.from_pretrained(
        base_id, torch_dtype=dtype, use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )
    print("[download] SDXL Refiner …")
    DiffusionPipeline.from_pretrained(
        refiner_id,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )
    print("[done] SDXL weights cached.")


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument(
        "--llama-only",
        action="store_true",
        help="Download only the TinyLlama GGUF (default behaviour)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download TinyLlama GGUF AND pre-cache SDXL weights (~13 GB)",
    )
    parser.add_argument(
        "--dest",
        default=os.environ.get("LLAMA_MODEL_DIR", "models"),
        help="Directory to save TinyLlama GGUF (default: models/)",
    )
    args = parser.parse_args()

    dest = args.dest
    # If LLAMA_MODEL_PATH is set to a full file path, derive the directory from it
    llama_path = os.environ.get("LLAMA_MODEL_PATH", "")
    if llama_path:
        dest = str(Path(llama_path).parent)

    download_tinyllama(dest)

    if args.all:
        precache_sdxl()


if __name__ == "__main__":
    main()
