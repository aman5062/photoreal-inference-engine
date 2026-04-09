# pipeline.py — Lightweight model loading: Stable Diffusion 1.5 (single pipeline)
import os
import torch
from diffusers import StableDiffusionPipeline

# --------------------------------------------------------------------------- #
# Model configuration                                                           #
# --------------------------------------------------------------------------- #
MODEL_SD = os.environ.get(
    "SD_MODEL",
    "runwayml/stable-diffusion-v1-5",
)

# --------------------------------------------------------------------------- #
# Device / dtype selection                                                       #
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_models():
    """
    Load Stable Diffusion 1.5 — a single lightweight pipeline that runs
    comfortably with 4 GB VRAM (or on CPU for development).

    Returns:
        pipe — StableDiffusionPipeline
    """
    print(f"  Device : {DEVICE.upper()}")
    print(f"  DType  : {DTYPE}")
    print(f"  Loading Stable Diffusion 1.5 ({MODEL_SD}) …")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_SD,
        torch_dtype=DTYPE,
        use_safetensors=True,
    ).to(DEVICE)

    # Reduce peak VRAM usage on GPU
    if DEVICE == "cuda":
        pipe.enable_attention_slicing()

    return pipe
