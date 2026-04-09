# pipeline.py — Model loading and shared resource management
import os
import torch
from diffusers import DiffusionPipeline
from llama_cpp import Llama

# --------------------------------------------------------------------------- #
# Model path configuration                                                      #
# --------------------------------------------------------------------------- #
MODEL_LLAMA = os.environ.get(
    "LLAMA_MODEL_PATH",
    "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
)
MODEL_BASE = os.environ.get(
    "SDXL_BASE_MODEL",
    "stabilityai/stable-diffusion-xl-base-1.0",
)
MODEL_REFINE = os.environ.get(
    "SDXL_REFINER_MODEL",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
)

# --------------------------------------------------------------------------- #
# Device / dtype selection                                                       #
# --------------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def load_models():
    """
    Load all three models required by the pipeline:
      - TinyLlama 1.1B GGUF (Q4_K_M) via llama-cpp-python
      - SDXL Base 1.0        via diffusers
      - SDXL Refiner 1.0     via diffusers (shares VAE + text_encoder_2 with Base)

    Sharing the VAE and text_encoder_2 between Base and Refiner saves ~3 GB VRAM.

    Returns:
        llm    — Llama instance
        base   — StableDiffusionXLPipeline
        refiner— StableDiffusionXLImg2ImgPipeline
    """
    if not os.path.isfile(MODEL_LLAMA):
        raise FileNotFoundError(
            f"TinyLlama GGUF not found at '{MODEL_LLAMA}'.\n"
            "Run: bash download_models.sh   OR   python download_models.py"
        )

    print(f"  Device : {DEVICE.upper()}")
    print(f"  DType  : {DTYPE}")

    # ---------------------------------------------------------------- TinyLlama
    print("  Loading TinyLlama 1.1B …")
    llm = Llama(
        model_path=MODEL_LLAMA,
        n_ctx=512,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )

    # ---------------------------------------------------------------- SDXL Base
    print("  Loading SDXL Base 1.0 …")
    base = DiffusionPipeline.from_pretrained(
        MODEL_BASE,
        torch_dtype=DTYPE,
        use_safetensors=True,
        variant="fp16" if DTYPE == torch.float16 else None,
    ).to(DEVICE)

    # Enable attention slicing on GPU to reduce peak VRAM
    if DEVICE == "cuda":
        base.enable_attention_slicing()

    # ------------------------------------------------------------ SDXL Refiner
    print("  Loading SDXL Refiner 1.0 …")
    refiner = DiffusionPipeline.from_pretrained(
        MODEL_REFINE,
        text_encoder_2=base.text_encoder_2,   # shared — saves ~2 GB VRAM
        vae=base.vae,                          # shared — saves ~1 GB VRAM
        torch_dtype=DTYPE,
        use_safetensors=True,
        variant="fp16" if DTYPE == torch.float16 else None,
    ).to(DEVICE)

    if DEVICE == "cuda":
        refiner.enable_attention_slicing()

    return llm, base, refiner
