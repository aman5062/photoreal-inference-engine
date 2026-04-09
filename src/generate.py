# generate.py — SDXL Base + Refiner two-stage inference
import numpy as np


def generate(base, refiner, positive: str, negative: str, params: dict) -> np.ndarray:
    """
    Run the two-stage SDXL pipeline:
      Stage 1 — Base model (30 steps, denoising_end=0.8, output_type='latent')
      Stage 2 — Refiner  (20 steps, denoising_start=0.8, adds fine detail)

    cfg_scale is derived from params['intensity'] (0-1) → cfg 4-12.
    A higher intensity means the model follows the prompt more strictly.

    Returns:
        uint8 NumPy array of shape (1024, 1024, 3)
    """
    # Map intensity [0, 1] → cfg_scale [4, 12] for a good photorealistic range
    cfg = 4.0 + params["intensity"] * 8.0

    # ------------------------------------------------------------------ Stage 1
    # Base produces a 128×128×4 latent tensor (denoised to 80%)
    latent = base(
        prompt=positive,
        negative_prompt=negative,
        num_inference_steps=30,
        guidance_scale=cfg,
        height=1024,
        width=1024,
        output_type="latent",   # stay in latent space — skip VAE decode here
        denoising_end=0.8,      # hand off at 80% denoising
    ).images

    # ------------------------------------------------------------------ Stage 2
    # Refiner receives the latent and resolves high-frequency detail
    image = refiner(
        prompt=positive,
        negative_prompt=negative,
        num_inference_steps=20,
        guidance_scale=cfg,
        image=latent,
        denoising_start=0.8,    # picks up the remaining 20% noise
    ).images[0]

    # PIL Image → (1024, 1024, 3) uint8 NumPy array
    return np.array(image, dtype=np.uint8)
