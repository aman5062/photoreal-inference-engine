# generate.py — Stable Diffusion 1.5 single-stage inference
import numpy as np


def generate(pipe, positive: str, negative: str, params: dict) -> np.ndarray:
    """
    Run a single-stage Stable Diffusion 1.5 inference pass.

    Produces a 512×512 image in ~20 steps — much faster and more
    memory-efficient than the two-stage SDXL pipeline.

    cfg_scale is derived from params['intensity'] (0-1) → cfg 4-12.

    Returns:
        uint8 NumPy array of shape (512, 512, 3)
    """
    # Map intensity [0, 1] → cfg_scale [4, 12]
    cfg = 4.0 + params["intensity"] * 8.0

    image = pipe(
        prompt=positive,
        negative_prompt=negative,
        num_inference_steps=20,
        guidance_scale=cfg,
        height=512,
        width=512,
    ).images[0]

    return np.array(image, dtype=np.uint8)
