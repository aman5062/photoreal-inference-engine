# prompt.py — Convert TinyLlama JSON params into SDXL prompt strings
QUALITY_TAGS = (
    "hyperrealistic, photorealistic, masterpiece, "
    "cinematic lighting, ultra-detailed, 8k resolution, "
    "sharp focus, professional photography, RAW photo, "
    "DSLR, 50mm lens, natural bokeh, award-winning"
)

BASE_NEGATIVE = (
    "blurry, low quality, low resolution, deformed, mutated, "
    "watermark, text, signature, logo, caption, "
    "cartoon, anime, painting, illustration, drawing, "
    "overexposed, underexposed, out of focus, grainy, noisy, "
    "bad anatomy, extra limbs, missing fingers, ugly, worst quality, "
    "jpeg artifacts, compression artifacts, oversaturated"
)


def build_prompts(params: dict) -> tuple[str, str]:
    """
    Compose positive and negative SDXL prompt strings from the
    structured parameter dict produced by TinyLlama.

    Returns:
        positive (str): full positive prompt
        negative (str): full negative prompt
    """
    positive = (
        f"{params['scene']}, "
        f"{params['style']} style, "
        f"{params['mood']}, "
        f"{QUALITY_TAGS}"
    )

    user_negative = params.get("negative", "").strip().rstrip(",")
    negative = f"{user_negative}, {BASE_NEGATIVE}" if user_negative else BASE_NEGATIVE

    return positive, negative
