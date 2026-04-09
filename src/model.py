# model.py — Rule-based prompt parser: converts a user prompt into structured params
#            (no LLM required — lightweight replacement for TinyLlama)

# Keywords used for style and mood detection
_STYLE_KEYWORDS = {
    "portrait": "portrait photography",
    "landscape": "landscape photography",
    "macro": "macro photography",
    "street": "street photography",
    "aerial": "aerial photography",
    "underwater": "underwater photography",
    "architecture": "architectural photography",
    "wildlife": "wildlife photography",
    "food": "food photography",
    "product": "product photography",
}

_MOOD_KEYWORDS = {
    "night": "night lighting, dark atmosphere",
    "sunset": "golden hour, warm sunset lighting",
    "sunrise": "golden hour, soft sunrise lighting",
    "storm": "dramatic stormy lighting, dark clouds",
    "fog": "foggy, misty atmosphere",
    "sunny": "bright daylight, natural sunlight",
    "overcast": "soft diffused lighting, overcast sky",
    "indoor": "indoor lighting, ambient light",
    "studio": "studio lighting, controlled light",
}

DEFAULT_PARAMS = {
    "scene": "photorealistic scene",
    "style": "dramatic cinematic photography",
    "mood": "cinematic, moody lighting",
    "intensity": 0.75,
    "negative": "blurry, low quality, cartoon, anime, watermark",
    "noise": 0.08,
    "blur": 0.02,
}


def get_params(prompt: str) -> dict:
    """
    Convert a free-text user prompt into a structured parameter dict using
    simple keyword matching.  No LLM or external model is required.

    Returns a dict with keys: scene, style, mood, intensity, negative, noise, blur.
    """
    lower = prompt.lower()

    # Detect photographic style from keywords in the prompt
    style = DEFAULT_PARAMS["style"]
    for keyword, label in _STYLE_KEYWORDS.items():
        if keyword in lower:
            style = label
            break

    # Detect mood / lighting from keywords in the prompt
    mood = DEFAULT_PARAMS["mood"]
    for keyword, label in _MOOD_KEYWORDS.items():
        if keyword in lower:
            mood = label
            break

    return {
        "scene": prompt,
        "style": style,
        "mood": mood,
        "intensity": 0.75,
        "negative": DEFAULT_PARAMS["negative"],
        "noise": 0.08,
        "blur": 0.02,
    }
