# model.py — TinyLlama 1.1B inference: parse user prompt into structured JSON
import json
import re

SYSTEM = '''Output ONLY valid JSON, no markdown, no explanation:
{
  "scene": "detailed scene description",
  "style": "photographic style keyword",
  "mood":  "lighting and atmosphere",
  "intensity": 0.0-1.0,
  "negative": "comma separated unwanted elements",
  "noise": 0.0-1.0,
  "blur": 0.0-1.0
}'''

# Sensible defaults returned when LLM fails completely
DEFAULT_PARAMS = {
    "scene": "photorealistic scene",
    "style": "dramatic cinematic photography",
    "mood": "cinematic, moody lighting",
    "intensity": 0.75,
    "negative": "blurry, low quality, cartoon, anime, watermark",
    "noise": 0.08,
    "blur": 0.02,
}


def _clamp_float(value, lo=0.0, hi=1.0) -> float:
    """Clamp a value to [lo, hi] and return as float."""
    try:
        return float(max(lo, min(hi, float(value))))
    except (TypeError, ValueError):
        return (lo + hi) / 2.0


def _validate(params: dict) -> dict:
    """Ensure all required keys are present and values are sane."""
    required_str = ["scene", "style", "mood", "negative"]
    required_float = ["intensity", "noise", "blur"]

    for key in required_str:
        if not isinstance(params.get(key), str) or not params[key].strip():
            params[key] = DEFAULT_PARAMS[key]

    for key in required_float:
        params[key] = _clamp_float(params.get(key, DEFAULT_PARAMS[key]))

    return params


def get_params(llm, prompt: str) -> dict:
    """
    Run TinyLlama to convert a free-text user prompt into a structured
    parameter dict.  Retries up to 3 times with increasing temperature.
    Falls back to sensible defaults on total failure.
    """
    for attempt in range(3):
        temperature = 0.4 + attempt * 0.1
        result = llm(
            f"[INST]{SYSTEM}\nDescribe: {prompt}[/INST]",
            max_tokens=300,
            temperature=temperature,
        )
        text = result["choices"][0]["text"]
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                params = json.loads(match.group())
                required_keys = [
                    "scene", "style", "mood", "intensity",
                    "negative", "noise", "blur",
                ]
                if all(k in params for k in required_keys):
                    return _validate(params)
        except (json.JSONDecodeError, KeyError):
            continue

    # If LLM never produced valid JSON, build a basic param dict from the
    # raw prompt so we can still generate something reasonable.
    print("[WARN] TinyLlama did not return valid JSON — using defaults.")
    fallback = dict(DEFAULT_PARAMS)
    fallback["scene"] = prompt
    return fallback
