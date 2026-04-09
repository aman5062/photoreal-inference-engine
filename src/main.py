#!/usr/bin/env python3
# main.py — CLI entry point for the photorealistic inference engine
import sys
import json
import time
from pathlib import Path

# Ensure src/ is on the path when run from the project root
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import load_models
from model import get_params
from prompt import build_prompts
from generate import generate
from filters import apply_filters
from PIL import Image


def run(prompt: str, output_dir: str = "outputs") -> Path:
    """
    Full pipeline: text prompt → 512×512 PNG.

    Args:
        prompt:     Free-text description of the desired image.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved PNG file.
    """
    t_total = time.time()

    # ----------------------------------------------------------------- [1/5] Load
    print("\n[1/5] Loading model …")
    t = time.time()
    pipe = load_models()
    print(f"      Done in {time.time() - t:.1f}s")

    # ----------------------------------------------------------------- [2/5] Parse
    print("\n[2/5] Building params from prompt …")
    t = time.time()
    params = get_params(prompt)
    print(f"      Done in {time.time() - t:.1f}s")
    print(json.dumps(params, indent=2))

    # ----------------------------------------------------------------- [3/5] Build prompts
    print("\n[3/5] Building SD prompts …")
    positive, negative = build_prompts(params)
    print(f"  (+) {positive[:120]} …")
    print(f"  (-) {negative[:120]} …")

    # ----------------------------------------------------------------- [4/5] Generate
    print("\n[4/5] Stable Diffusion 1.5 …")
    t = time.time()
    arr = generate(pipe, positive, negative, params)
    elapsed = time.time() - t
    print(f"      Done in {elapsed:.1f}s")
    print(f"      Output shape : {arr.shape}  dtype : {arr.dtype}")
    print(f"      Raw bytes    : {arr.nbytes:,}")

    # ----------------------------------------------------------------- [5/5] Filters
    print("\n[5/5] Applying byte-level filters …")
    t = time.time()
    arr = apply_filters(arr, params)
    print(f"      Done in {time.time() - t:.2f}s")

    # ----------------------------------------------------------------- Save
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filename: first 40 chars of prompt, spaces → underscores, .png
    safe_name = prompt[:40].replace(" ", "_").replace("/", "-")
    out_path = out_dir / f"{safe_name}.png"

    Image.fromarray(arr).save(str(out_path), format="PNG", optimize=False)
    print(f"\n✓ Saved : {out_path}  ({out_path.stat().st_size:,} bytes)")
    print(f"  Total pipeline time: {time.time() - t_total:.1f}s")

    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py '<your prompt>'")
        print("Example: python main.py 'a stormy ocean at night with lightning'")
        sys.exit(1)

    user_prompt = " ".join(sys.argv[1:])
    print(f"Prompt: {user_prompt}")
    run(user_prompt)
