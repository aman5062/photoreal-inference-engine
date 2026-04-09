# filters.py — Byte-level post-processing on raw uint8 image arrays
import numpy as np
from scipy.ndimage import convolve, gaussian_filter


def apply_filters(arr: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply a chain of byte-level NumPy / SciPy filters to enhance realism:
      1. Color grade  — subtle cool-blue cinematic tone
      2. S-curve contrast stretch per channel
      3. Unsharp masking  — high-frequency sharpening
      4. Radial vignette — darkened corners
      5. Gaussian blur  — controlled softness (params['blur'])
      6. Film grain     — cinematic noise texture (params['noise'])

    Args:
        arr:    uint8 array (1024, 1024, 3) straight from SDXL
        params: dict with keys 'intensity', 'noise', 'blur'

    Returns:
        Post-processed uint8 array (1024, 1024, 3)
    """
    arr = arr.astype(np.float32)

    # 1. Subtle color grade: reduce red, boost blue for cinematic coolness
    intensity = float(params.get("intensity", 0.75))
    arr[:, :, 0] = np.clip(arr[:, :, 0] * (1.0 - intensity * 0.12), 0, 255)  # red
    arr[:, :, 2] = np.clip(arr[:, :, 2] * (1.0 + intensity * 0.08), 0, 255)  # blue

    # 2. S-curve contrast stretch (per channel, float domain [0,1])
    arr_n = arr / 255.0
    strength = 1.0 + intensity * 0.5
    arr_n = (arr_n - 0.5) * strength + 0.5
    arr = np.clip(arr_n, 0.0, 1.0) * 255.0

    # 3. Unsharp masking: sharpen edges via Laplacian kernel convolution
    kernel = np.array(
        [[0, -1, 0],
         [-1, 5, -1],
         [0, -1, 0]],
        dtype=np.float32,
    )
    sharpened = np.stack(
        [convolve(arr[:, :, c], kernel) for c in range(3)], axis=2
    )
    # Blend: 65% original + 35% sharpened to avoid over-sharpening artifacts
    arr = arr * 0.65 + sharpened * 0.35
    arr = np.clip(arr, 0.0, 255.0)

    # 4. Radial vignette: smoothly darken corners
    h, w = arr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    # Vignette darkens at dist > ~0.7 from centre, bottoms out at 0.25
    vignette = np.clip(1.0 - dist * 0.42, 0.25, 1.0)
    arr *= vignette[:, :, np.newaxis]

    # 5. Gaussian blur (controlled by params['blur'])
    blur = float(params.get("blur", 0.02))
    if blur > 0.01:
        sigma = blur * 2.5
        arr = np.stack(
            [gaussian_filter(arr[:, :, c], sigma=sigma) for c in range(3)], axis=2
        )

    # 6. Film grain noise (controlled by params['noise'])
    noise = float(params.get("noise", 0.08))
    if noise > 0.01:
        grain_strength = noise * 22.0   # max ~22 stddev at noise=1.0
        grain = np.random.normal(0, grain_strength, arr.shape).astype(np.float32)
        arr = arr + grain

    return np.clip(arr, 0, 255).astype(np.uint8)
