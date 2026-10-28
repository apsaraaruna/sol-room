"""Core pixel operations: auto white balance and high-quality resizing.

All functions take/return RGB uint8 numpy arrays (H, W, 3), independent of
any UI or file I/O concerns.
"""

import cv2
import numpy as np


def auto_white_balance(image: np.ndarray, clip_percentile: float = 0.5) -> np.ndarray:
    """Gray-world white balance.

    Assumes the average scene color should be neutral gray, then scales each
    channel so its mean matches the overall gray mean. Near-black/near-white
    outlier pixels are excluded from the mean so blown highlights or deep
    shadows don't skew the color estimate.
    """
    img = image.astype(np.float32)
    luma = img.mean(axis=2)
    lo, hi = np.percentile(luma, (clip_percentile, 100 - clip_percentile))
    mask = (luma >= lo) & (luma <= hi)
    if not mask.any():
        mask = np.ones_like(luma, dtype=bool)

    channel_means = np.array([img[..., c][mask].mean() for c in range(img.shape[2])])
    gray = channel_means.mean()
    scale = gray / np.clip(channel_means, 1e-6, None)

    balanced = img * scale.reshape(1, 1, -1)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize with the interpolation best suited to the direction of the resize.

    Lanczos for upscaling (sharpest results when enlarging), area-averaging
    for downscaling (avoids moire/aliasing that Lanczos can introduce when
    shrinking).
    """
    width, height = max(1, int(width)), max(1, int(height))
    src_h, src_w = image.shape[:2]
    is_upscale = (width * height) > (src_w * src_h)
    interpolation = cv2.INTER_LANCZOS4 if is_upscale else cv2.INTER_AREA
    return cv2.resize(image, (width, height), interpolation=interpolation)


def apply_exposure_contrast(image: np.ndarray, exposure: float, contrast: float) -> np.ndarray:
    """Apply a simple exposure (brightness) and contrast adjustment.

    exposure: roughly -1.0 (much darker) to 1.0 (much brighter), 0 = no change.
        Applied as a 2**exposure multiplicative gain (photographic stops).
    contrast: roughly -1.0 (flatten) to 1.0 (punchier), 0 = no change.
        Applied as a scale pivoted around mid-gray.

    Pure pixel math on the array as given -- runs at whatever resolution
    the caller passes in, so applying this to the full-resolution original
    loses no quality.
    """
    img = image.astype(np.float32)
    gain = 2.0 ** np.clip(exposure, -2.0, 2.0)
    img = img * gain
    contrast_factor = 1.0 + np.clip(contrast, -1.0, 1.0)
    img = (img - 127.5) * contrast_factor + 127.5
    return np.clip(img, 0, 255).astype(np.uint8)


def resize_to_long_edge(image: np.ndarray, long_edge: int) -> np.ndarray:
    """Resize keeping aspect ratio so the longer side equals long_edge."""
    h, w = image.shape[:2]
    if w >= h:
        new_w = long_edge
        new_h = max(1, round(h * (long_edge / w)))
    else:
        new_h = long_edge
        new_w = max(1, round(w * (long_edge / h)))
    return resize_image(image, new_w, new_h)
