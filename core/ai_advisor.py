"""AI-assisted edit suggestions via the OpenAI API.

This module only ever *analyzes* a downscaled preview of the photo and
returns suggested edit parameters -- it never uploads the full-resolution
image and never regenerates pixels. The actual edit is applied locally
(see apply_exposure_contrast in image_ops.py) against the original
full-resolution array, so output quality matches the rest of the
resize/white-balance pipeline exactly: no resolution cap, no AI
re-rendering artifacts.
"""

import base64
import io
import json
import os

import numpy as np
from PIL import Image

DEFAULT_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")
ANALYSIS_MAX_DIM = 768  # thumbnail sent to the API; edits still apply at full res

_SYSTEM_PROMPT = (
    "You are a photo-editing assistant. Look at the photo and suggest a small, "
    "tasteful exposure and contrast correction. "
    'Respond with ONLY a JSON object of the form: '
    '{"exposure": <float, -1.0 to 1.0>, "contrast": <float, -1.0 to 1.0>, '
    '"reasoning": "<one short sentence>"}. '
    "0 means no change. Keep suggestions modest -- these are nudges, not overhauls."
)


class AdvisorError(RuntimeError):
    """Raised for any failure to get a usable suggestion (auth, network, parsing)."""


def _downscaled_jpeg_b64(image: np.ndarray, max_dim: int = ANALYSIS_MAX_DIM) -> str:
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    thumb = Image.fromarray(image)
    if scale < 1.0:
        thumb = thumb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def suggest_adjustments(image: np.ndarray, api_key: str | None = None, model: str = DEFAULT_MODEL) -> dict:
    """Ask the model for exposure/contrast suggestions on a downscaled preview.

    Returns {"exposure": float, "contrast": float, "reasoning": str}.
    Raises AdvisorError on any API, network, or parsing failure.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise AdvisorError("No OpenAI API key provided.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AdvisorError("The 'openai' package is not installed.") from exc

    b64 = _downscaled_jpeg_b64(image)
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Suggest an exposure/contrast correction for this photo."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
    except Exception as exc:  # covers auth/network/rate-limit errors from the SDK
        raise AdvisorError(f"OpenAI request failed: {exc}") from exc

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
        exposure = float(data["exposure"])
        contrast = float(data["contrast"])
        reasoning = str(data.get("reasoning", ""))
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise AdvisorError(f"Could not parse model response: {raw!r}") from exc

    return {
        "exposure": max(-1.0, min(1.0, exposure)),
        "contrast": max(-1.0, min(1.0, contrast)),
        "reasoning": reasoning,
    }
