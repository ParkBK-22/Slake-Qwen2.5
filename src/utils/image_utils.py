from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def to_pil_image(image: Any) -> Image.Image:
    """Convert common Hugging Face image objects to RGB PIL.Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    # Some datasets store images as dicts with bytes/path.
    if isinstance(image, dict):
        if "path" in image and image["path"]:
            return Image.open(image["path"]).convert("RGB")
        if "bytes" in image and image["bytes"]:
            import io

            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")

    # Some datasets store a path string.
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")

    raise TypeError(f"Cannot convert object of type {type(image)} to PIL.Image")


def resize_image(image: Image.Image, size: int = 512) -> Image.Image:
    """Resize an image to size x size for the controlled experiment."""
    image = to_pil_image(image)
    return image.resize((size, size), Image.Resampling.BICUBIC)


def make_black_image(image: Image.Image) -> Image.Image:
    """Create a black RGB image with the same size as input image."""
    image = to_pil_image(image)
    return Image.new("RGB", image.size, color=(0, 0, 0))
