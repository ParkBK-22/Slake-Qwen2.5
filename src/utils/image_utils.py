from __future__ import annotations

import random
from typing import Any

import numpy as np
from PIL import Image


def to_pil_image(image: Any) -> Image.Image:
    """
    Convert various image formats to PIL.Image.

    Supported:
    - PIL.Image
    - numpy array
    - dict with common image fields
    """
    if image is None:
        raise ValueError("image is None")

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")

    if isinstance(image, dict):
        for key in ["image", "img", "pil_image", "pixel_values"]:
            if key in image and image[key] is not None:
                return to_pil_image(image[key])

        if "path" in image and image["path"] is not None:
            return Image.open(image["path"]).convert("RGB")

        if "bytes" in image and image["bytes"] is not None:
            import io

            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")

    try:
        return Image.fromarray(np.asarray(image)).convert("RGB")
    except Exception as exc:
        raise TypeError(f"Cannot convert image of type {type(image)} to PIL.Image") from exc


def resize_image(image: Any, size: int = 512) -> Image.Image:
    """
    Resize image to size x size and convert to RGB.
    """
    image = to_pil_image(image)
    return image.resize((size, size), resample=Image.BICUBIC)


def make_black_image(image: Any) -> Image.Image:
    """
    Create a black image with the same size as the input image.
    """
    image = to_pil_image(image)
    return Image.new("RGB", image.size, color=(0, 0, 0))


def make_patchshuffle_image(
    image: Any,
    patch_size: int = 16,
    seed: int = 42,
) -> Image.Image:
    """
    Shuffle non-overlapping patch_size x patch_size patches.

    For image_size=512 and patch_size=16, this creates a 32 x 32 grid of patches.
    """
    image = to_pil_image(image)

    width, height = image.size

    if width % patch_size != 0 or height % patch_size != 0:
        raise ValueError(
            f"Image size {image.size} must be divisible by patch_size={patch_size}."
        )

    patches = []
    positions = []

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            box = (x, y, x + patch_size, y + patch_size)
            patches.append(image.crop(box))
            positions.append((x, y))

    rng = random.Random(seed)
    shuffled_patches = patches[:]
    rng.shuffle(shuffled_patches)

    out = Image.new("RGB", image.size)

    for patch, pos in zip(shuffled_patches, positions):
        out.paste(patch, pos)

    return out