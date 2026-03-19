"""Validation helpers for uploaded or downloaded source images."""

from __future__ import annotations

from io import BytesIO

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from app.config import ALLOWED_IMAGE_FORMATS, MAX_UPLOAD_SIZE_BYTES, MIN_IMAGE_WIDTH


def load_and_validate_image(image_bytes: bytes) -> Image.Image:
    """Validate raw image bytes and return a normalized RGB image."""
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Image size must not exceed 10MB.")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    image_format = (image.format or "").lower()
    if image_format not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Use jpg, jpeg, png, or webp.",
        )

    if image.width < MIN_IMAGE_WIDTH:
        raise HTTPException(
            status_code=400,
            detail=f"Image width must be at least {MIN_IMAGE_WIDTH}px.",
        )

    return image.convert("RGB")
