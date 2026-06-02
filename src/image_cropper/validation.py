"""Boundary validators that raise typed errors for bad inputs.

Each pipeline module's ``process_image`` (and similar entry points)
calls these validators first so the rest of the function can assume
its inputs are well-formed. All validators raise
:class:`~image_cropper.errors.ValidationError` for any failure so
``main()`` wrappers can catch a single type.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from image_cropper.errors import ValidationError
from image_cropper.types import CropBox


def validate_input_path(p: Path, supported_exts: set[str]) -> Path:
    """Ensure ``p`` exists and (if a file) has a supported extension.

    Directories are accepted unconditionally — the calling module is
    responsible for filtering their contents. This mirrors the existing
    CLI contract where most pipeline modules accept either a single
    file or a directory of images.
    """
    if not isinstance(p, Path):
        raise ValidationError(f"input path must be a pathlib.Path, got {type(p).__name__}")
    if not p.exists():
        raise ValidationError(f"input path does not exist: {p}")
    if p.is_file():
        ext = p.suffix.lower()
        if ext not in supported_exts:
            allowed = ", ".join(sorted(supported_exts))
            raise ValidationError(
                f"unsupported file extension '{ext}' for {p.name}; expected one of: {allowed}"
            )
    if not os.access(p, os.R_OK):
        raise ValidationError(f"input path is not readable: {p}")
    return p


def validate_output_path(p: Path) -> Path:
    """Ensure the parent of ``p`` exists and is writable, creating it if needed."""
    if not isinstance(p, Path):
        raise ValidationError(f"output path must be a pathlib.Path, got {type(p).__name__}")
    parent = p.parent if p.suffix else p
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ValidationError(f"cannot create output directory {parent}: {e}") from e
    if not os.access(parent, os.W_OK):
        raise ValidationError(f"output directory is not writable: {parent}")
    return p


def ensure_rgba(img: Image.Image) -> Image.Image:
    """Return ``img`` converted to RGBA, raising for non-image inputs."""
    if not isinstance(img, Image.Image):
        raise ValidationError(f"expected PIL.Image.Image, got {type(img).__name__}")
    if img.mode == "RGBA":
        return img
    try:
        return img.convert("RGBA")
    except (ValueError, OSError) as e:
        raise ValidationError(f"cannot convert image mode '{img.mode}' to RGBA: {e}") from e


def validate_image_array(
    arr: np.ndarray,
    *,
    channels: int = 3,
    dtype: type | np.dtype = np.uint8,
) -> None:
    """Assert that ``arr`` is an (H, W, channels) array of the requested dtype."""
    if not isinstance(arr, np.ndarray):
        raise ValidationError(f"expected numpy.ndarray, got {type(arr).__name__}")
    if arr.ndim != 3:
        raise ValidationError(
            f"expected 3-D array (H, W, C), got {arr.ndim}-D with shape {arr.shape}"
        )
    if arr.shape[2] != channels:
        raise ValidationError(
            f"expected {channels} channels, got {arr.shape[2]} (shape {arr.shape})"
        )
    expected_dtype = np.dtype(dtype)
    if arr.dtype != expected_dtype:
        raise ValidationError(f"expected dtype {expected_dtype}, got {arr.dtype}")


def validate_crop_box(box: CropBox, img_w: int, img_h: int) -> None:
    """Assert ``box`` is non-degenerate and intersects the (img_w × img_h) image."""
    if not isinstance(box, CropBox):
        raise ValidationError(f"expected CropBox, got {type(box).__name__}")
    if box.width <= 0 or box.height <= 0:
        raise ValidationError(
            f"crop box has non-positive dimensions: width={box.width}, height={box.height}"
        )
    if box.right <= 0 or box.bottom <= 0:
        raise ValidationError(f"crop box {box} lies entirely above/left of the image")
    if box.left >= img_w or box.top >= img_h:
        raise ValidationError(f"crop box {box} lies entirely outside image ({img_w}x{img_h})")


__all__ = [
    "validate_input_path",
    "validate_output_path",
    "ensure_rgba",
    "validate_image_array",
    "validate_crop_box",
]
