"""Tests for boundary validators."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_cropper.errors import ValidationError
from image_cropper.types import CropBox
from image_cropper.validation import (
    ensure_rgba,
    validate_crop_box,
    validate_image_array,
    validate_input_path,
    validate_output_path,
)

SUPPORTED = {".jpg", ".png"}


# ---------------------------------------------------------------------------
# validate_input_path
# ---------------------------------------------------------------------------

def test_validate_input_path_accepts_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "img.png"
    f.write_bytes(b"x")
    assert validate_input_path(f, SUPPORTED) == f


def test_validate_input_path_accepts_directory(tmp_path: Path) -> None:
    assert validate_input_path(tmp_path, SUPPORTED) == tmp_path


def test_validate_input_path_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="does not exist"):
        validate_input_path(tmp_path / "missing.png", SUPPORTED)


def test_validate_input_path_rejects_wrong_extension(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("hi")
    with pytest.raises(ValidationError, match="unsupported file extension"):
        validate_input_path(f, SUPPORTED)


def test_validate_input_path_rejects_non_path() -> None:
    with pytest.raises(ValidationError, match="pathlib.Path"):
        validate_input_path("not-a-path", SUPPORTED)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_output_path
# ---------------------------------------------------------------------------

def test_validate_output_path_creates_parent(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "result.png"
    assert validate_output_path(out) == out
    assert out.parent.exists()


def test_validate_output_path_accepts_dir(tmp_path: Path) -> None:
    out = tmp_path / "outdir"
    validate_output_path(out)
    assert out.exists()


def test_validate_output_path_rejects_non_path() -> None:
    with pytest.raises(ValidationError):
        validate_output_path("not-a-path")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ensure_rgba
# ---------------------------------------------------------------------------

def test_ensure_rgba_passes_through_rgba() -> None:
    img = Image.new("RGBA", (4, 4))
    assert ensure_rgba(img) is img


def test_ensure_rgba_converts_rgb() -> None:
    img = Image.new("RGB", (4, 4))
    out = ensure_rgba(img)
    assert out.mode == "RGBA"


def test_ensure_rgba_rejects_non_image() -> None:
    with pytest.raises(ValidationError, match="PIL.Image"):
        ensure_rgba("not-an-image")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_image_array
# ---------------------------------------------------------------------------

def test_validate_image_array_happy_path() -> None:
    arr = np.zeros((10, 20, 3), dtype=np.uint8)
    validate_image_array(arr)  # no raise


def test_validate_image_array_wrong_dimensions() -> None:
    with pytest.raises(ValidationError, match="3-D"):
        validate_image_array(np.zeros((10, 10), dtype=np.uint8))


def test_validate_image_array_wrong_channels() -> None:
    with pytest.raises(ValidationError, match="channels"):
        validate_image_array(np.zeros((10, 10, 4), dtype=np.uint8), channels=3)


def test_validate_image_array_wrong_dtype() -> None:
    with pytest.raises(ValidationError, match="dtype"):
        validate_image_array(np.zeros((10, 10, 3), dtype=np.float32), dtype=np.uint8)


def test_validate_image_array_rejects_non_array() -> None:
    with pytest.raises(ValidationError, match="numpy.ndarray"):
        validate_image_array([[1, 2, 3]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_crop_box
# ---------------------------------------------------------------------------

def test_validate_crop_box_happy_path() -> None:
    validate_crop_box(CropBox(10, 10, 90, 90), 100, 100)


def test_validate_crop_box_rejects_zero_width() -> None:
    with pytest.raises(ValidationError, match="non-positive"):
        validate_crop_box(CropBox(50, 10, 50, 90), 100, 100)


def test_validate_crop_box_rejects_inverted() -> None:
    with pytest.raises(ValidationError, match="non-positive"):
        validate_crop_box(CropBox(90, 90, 10, 10), 100, 100)


def test_validate_crop_box_rejects_box_off_screen() -> None:
    # box entirely past right/bottom edge
    with pytest.raises(ValidationError, match="entirely outside"):
        validate_crop_box(CropBox(200, 200, 300, 300), 100, 100)


def test_validate_crop_box_rejects_box_off_top_left() -> None:
    with pytest.raises(ValidationError, match="above/left"):
        validate_crop_box(CropBox(-50, -50, -10, -10), 100, 100)


def test_validate_crop_box_rejects_non_cropbox() -> None:
    with pytest.raises(ValidationError, match="CropBox"):
        validate_crop_box((0, 0, 10, 10), 100, 100)  # type: ignore[arg-type]
