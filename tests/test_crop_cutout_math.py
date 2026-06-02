"""Tests for pure-math helpers in `pipeline.crop_cutout`."""

from __future__ import annotations

import numpy as np
from PIL import Image

from image_cropper.pipeline.crop_cutout import (
    MIN_SHIRT_PIXELS,
    OUTPUT_SIZE,
    compute_crop,
    find_hair_top,
)

# ---------------------------------------------------------------------------
# find_hair_top
# ---------------------------------------------------------------------------


def test_find_hair_top_returns_none_for_rgb() -> None:
    img = Image.new("RGB", (32, 32))
    assert find_hair_top(img) is None


def test_find_hair_top_returns_none_for_fully_transparent() -> None:
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    assert find_hair_top(img) is None


def test_find_hair_top_single_opaque_row() -> None:
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[15, :, 3] = 255
    img = Image.fromarray(arr, "RGBA")
    assert find_hair_top(img) == 15


def test_find_hair_top_ignores_subthreshold_alpha() -> None:
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[5, :, 3] = 5  # at the threshold — should be skipped (ALPHA_THRESHOLD = 10)
    arr[20, :, 3] = 255
    img = Image.fromarray(arr, "RGBA")
    assert find_hair_top(img) == 20


def test_find_hair_top_returns_topmost() -> None:
    arr = np.zeros((50, 50, 4), dtype=np.uint8)
    arr[10:20, 10:20, 3] = 200
    img = Image.fromarray(arr, "RGBA")
    assert find_hair_top(img) == 10


# ---------------------------------------------------------------------------
# compute_crop
# ---------------------------------------------------------------------------


def _geo(**overrides: float) -> dict:
    base = {
        "chin_y": 500.0,
        "forehead_y": 200.0,
        "left_x": 250.0,
        "right_x": 450.0,
        "face_height": 300.0,
        "detector": "test",
    }
    base.update(overrides)
    return base


def test_compute_crop_returns_none_when_no_shirt_below_chin() -> None:
    # img_h only slightly below chin → fewer than MIN_SHIRT_PIXELS available
    img_h = int(500 + MIN_SHIRT_PIXELS // 2)
    assert compute_crop(_geo(), img_w=800, img_h=img_h) is None


def test_compute_crop_returns_box_when_shirt_present() -> None:
    box = compute_crop(_geo(), img_w=800, img_h=1000)
    assert box is not None
    left, top, right, bottom = box
    assert right > left
    assert bottom > top


def test_compute_crop_output_is_square() -> None:
    box = compute_crop(_geo(), img_w=800, img_h=1000)
    assert box is not None
    left, top, right, bottom = box
    assert (right - left) == (bottom - top), f"non-square crop: {box}"


def test_compute_crop_hair_top_override_drives_top_edge() -> None:
    geo = _geo(forehead_y=200.0)
    # supply a much higher hair_top
    box = compute_crop(geo, img_w=800, img_h=1000, hair_top_y=50)
    assert box is not None
    _, top, _, _ = box
    # The supplied hair_top_y should pin the top edge well above the forehead
    # landmark; allow re-centering for square expansion to push it further up.
    assert top <= 200


def test_compute_crop_extends_when_face_is_wider_than_tall() -> None:
    # Very wide face → side-padding driven square expansion
    geo = _geo(left_x=100.0, right_x=700.0, forehead_y=400.0, chin_y=480.0, face_height=80.0)
    box = compute_crop(geo, img_w=800, img_h=1000, hair_top_y=350)
    assert box is not None
    left, top, right, bottom = box
    assert (right - left) == (bottom - top)


def test_compute_crop_respects_chin_pixels_override() -> None:
    """Larger chin_pixels should make the crop taller (more shirt visible)."""
    box_default = compute_crop(_geo(), img_w=800, img_h=1000)
    box_big = compute_crop(_geo(), img_w=800, img_h=1000, chin_pixels=OUTPUT_SIZE // 2)
    assert box_default is not None and box_big is not None
    assert (box_big[3] - box_big[1]) >= (box_default[3] - box_default[1])
