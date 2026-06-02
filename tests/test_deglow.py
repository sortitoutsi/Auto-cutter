"""Smoke tests for `pipeline.deglow.deglow_image`.

Guarded behind `pytest.importorskip("scipy")` so the suite stays green
on environments without the scientific stack.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

scipy = pytest.importorskip("scipy")

from image_cropper.pipeline.deglow import deglow_image  # noqa: E402


def _opaque_red_square(size: int = 32, fringe: bool = True) -> Image.Image:
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[:, :, 0] = 200
    arr[8:24, 8:24, 3] = 255  # opaque interior
    if fringe:
        # bright fringe ring (alpha ~120) that deglow should darken
        arr[6:8, 8:24, 0] = 255  # bright fringe
        arr[6:8, 8:24, 3] = 120
        arr[24:26, 8:24, 0] = 255
        arr[24:26, 8:24, 3] = 120
    return Image.fromarray(arr, "RGBA")


def test_no_fringe_returns_image_unchanged() -> None:
    img = _opaque_red_square(fringe=False)
    out = deglow_image(img)
    # When there's no semi-transparent fringe the function should return the
    # input image (possibly the same object) untouched.
    out_arr = np.array(out)
    in_arr = np.array(img)
    np.testing.assert_array_equal(out_arr, in_arr)


def test_fringe_pixels_become_less_bright() -> None:
    img = _opaque_red_square(fringe=True)
    out = deglow_image(img)
    in_arr = np.array(img).astype(np.int16)
    out_arr = np.array(out).astype(np.int16)
    # Fringe rows are y=6,7 and y=24,25; opaque interior is y=8..23
    fringe_in = in_arr[6:8, 8:24, 0]
    fringe_out = out_arr[6:8, 8:24, 0]
    # Bright fringe should be pulled down toward the (dark red) anchor.
    assert fringe_out.mean() < fringe_in.mean()


def test_opaque_interior_untouched() -> None:
    img = _opaque_red_square(fringe=True)
    out = deglow_image(img)
    in_arr = np.array(img)
    out_arr = np.array(out)
    # Interior alpha pixels (>=220) should be left alone.
    interior_in = in_arr[10:22, 10:22]
    interior_out = out_arr[10:22, 10:22]
    np.testing.assert_array_equal(interior_in, interior_out)


def test_alpha_channel_preserved() -> None:
    img = _opaque_red_square(fringe=True)
    out = deglow_image(img)
    in_alpha = np.array(img)[:, :, 3]
    out_alpha = np.array(out)[:, :, 3]
    np.testing.assert_array_equal(in_alpha, out_alpha)


def test_non_rgba_image_converted() -> None:
    """deglow_image converts non-RGBA via `convert('RGBA')` — should not raise."""
    img = Image.new("RGB", (16, 16), (10, 20, 30))
    out = deglow_image(img)
    assert out.size == (16, 16)
