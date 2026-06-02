"""Tests for `pipeline.finalize_cutout.center_on_canvas` behaviour.

The function writes a 250x250 RGBA PNG centred on a non-transparent
bounding box. We don't have a pure-math extracted helper, so we test
via the file API on tiny synthetic inputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from image_cropper.pipeline.finalize_cutout import CANVAS_SIZE, center_on_canvas


def _make_cutout(path: Path, size: tuple[int, int], opaque_box: tuple[int, int, int, int]) -> None:
    """Write a small RGBA PNG with a single opaque rectangle inside it."""
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    x0, y0, x1, y1 = opaque_box
    arr[y0:y1, x0:x1, :3] = [200, 50, 50]
    arr[y0:y1, x0:x1, 3] = 255
    Image.fromarray(arr, "RGBA").save(path, format="PNG")


def test_canvas_size_constant() -> None:
    assert CANVAS_SIZE == (250, 250)


def test_centre_small_cutout(tmp_path: Path) -> None:
    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    _make_cutout(src, size=(80, 80), opaque_box=(10, 10, 70, 70))  # 60x60 opaque region
    center_on_canvas(src, dst)

    out = Image.open(dst)
    assert out.mode == "RGBA"
    assert out.size == CANVAS_SIZE
    alpha = np.array(out)[:, :, 3]
    # The 60x60 opaque region should be centred on the 250x250 canvas:
    # margin = (250 - 60) / 2 = 95 on each side.
    assert alpha[94, 124] == 0  # just outside left edge
    assert alpha[95, 125] == 255  # inside top-left corner of subject
    assert alpha[154, 124] == 255  # inside bottom-left corner
    assert alpha[155, 124] == 0  # just outside bottom edge


def test_oversized_cutout_is_thumbnailed(tmp_path: Path) -> None:
    src = tmp_path / "big.png"
    dst = tmp_path / "out.png"
    _make_cutout(src, size=(500, 600), opaque_box=(0, 0, 500, 600))
    center_on_canvas(src, dst)
    out = Image.open(dst)
    assert out.size == CANVAS_SIZE
    # subject filled the entire input → after thumbnail, alpha should cover most of canvas
    alpha = np.array(out)[:, :, 3]
    assert (alpha > 0).sum() > 0.5 * 250 * 250


def test_fully_transparent_input_skips_write(tmp_path: Path) -> None:
    src = tmp_path / "blank.png"
    dst = tmp_path / "should_not_exist.png"
    arr = np.zeros((20, 20, 4), dtype=np.uint8)
    Image.fromarray(arr, "RGBA").save(src, format="PNG")
    center_on_canvas(src, dst)
    assert not dst.exists()
