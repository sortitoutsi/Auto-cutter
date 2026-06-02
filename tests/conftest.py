"""Shared pytest fixtures.

Fixtures here are kept import-cheap: nothing imports torch / mediapipe /
dlib, so the whole suite runs under the lightweight `requirements-ci.txt`
environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "benchmarks" / "golden"


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    if not GOLDEN_DIR.exists():
        pytest.skip(f"golden directory missing: {GOLDEN_DIR}")
    return GOLDEN_DIR


@pytest.fixture(scope="session")
def golden_dark_portrait(golden_dir: Path) -> Path:
    p = golden_dir / "portrait_centered_dark.png"
    if not p.exists():
        pytest.skip(f"golden image missing: {p}")
    return p


@pytest.fixture(scope="session")
def golden_light_portrait(golden_dir: Path) -> Path:
    p = golden_dir / "portrait_offset_light.png"
    if not p.exists():
        pytest.skip(f"golden image missing: {p}")
    return p


@pytest.fixture
def synthetic_rgba_32() -> Image.Image:
    """32x32 RGBA: opaque red square in the middle, transparent border, soft fringe."""
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[:, :, 0] = 200  # red channel
    # opaque interior
    arr[8:24, 8:24, 3] = 255
    # fringe ring (rows 6-7, 24-25, cols 6-7, 24-25) at ~50% alpha
    arr[6:8, 6:26, 3] = 128
    arr[24:26, 6:26, 3] = 128
    arr[6:26, 6:8, 3] = 128
    arr[6:26, 24:26, 3] = 128
    return Image.fromarray(arr, "RGBA")


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    return out
