"""Tests for the pure-math helpers in `pipeline.align`.

Only `compute_rotation_angle` and `_centroid` are exercised — they have
no ML dependencies and can be imported without triggering mediapipe /
dlib model loads (the heavy imports live inside the detector functions).
"""
from __future__ import annotations

import math

import pytest

from image_cropper.pipeline.align import _centroid, compute_rotation_angle


# ---------------------------------------------------------------------------
# compute_rotation_angle
# ---------------------------------------------------------------------------

def test_horizontal_eyes_zero_angle() -> None:
    angle = compute_rotation_angle(left_eye=(100.0, 200.0), right_eye=(300.0, 200.0))
    assert angle == pytest.approx(0.0, abs=1e-9)


def test_right_eye_lower_positive_angle() -> None:
    """Right eye 100px below left over a 100px horizontal span → +45°."""
    angle = compute_rotation_angle(left_eye=(0.0, 0.0), right_eye=(100.0, 100.0))
    assert angle == pytest.approx(45.0, abs=1e-9)


def test_right_eye_higher_negative_angle() -> None:
    angle = compute_rotation_angle(left_eye=(0.0, 100.0), right_eye=(100.0, 0.0))
    assert angle == pytest.approx(-45.0, abs=1e-9)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ((0.0, 0.0), (1.0, 0.0), 0.0),
        ((0.0, 0.0), (0.0, 1.0), 90.0),
        ((0.0, 0.0), (-1.0, 0.0), 180.0),
        ((0.0, 0.0), (0.0, -1.0), -90.0),
    ],
)
def test_compute_rotation_angle_quadrants(
    left: tuple[float, float], right: tuple[float, float], expected: float
) -> None:
    assert compute_rotation_angle(left, right) == pytest.approx(expected, abs=1e-9)


def test_compute_rotation_angle_bounded() -> None:
    """The angle returned by atan2 stays in [-180, 180]."""
    for lx in (-1.0, 0.0, 1.0):
        for ly in (-1.0, 0.0, 1.0):
            for rx in (-1.0, 0.0, 1.0):
                for ry in (-1.0, 0.0, 1.0):
                    if (lx, ly) == (rx, ry):
                        continue
                    a = compute_rotation_angle((lx, ly), (rx, ry))
                    assert -180.0 <= a <= 180.0


# ---------------------------------------------------------------------------
# _centroid
# ---------------------------------------------------------------------------

def test_centroid_single_point() -> None:
    assert _centroid([(10.0, 20.0)]) == (10.0, 20.0)


def test_centroid_two_points_midpoint() -> None:
    assert _centroid([(0.0, 0.0), (10.0, 20.0)]) == (5.0, 10.0)


def test_centroid_many_points() -> None:
    pts = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
    cx, cy = _centroid(pts)
    assert cx == pytest.approx(5.0)
    assert cy == pytest.approx(5.0)


def test_centroid_returns_floats() -> None:
    cx, cy = _centroid([(1, 2), (3, 4)])  # ints in
    assert isinstance(cx, float) and isinstance(cy, float)
