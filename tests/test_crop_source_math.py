"""Tests for `pipeline.crop_source.compute_crop_box`."""

from __future__ import annotations

import pytest

from image_cropper.pipeline.crop_source import MIN_OUTPUT_SIZE, compute_crop_box


def test_crop_centred_face_within_image() -> None:
    box = compute_crop_box(face_x=400, face_y=300, face_w=200, face_h=300, img_w=1200, img_h=1600)
    left, top, right, bottom = box
    assert left >= 0 and top >= 0
    assert right <= 1200 and bottom <= 1600
    assert (right - left) >= MIN_OUTPUT_SIZE
    assert (bottom - top) >= MIN_OUTPUT_SIZE


def test_crop_box_clamped_to_image_at_top_left_corner() -> None:
    box = compute_crop_box(face_x=10, face_y=10, face_w=200, face_h=300, img_w=1200, img_h=1600)
    left, top, _, _ = box
    assert left >= 0
    assert top >= 0


def test_crop_box_clamped_to_image_at_bottom_right_corner() -> None:
    box = compute_crop_box(face_x=900, face_y=1200, face_w=200, face_h=300, img_w=1200, img_h=1600)
    _, _, right, bottom = box
    assert right <= 1200
    assert bottom <= 1600


def test_crop_box_expands_to_min_size_when_face_is_small() -> None:
    box = compute_crop_box(face_x=300, face_y=300, face_w=80, face_h=120, img_w=2000, img_h=2000)
    left, top, right, bottom = box
    assert (right - left) >= MIN_OUTPUT_SIZE
    assert (bottom - top) >= MIN_OUTPUT_SIZE


def test_crop_box_does_not_exceed_image_for_small_image() -> None:
    """If the image itself is smaller than MIN_OUTPUT_SIZE, the box clamps to image bounds."""
    box = compute_crop_box(face_x=50, face_y=50, face_w=80, face_h=100, img_w=300, img_h=300)
    left, top, right, bottom = box
    assert left >= 0 and top >= 0
    assert right <= 300 and bottom <= 300


@pytest.mark.parametrize("face_x, face_y", [(0, 0), (1000, 0), (0, 1500), (1000, 1500)])
def test_crop_box_stays_within_image_for_corner_faces(face_x: int, face_y: int) -> None:
    box = compute_crop_box(
        face_x=face_x, face_y=face_y, face_w=200, face_h=300, img_w=1200, img_h=1600
    )
    left, top, right, bottom = box
    assert 0 <= left < right <= 1200
    assert 0 <= top < bottom <= 1600
