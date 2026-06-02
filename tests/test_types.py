"""Unit tests for typed value objects."""
from __future__ import annotations

import dataclasses

import pytest

from image_cropper.types import CropBox, EyeDetection, FaceBBox, FaceGeometry


def test_facebbox_derived_properties() -> None:
    box = FaceBBox(x=10, y=20, w=50, h=80)
    assert box.right == 60
    assert box.bottom == 100


def test_facebbox_is_frozen() -> None:
    box = FaceBBox(x=0, y=0, w=10, h=10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        box.x = 5  # type: ignore[misc]


def test_cropbox_width_height() -> None:
    box = CropBox(left=10, top=20, right=110, bottom=220)
    assert box.width == 100
    assert box.height == 200
    assert not box.is_square


def test_cropbox_is_square_property() -> None:
    assert CropBox(0, 0, 100, 100).is_square
    assert not CropBox(0, 0, 100, 99).is_square


def test_cropbox_as_tuple_matches_pil_signature() -> None:
    box = CropBox(left=1, top=2, right=3, bottom=4)
    assert box.as_tuple() == (1, 2, 3, 4)


@pytest.mark.parametrize(
    "left, top, right, bottom, expected",
    [
        (1.4, 2.4, 3.6, 4.6, (1, 2, 4, 5)),
        (-0.6, 0.5, 9.5, 9.5, (-1, 0, 10, 10)),
        (0.5, 0.5, 1.5, 1.5, (0, 0, 2, 2)),
    ],
)
def test_cropbox_from_ltrb_rounds(
    left: float, top: float, right: float, bottom: float, expected: tuple[int, int, int, int]
) -> None:
    assert CropBox.from_ltrb(left, top, right, bottom).as_tuple() == expected


def test_cropbox_clamp_to_image_inside() -> None:
    box = CropBox(10, 10, 90, 90)
    assert box.clamp_to_image(100, 100) == box


def test_cropbox_clamp_to_image_negative() -> None:
    box = CropBox(-20, -30, 50, 60)
    assert box.clamp_to_image(100, 100) == CropBox(0, 0, 50, 60)


def test_cropbox_clamp_to_image_overflow() -> None:
    box = CropBox(50, 50, 200, 200)
    assert box.clamp_to_image(100, 100) == CropBox(50, 50, 100, 100)


def test_eye_detection_holds_points_and_name() -> None:
    det = EyeDetection(left_eye=(10.0, 20.0), right_eye=(50.0, 22.0), detector_name="dlib")
    assert det.left_eye == (10.0, 20.0)
    assert det.right_eye == (50.0, 22.0)
    assert det.detector_name == "dlib"


def test_face_geometry_fields() -> None:
    geo = FaceGeometry(
        chin_y=500.0,
        forehead_y=200.0,
        left_x=100.0,
        right_x=400.0,
        face_height=300.0,
        detector="MediaPipe",
    )
    assert geo.face_height == 300.0
    assert geo.detector == "MediaPipe"
