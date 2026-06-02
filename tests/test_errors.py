"""Sanity checks on the exception hierarchy."""
from __future__ import annotations

import pytest

from image_cropper.errors import (
    BackgroundRemovalError,
    DetectionError,
    ImageCropperError,
    ModelError,
    PipelineError,
    ValidationError,
)


@pytest.mark.parametrize(
    "subclass",
    [ValidationError, DetectionError, ModelError, BackgroundRemovalError, PipelineError],
)
def test_all_errors_inherit_from_base(subclass: type[Exception]) -> None:
    assert issubclass(subclass, ImageCropperError)


def test_background_removal_error_is_model_error() -> None:
    assert issubclass(BackgroundRemovalError, ModelError)


def test_errors_carry_message() -> None:
    err = ValidationError("bad input: x=5")
    assert "bad input" in str(err)


def test_catch_all_with_base_class() -> None:
    """Pipeline main() wrappers rely on catching ImageCropperError to handle every subclass."""
    for exc in (
        ValidationError("x"),
        DetectionError("x"),
        ModelError("x"),
        BackgroundRemovalError("x"),
        PipelineError("x"),
    ):
        with pytest.raises(ImageCropperError):
            raise exc
