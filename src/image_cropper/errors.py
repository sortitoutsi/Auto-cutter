"""Exception hierarchy for the image-cropper pipeline.

Every error raised by the package inherits from :class:`ImageCropperError`,
so callers (including pipeline ``main()`` wrappers and the GUI subprocess
runner) can catch a single base type and still distinguish failure modes
when needed.
"""
from __future__ import annotations


class ImageCropperError(Exception):
    """Base class for all errors raised inside the image-cropper package."""


class ValidationError(ImageCropperError):
    """Raised when an input fails a precondition check.

    Examples: file does not exist, unsupported image format, array has
    the wrong dtype or channel count, crop box collapses to zero area.
    """


class DetectionError(ImageCropperError):
    """Raised when a face / eye / landmark detector returns no result.

    Distinct from :class:`ModelError`: the model loaded fine, it just
    could not find a face in this image.
    """


class ModelError(ImageCropperError):
    """Raised when a model file is missing, fails to load, or cannot be downloaded."""


class BackgroundRemovalError(ModelError):
    """Raised when the BiRefNet matting / salient model pipeline fails."""


class PipelineError(ImageCropperError):
    """Raised when a pipeline stage cannot continue because a prior stage's output is missing or malformed."""


__all__ = [
    "ImageCropperError",
    "ValidationError",
    "DetectionError",
    "ModelError",
    "BackgroundRemovalError",
    "PipelineError",
]
