"""Typed value objects shared across pipeline modules.

Geometry results (eye positions, face bounding boxes, crop boxes,
landmark-derived face geometry) flow between detection, computation
and rendering code. Returning bare tuples or dicts hides field-name
typos from the type checker; the dataclasses below make every field
explicit and immutable.

Serialization shapes that need to round-trip through JSON (benchmark
metrics, scraped queue entries) use :class:`TypedDict` instead of
dataclasses so ``json.dump`` keeps working on plain dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

Point2D = tuple[float, float]


@dataclass(frozen=True, slots=True)
class EyeDetection:
    """Result of an eye-pair detector.

    ``left_eye`` is always the left side of the *image* (smaller x).
    """

    left_eye: Point2D
    right_eye: Point2D
    detector_name: str


@dataclass(frozen=True, slots=True)
class FaceBBox:
    """Axis-aligned face bounding box in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(frozen=True, slots=True)
class FaceGeometry:
    """Landmark-derived face geometry used to drive the portrait crop."""

    chin_y: float
    forehead_y: float
    left_x: float
    right_x: float
    face_height: float
    detector: str


@dataclass(frozen=True, slots=True)
class CropBox:
    """Crop box in left/top/right/bottom pixel coordinates.

    May extend past the source image; :meth:`clamp_to_image` returns a
    new box clamped to the image bounds.
    """

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def is_square(self) -> bool:
        return self.width == self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return the ``(left, top, right, bottom)`` tuple expected by PIL."""
        return (self.left, self.top, self.right, self.bottom)

    @classmethod
    def from_ltrb(cls, left: float, top: float, right: float, bottom: float) -> CropBox:
        """Construct a CropBox from float coordinates, rounding to int."""
        return cls(int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))

    def clamp_to_image(self, img_w: int, img_h: int) -> CropBox:
        """Return a copy clamped so all edges lie inside the image rect."""
        return CropBox(
            left=max(0, min(self.left, img_w)),
            top=max(0, min(self.top, img_h)),
            right=max(0, min(self.right, img_w)),
            bottom=max(0, min(self.bottom, img_h)),
        )


class MetricsDict(TypedDict):
    """Quality metrics emitted by :func:`image_cropper.pipeline.benchmark.compute_metrics`.

    Stored verbatim in ``benchmarks/baseline.json`` and ``results-ci.json``.
    """

    fg_coverage_pct: float
    fringe_density_pct: float
    mean_fringe_brightness: float
    alpha_edge_sharpness: float
    h_center_of_mass: float
    v_center_of_mass: float


class QueueEntry(TypedDict):
    """One row from the sortitoutsi.net submission queue scrape."""

    url: str
    alt: str


class SubmissionMeta(TypedDict):
    """Metadata scraped from a sortitoutsi.net collection page, saved as a sidecar."""

    submission_id: int
    person_id: int | None
    alt: str
    status: str       # "pending" | "in_progress" | "completed" | "rejected"
    image_type: str   # "source" | "game_ready"
    collection_url: str
    downloaded_at: str  # ISO-8601


class SubmitResult(TypedDict):
    """Result of posting a cutout back to sortitoutsi.net."""

    ok: bool
    submission_url: str | None  # URL of the created submission if successful
    message: str


__all__ = [
    "Point2D",
    "EyeDetection",
    "FaceBBox",
    "FaceGeometry",
    "CropBox",
    "MetricsDict",
    "QueueEntry",
    "SubmissionMeta",
    "SubmitResult",
]
