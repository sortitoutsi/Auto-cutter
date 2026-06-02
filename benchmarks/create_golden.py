#!/usr/bin/env python3
"""
Generate deterministic synthetic portrait images for benchmarking.

These images simulate final pipeline output (250×250 RGBA cutouts) without
requiring any ML models. They are committed alongside baseline.json so CI can
validate benchmark.py consistently without running the heavy pipeline.

Usage:
  python benchmarks/create_golden.py
  # → writes benchmarks/golden/*.png and benchmarks/baseline.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from image_cropper.pipeline.benchmark import compute_metrics

GOLDEN_DIR = Path(__file__).parent / "golden"
BASELINE_PATH = Path(__file__).parent / "baseline.json"


def _make_portrait(
    size: int = 250,
    *,
    face_radius: int = 70,
    face_center: tuple[int, int] = (125, 110),
    hair_ry: int = 85,
    shoulder_bottom: int = 235,
    fringe_width: int = 6,
    subject_color: tuple[int, int, int] = (90, 65, 50),
) -> Image.Image:
    """
    Synthetic portrait cutout: oval face + hair dome + shoulder rectangle.
    A gradient fringe ring simulates the semi-transparent edge a real cutout
    has after background removal.
    """
    arr = np.zeros((size, size, 4), dtype=np.float32)

    ys, xs = np.mgrid[0:size, 0:size]
    cx, cy = face_center

    # Face oval (slightly taller than wide)
    face_rx = face_radius
    face_ry = int(face_radius * 1.15)
    face_dist = np.sqrt(((xs - cx) / face_rx) ** 2 + ((ys - cy) / face_ry) ** 2)

    # Hair dome (wider, taller ellipse above the face centre)
    hair_rx = int(face_radius * 1.1)
    hair_dist = np.sqrt(((xs - cx) / hair_rx) ** 2 + ((ys - (cy - 10)) / hair_ry) ** 2)

    # Shoulder area: trapezoid below chin
    chin_y = cy + face_ry
    shoulder_width_top = int(face_radius * 1.8)
    shoulder_width_bot = int(face_radius * 2.8)
    shoulder_top = chin_y - 10

    def shoulder_mask() -> np.ndarray:
        in_y = (ys >= shoulder_top) & (ys <= shoulder_bottom)
        t = np.clip((ys - shoulder_top) / max(shoulder_bottom - shoulder_top, 1), 0, 1)
        half_w = shoulder_width_top + (shoulder_width_bot - shoulder_width_top) * t
        in_x = np.abs(xs - cx) <= half_w
        return in_y & in_x

    body_mask = (face_dist <= 1.0) | (hair_dist <= 1.0) | shoulder_mask()

    # Fringe: pixels just outside the hard edge get linearly decreasing alpha
    # Use distance-from-body-edge to create a smooth fringe
    from scipy.ndimage import distance_transform_edt
    fringe_alpha = np.zeros((size, size), dtype=np.float32)
    outside = ~body_mask
    dist_outside = distance_transform_edt(outside)
    fringe_zone = outside & (dist_outside <= fringe_width)
    fringe_alpha[fringe_zone] = (1.0 - dist_outside[fringe_zone] / fringe_width) * 180.0

    # Compose final alpha
    alpha = np.zeros((size, size), dtype=np.float32)
    alpha[body_mask] = 255.0
    alpha[fringe_zone] = fringe_alpha[fringe_zone]

    # RGB: uniform subject colour with slight per-pixel noise for realism
    rng = np.random.default_rng(seed=42)
    noise = rng.integers(-8, 8, size=(size, size, 3), dtype=np.int16)
    r = int(np.clip(subject_color[0] + noise[:, :, 0], 0, 255).mean())  # keep uniform for reproducibility
    subject_rgb = np.array(subject_color, dtype=np.float32)

    arr[body_mask | fringe_zone, :3] = subject_rgb
    arr[:, :, 3] = alpha

    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")


GOLDEN_SPECS = [
    # (filename, kwargs for _make_portrait)
    # 1. Centered dark-haired subject — baseline portrait
    ("portrait_centered_dark.png", {}),
    # 2. Slightly off-centre, lighter subject — tests centering + fringe brightness
    (
        "portrait_offset_light.png",
        {
            "face_center": (135, 105),
            "subject_color": (200, 175, 155),
            "fringe_width": 8,
        },
    ),
]


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for filename, kwargs in GOLDEN_SPECS:
        path = GOLDEN_DIR / filename
        img = _make_portrait(**kwargs)
        img.save(path, format="PNG", compress_level=9)
        metrics = compute_metrics(path)
        results[filename] = metrics
        print(f"  {filename}")
        for k, v in metrics.items():
            print(f"    {k:<30} {v}")

    with open(BASELINE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline written → {BASELINE_PATH}")
    print(f"Golden images written → {GOLDEN_DIR}/")


if __name__ == "__main__":
    main()
