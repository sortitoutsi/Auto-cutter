#!/usr/bin/env python3
"""
Removes the bright "glow" / halo fringe around cutout subjects.

The glow is the semi-transparent fringe at alpha edges that retains light from
the original background. This script darkens those fringe pixels toward the
colour of the nearest fully-opaque interior pixels, with stronger darkening for
dark subjects (dark hair / skin) and lighter touch for blond / light subjects.

Usage:
    python -m image_cropper.pipeline.deglow input.png [output.png]
    python -m image_cropper.pipeline.deglow input_dir/ output_dir/
    python -m image_cropper.pipeline.deglow input_dir/ output_dir/ --strength 0.8 --radius 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from image_cropper.errors import ImageCropperError, ValidationError

SUPPORTED: set[str] = {".png", ".webp"}

# Alpha thresholds
OPAQUE_MIN: int = 220
FRINGE_MAX: int = 219
FRINGE_MIN: int = 5

# Search radius (px) when looking for interior colour anchor
ANCHOR_RADIUS: int = 8

# Strength range: how aggressively we pull fringe toward anchor colour.
STRENGTH_MIN: float = 0.25
STRENGTH_MAX: float = 0.90


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Perceptual luminance, 0-1, shape (...,)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def build_anchor_map(
    rgb: np.ndarray,
    alpha: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (anchor_rgb, dist_to_opaque).

    For every pixel, the nearest fully-opaque pixel's RGB is gathered into
    ``anchor_rgb`` and the Euclidean distance into ``dist_to_opaque``.
    """
    opaque_mask = alpha >= OPAQUE_MIN
    dist, (nearest_row, nearest_col) = distance_transform_edt(~opaque_mask, return_indices=True)
    anchor_rgb = rgb[nearest_row, nearest_col].astype(np.float32)
    return anchor_rgb, dist


def deglow_image(
    img: Image.Image,
    strength_scale: float = 1.0,
    radius: int = ANCHOR_RADIUS,
) -> Image.Image:
    """Apply deglow to a single RGBA image.

    Non-RGBA inputs are converted to RGBA first.
    """
    rgba = np.array(img.convert("RGBA"), dtype=np.float32)
    assert rgba.ndim == 3 and rgba.shape[2] == 4, f"unexpected RGBA shape: {rgba.shape}"
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    assert (alpha >= 0).all() and (alpha <= 255).all(), "alpha out of [0,255]"

    fringe = (alpha >= FRINGE_MIN) & (alpha < OPAQUE_MIN)
    if not fringe.any():
        return img  # no semi-transparent fringe, nothing to do

    anchor_rgb, dist_to_opaque = build_anchor_map(rgb, alpha, radius)
    assert anchor_rgb.shape == rgb.shape, f"anchor shape drift: {anchor_rgb.shape} vs {rgb.shape}"

    in_range = fringe & (dist_to_opaque <= radius)
    if not in_range.any():
        return img

    anchor_lum = luminance(anchor_rgb / 255.0)
    dark_factor = (1.0 - anchor_lum) ** 1.5
    strength = STRENGTH_MIN + (STRENGTH_MAX - STRENGTH_MIN) * dark_factor
    strength = np.clip(strength * strength_scale, 0.0, 1.0)

    out_rgb = rgb.copy()
    s = strength[..., np.newaxis]
    blended = rgb * (1.0 - s) + anchor_rgb * s

    excess = np.clip(rgb - anchor_rgb, 0, None)
    blended -= excess * s * 0.5

    out_rgb[in_range] = np.clip(blended, 0, 255)[in_range]

    out = np.dstack([out_rgb, alpha[..., np.newaxis]]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def process_file(src: Path, dst: Path, strength: float, radius: int) -> bool:
    """Deglow one file. Raises :class:`ValidationError` if input has no alpha."""
    img = Image.open(src)
    if img.mode not in ("RGBA", "LA", "PA"):
        raise ValidationError(f"input has no alpha channel (mode {img.mode}): {src.name}")
    result = deglow_image(img, strength_scale=strength, radius=radius)
    dst.parent.mkdir(parents=True, exist_ok=True)
    result.save(dst, format="PNG", optimize=False)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Remove glow/halo from cutout PNGs.")
    p.add_argument("input", help="Input PNG/WebP file or directory")
    p.add_argument(
        "output",
        nargs="?",
        help="Output file or directory (default: beside input with '_dg' suffix)",
    )
    p.add_argument(
        "--strength", type=float, default=1.0, help="Global strength multiplier 0-2 (default: 1.0)"
    )
    p.add_argument(
        "--radius",
        type=int,
        default=ANCHOR_RADIUS,
        help=f"Max fringe radius in px (default: {ANCHOR_RADIUS})",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    try:
        if args.strength < 0:
            raise ValidationError(f"--strength must be >= 0, got {args.strength}")
        if args.radius < 0:
            raise ValidationError(f"--radius must be >= 0, got {args.radius}")

        src = Path(args.input)
        if not src.exists():
            raise ValidationError(f"input not found: {src}")

        if src.is_dir():
            files = sorted(f for f in src.iterdir() if f.suffix.lower() in SUPPORTED)
            if not files:
                print(f"No supported images in {src}/")
                sys.exit(0)
            out_dir = Path(args.output) if args.output else src.parent / (src.name + "_dg")
            out_dir.mkdir(parents=True, exist_ok=True)
            ok = 0
            for f in files:
                dst = out_dir / (f.stem + ".png")
                if dst.exists() and not args.overwrite:
                    print(f"  skip (exists): {dst.name}")
                    continue
                try:
                    process_file(f, dst, args.strength, args.radius)
                    print(f"  ok  {f.name} → {dst.name}")
                    ok += 1
                except ImageCropperError as e:
                    print(f"  [!] {e}", file=sys.stderr)
            print(f"\nDone: {ok}/{len(files)} images processed → {out_dir}/")
        else:
            dst = (
                Path(args.output)
                if args.output
                else src.with_stem(src.stem + "_dg").with_suffix(".png")
            )
            if dst.exists() and not args.overwrite:
                print(f"Output already exists: {dst}  (use --overwrite)")
                sys.exit(1)
            process_file(src, dst, args.strength, args.radius)
            print(f"Saved → {dst}")
    except ImageCropperError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
