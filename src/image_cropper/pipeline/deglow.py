#!/usr/bin/env python3
"""
Removes the bright "glow" / halo fringe around cutout subjects.

The glow is the semi-transparent fringe at alpha edges that retains light from the
original background.  This script darkens those fringe pixels toward the colour of
the nearest fully-opaque interior pixels, with stronger darkening for dark subjects
(dark hair / skin) and lighter touch for blond / light subjects.

Usage:
    python deglow.py input.png [output.png]
    python deglow.py input_dir/ output_dir/
    python deglow.py input_dir/ output_dir/ --strength 0.8 --radius 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

SUPPORTED = {".png", ".webp"}

# Alpha thresholds
OPAQUE_MIN = 220  # pixel is "interior" if alpha >= this
FRINGE_MAX = 219  # pixel is "fringe"   if alpha <  this
FRINGE_MIN = 5  # ignore near-invisible dust below this

# Search radius (px) when looking for interior colour anchor
ANCHOR_RADIUS = 8

# Strength range: how aggressively we pull fringe toward anchor colour.
# For a very dark anchor (brightness ~0) we use STRENGTH_MAX.
# For a very light anchor (brightness ~255) we use STRENGTH_MIN.
STRENGTH_MIN = 0.25  # light blond / pale skin
STRENGTH_MAX = 0.90  # black hair / very dark skin


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Perceptual luminance, 0-1, shape (...,)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def build_anchor_map(
    rgb: np.ndarray,
    alpha: np.ndarray,
    radius: int,
) -> np.ndarray:
    """
    For every pixel, find the nearest fully-opaque pixel and return its RGB.
    Uses distance transform to find the nearest opaque pixel, then a small
    window around it to average a stable colour.
    """
    H, W = alpha.shape
    opaque_mask = alpha >= OPAQUE_MIN  # (H, W) bool

    # distance transform: distance of every pixel to nearest opaque pixel
    dist, (nearest_row, nearest_col) = distance_transform_edt(~opaque_mask, return_indices=True)

    # Gather the nearest opaque pixel's colour for every fringe pixel
    anchor_rgb = rgb[nearest_row, nearest_col].astype(np.float32)  # (H, W, 3)

    return anchor_rgb, dist


def deglow_image(
    img: Image.Image,
    strength_scale: float = 1.0,
    radius: int = ANCHOR_RADIUS,
) -> Image.Image:
    """Apply deglow to a single RGBA image."""
    rgba = np.array(img.convert("RGBA"), dtype=np.float32)  # (H,W,4) 0-255
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]

    # --- fringe mask ---
    fringe = (alpha >= FRINGE_MIN) & (alpha < OPAQUE_MIN)
    if not fringe.any():
        return img  # no semi-transparent fringe, nothing to do

    # --- anchor colours (nearest opaque pixel RGB) ---
    anchor_rgb, dist_to_opaque = build_anchor_map(rgb, alpha, radius)

    # Only work on fringe pixels within `radius` of an opaque pixel.
    # Beyond that we're too far from the subject to reliably correct.
    in_range = fringe & (dist_to_opaque <= radius)
    if not in_range.any():
        return img

    # --- per-pixel strength based on anchor luminance ---
    anchor_lum = luminance(anchor_rgb / 255.0)  # 0-1, shape (H,W)
    # Dark anchor → high strength; bright anchor → low strength
    # Quadratic ramp so very dark values jump up quickly
    dark_factor = (1.0 - anchor_lum) ** 1.5  # 0-1
    strength = STRENGTH_MIN + (STRENGTH_MAX - STRENGTH_MIN) * dark_factor
    strength = np.clip(strength * strength_scale, 0.0, 1.0)

    # --- blend fringe pixels toward anchor ---
    out_rgb = rgb.copy()
    s = strength[..., np.newaxis]  # (H,W,1)
    blended = rgb * (1.0 - s) + anchor_rgb * s  # move fringe colour toward anchor

    # Also darken slightly even beyond the blend, proportional to how bright
    # the fringe is compared to its anchor (i.e. glow = excess brightness)
    excess = np.clip(rgb - anchor_rgb, 0, None)  # only the bright excess
    blended -= excess * s * 0.5  # suppress half of excess beyond the blend

    out_rgb[in_range] = np.clip(blended, 0, 255)[in_range]

    out = np.dstack([out_rgb, alpha[..., np.newaxis]]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def process_file(src: Path, dst: Path, strength: float, radius: int) -> bool:
    try:
        img = Image.open(src)
        if img.mode not in ("RGBA", "LA", "PA"):
            print(f"  skip (no alpha): {src.name}")
            return False
        result = deglow_image(img, strength_scale=strength, radius=radius)
        dst.parent.mkdir(parents=True, exist_ok=True)
        result.save(dst, format="PNG", optimize=False)
        return True
    except Exception as exc:
        print(f"  error {src.name}: {exc}", file=sys.stderr)
        return False


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

    src = Path(args.input)

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
            if process_file(f, dst, args.strength, args.radius):
                print(f"  ok  {f.name} → {dst.name}")
                ok += 1
        print(f"\nDone: {ok}/{len(files)} images processed → {out_dir}/")
    else:
        if not src.exists():
            print(f"Error: '{src}' not found", file=sys.stderr)
            sys.exit(1)
        if args.output:
            dst = Path(args.output)
        else:
            dst = src.with_stem(src.stem + "_dg").with_suffix(".png")
        if dst.exists() and not args.overwrite:
            print(f"Output already exists: {dst}  (use --overwrite)")
            sys.exit(1)
        if process_file(src, dst, args.strength, args.radius):
            print(f"Saved → {dst}")


if __name__ == "__main__":
    main()
