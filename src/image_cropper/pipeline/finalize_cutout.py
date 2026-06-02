#!/usr/bin/env python3
"""
Centers a cutout PNG onto a 250x250 transparent canvas.

Usage:
    python -m image_cropper.pipeline.finalize_cutout input.png [output.png]
    python -m image_cropper.pipeline.finalize_cutout input_dir/ output_dir/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from image_cropper.errors import ImageCropperError, ValidationError
from image_cropper.validation import validate_input_path, validate_output_path

CANVAS_SIZE: tuple[int, int] = (250, 250)
SUPPORTED_EXTENSIONS: set[str] = {".png", ".webp"}


def center_on_canvas(src: Path, dst: Path) -> bool:
    """Center the non-transparent region of ``src`` onto a CANVAS_SIZE RGBA canvas.

    Returns True if a file was written, False if the input was fully
    transparent (no output produced).
    """
    validate_input_path(src, SUPPORTED_EXTENSIONS)
    validate_output_path(dst)

    img = Image.open(src).convert("RGBA")

    bbox = img.getbbox()
    if bbox is None:
        print(f"  skip (fully transparent): {src.name}")
        return False
    cropped = img.crop(bbox)

    cw, ch = CANVAS_SIZE
    iw, ih = cropped.size
    if iw > cw or ih > ch:
        cropped.thumbnail(CANVAS_SIZE, Image.LANCZOS)
        iw, ih = cropped.size

    assert iw <= cw and ih <= ch, f"thumbnail produced oversize image: {(iw, ih)} > {CANVAS_SIZE}"

    canvas = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    x = (cw - iw) // 2
    y = (ch - ih) // 2
    canvas.paste(cropped, (x, y), cropped)

    assert canvas.size == CANVAS_SIZE, f"canvas size drift: {canvas.size}"

    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, format="PNG")
    print(f"  {src.name} → {dst}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Center cutout(s) on a 250×250 canvas")
    parser.add_argument("input", help="Input PNG file or directory")
    parser.add_argument(
        "output", nargs="?", help="Output PNG file or directory (default: alongside input)"
    )
    args = parser.parse_args()

    try:
        src = Path(args.input)
        if not src.exists():
            raise ValidationError(f"input path does not exist: {src}")

        if src.is_dir():
            files = sorted(f for f in src.glob("*.png") if f.suffix.lower() in SUPPORTED_EXTENSIONS)
            if not files:
                raise ValidationError(f"no PNG files found in {src}")
            out_dir = Path(args.output) if args.output else src.parent / (src.name + "_finalized")
            for f in files:
                center_on_canvas(f, out_dir / f.name)
        else:
            dst = Path(args.output) if args.output else src.with_name(src.stem + "_final.png")
            center_on_canvas(src, dst)
    except ImageCropperError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
