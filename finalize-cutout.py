#!/usr/bin/env python3
"""
Centers a cutout PNG onto a 250x250 transparent canvas.

Usage:
    python finalize-cutout.py input.png [output.png]
    python finalize-cutout.py input_dir/ output_dir/
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from PIL import Image


CANVAS_SIZE = (250, 250)


def center_on_canvas(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGBA")

    # Crop to the bounding box of non-transparent pixels
    bbox = img.getbbox()
    if bbox is None:
        print(f"  skip (fully transparent): {src.name}")
        return
    cropped = img.crop(bbox)

    # Scale down if the cutout is larger than the canvas
    cw, ch = CANVAS_SIZE
    iw, ih = cropped.size
    if iw > cw or ih > ch:
        cropped.thumbnail(CANVAS_SIZE, Image.LANCZOS)
        iw, ih = cropped.size

    canvas = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    x = (cw - iw) // 2
    y = (ch - ih) // 2
    canvas.paste(cropped, (x, y), cropped)

    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, format="PNG")
    print(f"  {src.name} → {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Center cutout(s) on a 250×250 canvas")
    parser.add_argument("input", help="Input PNG file or directory")
    parser.add_argument("output", nargs="?", help="Output PNG file or directory (default: alongside input)")
    args = parser.parse_args()

    src = Path(args.input)

    if src.is_dir():
        files = sorted(src.glob("*.png"))
        if not files:
            print(f"No PNG files found in {src}", file=sys.stderr)
            sys.exit(1)
        out_dir = Path(args.output) if args.output else src.parent / (src.name + "_finalized")
        for f in files:
            center_on_canvas(f, out_dir / f.name)
    else:
        if not src.exists():
            print(f"File not found: {src}", file=sys.stderr)
            sys.exit(1)
        if args.output:
            dst = Path(args.output)
        else:
            dst = src.with_name(src.stem + "_final.png")
        center_on_canvas(src, dst)


if __name__ == "__main__":
    main()
