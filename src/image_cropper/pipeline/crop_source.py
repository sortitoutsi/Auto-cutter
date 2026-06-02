#!/usr/bin/env python3
"""
Face cropper: detects faces and crops to include full hair and shoulders.
Reads from input/, writes to output/.
Minimum output size: 500x500px. No downscaling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from image_cropper.errors import DetectionError, ImageCropperError
from image_cropper.models import face_detector_path
from image_cropper.types import CropBox, FaceBBox

SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Padding multipliers relative to detected face height
HAIR_PADDING_FACTOR: float = 1.1
SHOULDER_PADDING_FACTOR: float = 0.8
SIDE_PADDING_FACTOR: float = 0.35

MIN_OUTPUT_SIZE: int = 500


def ensure_model() -> bool:
    return face_detector_path().exists()


def detect_face_mediapipe(image_rgb: np.ndarray) -> FaceBBox | None:
    """Detect face using MediaPipe Tasks API (0.10.x+)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    base_opts = mp_tasks.BaseOptions(model_asset_path=str(face_detector_path()))
    opts = mp_vision.FaceDetectorOptions(
        base_options=base_opts,
        min_detection_confidence=0.4,
    )
    with mp_vision.FaceDetector.create_from_options(opts) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = detector.detect(mp_image)

    if not result.detections:
        return None

    best = max(result.detections, key=lambda d: d.categories[0].score)
    bbox = best.bounding_box
    return FaceBBox(
        x=int(bbox.origin_x),
        y=int(bbox.origin_y),
        w=int(bbox.width),
        h=int(bbox.height),
    )


def detect_face_opencv(image_rgb: np.ndarray) -> FaceBBox | None:
    """Fallback: Haar cascade face detector (bundled with OpenCV)."""
    import cv2

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if not len(faces):
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return FaceBBox(x=int(x), y=int(y), w=int(w), h=int(h))


def detect_face(image_rgb: np.ndarray) -> FaceBBox | None:
    """Try MediaPipe first, fall back to OpenCV Haar cascade."""
    if face_detector_path().exists():
        try:
            result = detect_face_mediapipe(image_rgb)
            if result:
                return result
        except Exception as e:
            print(f"  [!] MediaPipe detection failed ({e}), falling back to OpenCV")
    return detect_face_opencv(image_rgb)


def compute_crop_box(
    face_x: int,
    face_y: int,
    face_w: int,
    face_h: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Expand face bbox with padding for hair and shoulders.

    Returns ``(left, top, right, bottom)``. Use ``CropBox.from_ltrb(*...)``
    if a value object is desired.
    """
    assert face_w > 0 and face_h > 0, f"degenerate face bbox: {face_w}x{face_h}"
    assert img_w > 0 and img_h > 0, f"degenerate image: {img_w}x{img_h}"

    hair_pad = int(face_h * HAIR_PADDING_FACTOR)
    shoulder_pad = int(face_h * SHOULDER_PADDING_FACTOR)
    side_pad = int(face_w * SIDE_PADDING_FACTOR)

    left = face_x - side_pad
    top = face_y - hair_pad
    right = face_x + face_w + side_pad
    bottom = face_y + face_h + shoulder_pad

    # Clamp to image bounds
    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)

    crop_w = right - left
    crop_h = bottom - top

    # Expand to meet minimum size, centred on the current crop
    if crop_w < MIN_OUTPUT_SIZE:
        expand = MIN_OUTPUT_SIZE - crop_w
        left = max(0, left - expand // 2)
        right = min(img_w, left + MIN_OUTPUT_SIZE)
        left = max(0, right - MIN_OUTPUT_SIZE)

    if crop_h < MIN_OUTPUT_SIZE:
        expand = MIN_OUTPUT_SIZE - crop_h
        top = max(0, top - expand // 2)
        bottom = min(img_h, top + MIN_OUTPUT_SIZE)
        top = max(0, bottom - MIN_OUTPUT_SIZE)

    assert right > left and bottom > top, f"degenerate crop: ({left},{top})-({right},{bottom})"
    assert left >= 0 and right <= img_w, f"crop x out of bounds: ({left},{right}) vs {img_w}"
    assert top >= 0 and bottom <= img_h, f"crop y out of bounds: ({top},{bottom}) vs {img_h}"

    return left, top, right, bottom


def process_image(input_path: Path, output_path: Path) -> bool:
    """Detect a face in ``input_path`` and save a head+shoulders crop.

    Raises :class:`DetectionError` if no face is found.
    """
    image = Image.open(input_path)
    img_rgb = np.array(image.convert("RGB"))
    img_h, img_w = img_rgb.shape[:2]

    face = detect_face(img_rgb)
    if face is None:
        raise DetectionError(f"no face detected in {input_path.name}")

    left, top, right, bottom = compute_crop_box(face.x, face.y, face.w, face.h, img_w, img_h)
    box = CropBox(left=left, top=top, right=right, bottom=bottom)

    print(
        f"  Face at ({face.x},{face.y}) {face.w}x{face.h}px  →  "
        f"crop ({box.left},{box.top}) {box.width}x{box.height}px"
    )

    cropped = image.crop(box.as_tuple())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        if cropped.mode == "RGBA":
            cropped = cropped.convert("RGB")
        cropped.save(output_path, quality=95, subsampling=0)
    elif suffix == ".webp":
        cropped.save(output_path, quality=95, method=6)
    else:
        cropped.save(output_path)

    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Detect faces and crop to head+shoulders.")
    parser.add_argument(
        "input_dir", nargs="?", default="input", help="Input directory (default: input)"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="output/cropped",
        help="Output directory (default: output/cropped)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: '{input_dir}' directory not found.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(exist_ok=True)
    ensure_model()

    images = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not images:
        print(f"No supported images found in {input_dir}/")
        sys.exit(0)

    print(f"Found {len(images)} image(s) in {input_dir}/\n")
    ok = 0
    for img_path in images:
        out_path = output_dir / img_path.name
        print(f"Processing: {img_path.name}")
        try:
            if process_image(img_path, out_path):
                ok += 1
                print(f"  Saved → {out_path}\n")
            else:
                print()
        except ImageCropperError as e:
            print(f"  [!] {e}\n")
        except Exception as e:
            print(f"  [!] unexpected error: {e}\n", file=sys.stderr)

    print(f"Done. {ok}/{len(images)} image(s) processed.")


if __name__ == "__main__":
    main()
