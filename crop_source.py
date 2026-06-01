#!/usr/bin/env python3
"""
Face cropper: detects faces and crops to include full hair and shoulders.
Reads from input/, writes to output/.
Minimum output size: 500x500px. No downscaling.
"""

import os
import sys
import urllib.request
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Padding multipliers relative to detected face height
HAIR_PADDING_FACTOR = 1.1  # space above face top (covers afros, long hair)
SHOULDER_PADDING_FACTOR = 0.8  # space below chin (to mid-chest)
SIDE_PADDING_FACTOR = 0.35  # horizontal padding on each side

MIN_OUTPUT_SIZE = 500

MODEL_PATH = Path(__file__).parent / "face_detector.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def ensure_model():
    if MODEL_PATH.exists():
        return True
    print(f"Downloading face detection model → {MODEL_PATH.name} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  Model downloaded.\n")
        return True
    except Exception as e:
        print(f"  [!] Could not download model: {e}")
        return False


def detect_face_mediapipe(image_rgb: np.ndarray):
    """Detect face using MediaPipe Tasks API (0.10.x+). Returns (x,y,w,h) or None."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks import python as mp_tasks

    base_opts = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
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
    return bbox.origin_x, bbox.origin_y, bbox.width, bbox.height


def detect_face_opencv(image_rgb: np.ndarray):
    """Fallback: Haar cascade face detector (bundled with OpenCV)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if not len(faces):
        return None
    # Pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def detect_face(image_rgb: np.ndarray):
    """Try MediaPipe first, fall back to OpenCV Haar cascade."""
    if MODEL_PATH.exists():
        try:
            result = detect_face_mediapipe(image_rgb)
            if result:
                return result
        except Exception as e:
            print(f"  [!] MediaPipe detection failed ({e}), falling back to OpenCV")
    return detect_face_opencv(image_rgb)


def compute_crop_box(face_x, face_y, face_w, face_h, img_w, img_h):
    """Expand face bbox with padding for hair and shoulders."""
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

    return left, top, right, bottom


def process_image(input_path: Path, output_path: Path):
    image = Image.open(input_path)
    img_rgb = np.array(image.convert("RGB"))
    img_h, img_w = img_rgb.shape[:2]

    face = detect_face(img_rgb)
    if face is None:
        print(f"  [!] No face detected — skipping {input_path.name}")
        return False

    face_x, face_y, face_w, face_h = face
    left, top, right, bottom = compute_crop_box(
        face_x, face_y, face_w, face_h, img_w, img_h
    )

    crop_w = right - left
    crop_h = bottom - top
    print(
        f"  Face at ({face_x},{face_y}) {face_w}x{face_h}px  →  crop ({left},{top}) {crop_w}x{crop_h}px"
    )

    cropped = image.crop((left, top, right, bottom))

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detect faces and crop to head+shoulders.")
    parser.add_argument("input_dir", nargs="?", default="input", help="Input directory (default: input)")
    parser.add_argument("output_dir", nargs="?", default="output/cropped", help="Output directory (default: output/cropped)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: '{input_dir}' directory not found.")
        sys.exit(1)

    output_dir.mkdir(exist_ok=True)
    ensure_model()

    images = [
        p
        for p in sorted(input_dir.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
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
        except Exception as e:
            print(f"  [!] Error: {e}\n")

    print(f"Done. {ok}/{len(images)} image(s) processed.")


if __name__ == "__main__":
    main()
