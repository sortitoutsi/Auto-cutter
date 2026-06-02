#!/usr/bin/env python3
"""
Eye aligner: rotates images so both eyes are on the same horizontal line.

Usage:
  python -m image_cropper.pipeline.align                        # input/ → output/aligned/
  python -m image_cropper.pipeline.align src/ out/              # custom dirs
  python -m image_cropper.pipeline.align src/ out/ --debug      # also writes debug overlays
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from image_cropper.errors import DetectionError, ImageCropperError
from image_cropper.models import dlib_model_path, face_landmarker_path
from image_cropper.types import EyeDetection, Point2D

SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# MediaPipe iris landmark indices. Landmarks 468-472: left iris,
# 473-477: right iris (center is 468 and 473).
_LEFT_IRIS_IDX: list[int] = [468, 469, 470, 471, 472]
_RIGHT_IRIS_IDX: list[int] = [473, 474, 475, 476, 477]

# dlib eye landmark indices in the 68-point model
_DLIB_LEFT_EYE: list[int] = list(range(36, 42))
_DLIB_RIGHT_EYE: list[int] = list(range(42, 48))


def ensure_landmarker() -> bool:
    return face_landmarker_path().exists()


def ensure_dlib_model() -> bool:
    return dlib_model_path().exists()


def _centroid(points: Sequence[tuple[float, float]]) -> Point2D:
    arr = np.array(points)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _order_left_right(a: Point2D, b: Point2D) -> tuple[Point2D, Point2D]:
    """Return (left, right) so left.x <= right.x."""
    return (a, b) if a[0] <= b[0] else (b, a)


def detect_eyes_landmarker(image_rgb: np.ndarray) -> EyeDetection | None:
    """MediaPipe Face Landmarker with iris refinement."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    h, w = image_rgb.shape[:2]
    base_opts = mp_tasks.BaseOptions(model_asset_path=str(face_landmarker_path()))
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    with mp_vision.FaceLandmarker.create_from_options(opts) as detector:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = detector.detect(mp_image)

    if not result.face_landmarks:
        return None

    lms = result.face_landmarks[0]
    total = len(lms)

    if total >= 478:
        left_pts = [(lms[i].x * w, lms[i].y * h) for i in _LEFT_IRIS_IDX if i < total]
        right_pts = [(lms[i].x * w, lms[i].y * h) for i in _RIGHT_IRIS_IDX if i < total]
        label = "MediaPipe Iris"
    else:
        # Eye-contour fallback (no iris landmarks)
        left_pts = [
            (lms[i].x * w, lms[i].y * h) for i in [33, 133, 160, 158, 144, 153] if i < total
        ]
        right_pts = [
            (lms[i].x * w, lms[i].y * h) for i in [362, 263, 387, 385, 373, 380] if i < total
        ]
        label = "MediaPipe Eye Contour"

    if not left_pts or not right_pts:
        return None

    left, right = _order_left_right(_centroid(left_pts), _centroid(right_pts))
    return EyeDetection(left_eye=left, right_eye=right, detector_name=label)


def detect_eyes_dlib(image_rgb: np.ndarray) -> EyeDetection | None:
    """dlib 68-point landmark detector. Averages eye-ring landmarks."""
    import cv2
    import dlib

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(dlib_model_path()))

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dets = detector(gray, 1)
    if not dets:
        return None

    det = max(dets, key=lambda d: d.area())
    shape = predictor(gray, det)
    pts = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

    left, right = _order_left_right(
        _centroid([pts[i] for i in _DLIB_LEFT_EYE]),
        _centroid([pts[i] for i in _DLIB_RIGHT_EYE]),
    )
    return EyeDetection(left_eye=left, right_eye=right, detector_name="dlib 68-point")


def detect_eyes_opencv(image_rgb: np.ndarray) -> EyeDetection | None:
    """Last-resort: Haar cascade eye detector."""
    import cv2

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if not len(faces):
        return None

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    face_gray = gray[fy : fy + fh, fx : fx + fw]

    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 10, minSize=(20, 20))
    if len(eyes) < 2:
        return None

    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    centers = [(fx + ex + ew // 2, fy + ey + eh // 2) for ex, ey, ew, eh in eyes]
    centers.sort(key=lambda c: c[0])
    left, right = (
        (float(centers[0][0]), float(centers[0][1])),
        (
            float(centers[1][0]),
            float(centers[1][1]),
        ),
    )
    return EyeDetection(left_eye=left, right_eye=right, detector_name="OpenCV Haar")


def detect_eyes(image_rgb: np.ndarray) -> EyeDetection | None:
    """Try MediaPipe Landmarker → dlib → OpenCV Haar."""
    if face_landmarker_path().exists():
        try:
            result = detect_eyes_landmarker(image_rgb)
            if result:
                return result
            print("  [!] MediaPipe Landmarker found no face, trying dlib...")
        except Exception as e:
            print(f"  [!] MediaPipe Landmarker failed ({e}), trying dlib...")

    if dlib_model_path().exists():
        try:
            result = detect_eyes_dlib(image_rgb)
            if result:
                return result
            print("  [!] dlib found no face, trying OpenCV Haar...")
        except Exception as e:
            print(f"  [!] dlib failed ({e}), trying OpenCV Haar...")

    return detect_eyes_opencv(image_rgb)


def compute_rotation_angle(left_eye: Point2D, right_eye: Point2D) -> float:
    """Angle (degrees) between the eye line and the horizontal, in [-180, 180]."""
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    assert -180.0 <= angle <= 180.0, f"angle out of range: {angle}"
    return angle


def save_debug_overlay(
    image: Image.Image,
    left_eye: Point2D,
    right_eye: Point2D,
    angle: float,
    detector_name: str,
    debug_path: Path,
) -> None:
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")

    w, h = overlay.size
    radius = max(8, int(min(w, h) * 0.012))
    line_width = max(2, radius // 3)

    lx, ly = left_eye
    rx, ry = right_eye
    cx = (lx + rx) / 2
    cy = (ly + ry) / 2

    draw.line([(lx, ly), (rx, ry)], fill=(255, 80, 80, 220), width=line_width)

    ref_len = (rx - lx) * 0.6
    draw.line(
        [(cx - ref_len, cy), (cx + ref_len, cy)],
        fill=(80, 200, 80, 180),
        width=max(1, line_width - 1),
    )

    for (ex, ey), color in [
        (left_eye, (255, 80, 80, 230)),
        (right_eye, (255, 160, 40, 230)),
    ]:
        draw.ellipse(
            [(ex - radius, ey - radius), (ex + radius, ey + radius)],
            outline=color,
            width=line_width,
        )

    pr = max(4, radius // 2)
    draw.ellipse(
        [(cx - pr, cy - pr), (cx + pr, cy + pr)],
        fill=(255, 255, 80, 220),
    )

    font_size = max(16, int(min(w, h) * 0.025))
    font: Any
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()

    lines = [
        f"Detector : {detector_name}",
        f"Left eye : ({lx:.1f}, {ly:.1f})",
        f"Right eye: ({rx:.1f}, {ry:.1f})",
        f"Tilt     : {angle:.2f}°  →  rotate {-angle:.2f}°",
    ]
    pad = font_size // 2
    box_h = (font_size + 4) * len(lines) + pad * 2
    box_w = max(len(line) for line in lines) * (font_size // 2) + pad * 2
    draw.rectangle((0, 0, box_w, box_h), fill=(0, 0, 0, 160))
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * (font_size + 4)), line, fill=(255, 255, 255, 255), font=font)

    debug_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(debug_path, quality=92, subsampling=0)


def process_image(
    input_path: Path,
    output_path: Path,
    debug_dir: Path | None = None,
) -> bool:
    """Align a single image. Raises DetectionError if eyes cannot be located."""
    image = Image.open(input_path)
    with contextlib.suppress(Exception):
        from PIL import ImageOps

        image = ImageOps.exif_transpose(image)

    img_rgb = np.array(image.convert("RGB"))

    detection = detect_eyes(img_rgb)
    if detection is None:
        raise DetectionError(f"could not detect eyes in {input_path.name}")

    angle = compute_rotation_angle(detection.left_eye, detection.right_eye)

    print(
        f"  [{detection.detector_name}]  "
        f"Left eye ({detection.left_eye[0]:.1f}, {detection.left_eye[1]:.1f})  "
        f"Right eye ({detection.right_eye[0]:.1f}, {detection.right_eye[1]:.1f})  "
        f"→  rotate {-angle:.2f}°"
    )

    if debug_dir is not None:
        debug_path = debug_dir / (input_path.stem + "_debug.jpg")
        save_debug_overlay(
            image,
            detection.left_eye,
            detection.right_eye,
            angle,
            detection.detector_name,
            debug_path,
        )
        print(f"  Debug   → {debug_path}")

    if abs(angle) < 0.1:
        print("  Eyes already level — copying without rotation.")
        rotated = image
    else:
        cx = (detection.left_eye[0] + detection.right_eye[0]) / 2
        cy = (detection.left_eye[1] + detection.right_eye[1]) / 2
        rotated = image.rotate(-angle, resample=Image.BICUBIC, expand=True, center=(cx, cy))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        if rotated.mode == "RGBA":
            rotated = rotated.convert("RGB")
        rotated.save(output_path, quality=95, subsampling=0)
    elif suffix == ".webp":
        rotated.save(output_path, quality=95, method=6)
    else:
        rotated.save(output_path)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Align images so eyes are level.")
    parser.add_argument(
        "input_dir", nargs="?", default="input", help="Input directory (default: input)"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="output/aligned",
        help="Output directory (default: output/aligned)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Save debug overlay images to <output_dir>/debug/"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug" if args.debug else None

    if not input_dir.exists():
        print(f"Error: '{input_dir}' directory not found.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(exist_ok=True)
    ensure_landmarker()
    ensure_dlib_model()

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
            if process_image(img_path, out_path, debug_dir=debug_dir):
                ok += 1
                print(f"  Saved   → {out_path}\n")
            else:
                print()
        except ImageCropperError as e:
            print(f"  [!] {e}\n")
        except Exception as e:
            print(f"  [!] unexpected error: {e}\n", file=sys.stderr)

    print(f"Done. {ok}/{len(images)} image(s) aligned.")


if __name__ == "__main__":
    main()
