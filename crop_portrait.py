#!/usr/bin/env python3
"""
Portrait cropper: crops images from top-of-head down through the shirt neckline,
horizontally centered so both ears have equal margins to the canvas edge.
Output: 250×250 PNG.

Only processes images that have enough pixels below the chin to include a shirt
area — images cropped tight to the chin/neck are skipped.

Usage:
  python crop_portrait.py                    # input/ → output/portrait/
  python crop_portrait.py src/ out/          # custom dirs
  python crop_portrait.py image.png out/     # single file
"""

import argparse
import sys
import urllib.request
import numpy as np
from pathlib import Path
from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

LANDMARKER_PATH = Path(__file__).parent / "face_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

DLIB_MODEL_PATH = Path(__file__).parent / "shape_predictor_68_face_landmarks.dat"

OUTPUT_SIZE = 250

# Fallback: extra skull height above the forehead landmark when there is no alpha channel
HEAD_EXTRA_FRACTION = 0.18

# How many output pixels (at 250×250) to include below the chin
CHIN_PIXELS_AT_OUTPUT = 10

# Minimum pixels of shirt that must be available below the chin to process the image
MIN_SHIRT_PIXELS = 20

# Horizontal padding added outside the widest detected face point on each side
EAR_SIDE_PADDING_FRACTION = 0.08

# Alpha threshold: pixels with alpha below this are considered empty/background
ALPHA_THRESHOLD = 10


def find_hair_top(image: Image.Image) -> int | None:
    """
    Scan from the top row down and return the y-coordinate of the first row
    that contains at least one non-transparent pixel (alpha > ALPHA_THRESHOLD).
    Returns None if the image has no alpha channel.
    """
    if image.mode != "RGBA":
        return None
    alpha = np.array(image)[:, :, 3]  # shape (H, W)
    rows_with_content = np.any(alpha > ALPHA_THRESHOLD, axis=1)
    hits = np.where(rows_with_content)[0]
    return int(hits[0]) if len(hits) else None


def find_beard_bottom(
    image: Image.Image,
    chin_y: float,
    left_x: float,
    right_x: float,
    face_height: float,
) -> int | None:
    """
    Scan downward from chin_y within the central face band and return the
    y-coordinate of the lowest non-transparent row (beard tip).

    Search is capped at half a face-height below the chin so that shirt pixels
    (which are also non-transparent) are never mistaken for beard.
    Returns None if the image has no alpha channel or no beard is found.
    """
    if image.mode != "RGBA":
        return None
    alpha = np.array(image)[:, :, 3]
    h, w = alpha.shape

    face_center_x = (left_x + right_x) / 2
    face_half_width = (right_x - left_x) / 2
    band_left = max(0, int(face_center_x - face_half_width * 0.6))
    band_right = min(w, int(face_center_x + face_half_width * 0.6))

    chin_row = max(0, int(chin_y))
    max_row = min(h, int(chin_y + face_height * 0.5))
    if chin_row >= h:
        return None

    band = alpha[chin_row:max_row, band_left:band_right]
    rows_with_content = np.any(band > ALPHA_THRESHOLD, axis=1)
    hits = np.where(rows_with_content)[0]
    return chin_row + int(hits[-1]) if len(hits) else None


def ensure_landmarker() -> bool:
    if LANDMARKER_PATH.exists():
        return True
    print(f"Downloading MediaPipe Face Landmarker → {LANDMARKER_PATH.name} ...")
    try:
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)
        print("  Model downloaded.\n")
        return True
    except Exception as e:
        print(f"  [!] Could not download landmarker model: {e}")
        return False


# ---------------------------------------------------------------------------
# Landmark detection
# ---------------------------------------------------------------------------


def _landmarks_mediapipe(image_rgb: np.ndarray):
    """
    Returns list of (x, y) pixel coordinates for all 478 landmarks, or None.
    Index 10  → top of forehead
    Index 152 → bottom of chin
    """
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks import python as mp_tasks

    h, w = image_rgb.shape[:2]
    base_opts = mp_tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH))
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
    return [(lm.x * w, lm.y * h) for lm in lms]


def _landmarks_dlib(image_rgb: np.ndarray):
    """
    Returns a dict with keys: chin_y, forehead_y, left_x, right_x.
    Uses dlib 68-point; forehead is estimated from eye + chin geometry.
    """
    import dlib
    import cv2

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(DLIB_MODEL_PATH))

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dets = detector(gray, 1)
    if not dets:
        return None

    det = max(dets, key=lambda d: d.area())
    shape = predictor(gray, det)
    pts = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

    # Chin: point 8 (bottom-centre of jaw)
    chin_y = pts[8][1]

    # Eyes: points 36-41 (left eye), 42-47 (right eye)
    eye_ys = [pts[i][1] for i in range(36, 48)]
    eye_y = np.mean(eye_ys)

    # Estimate forehead: eyes are roughly 60% of the way from hairline to chin
    # → hairline_y = eye_y - 0.6 * (chin_y - eye_y) / 0.6 ... simpler:
    # face_h ≈ chin_y - eye_y, hairline ≈ eye_y - 0.5 * face_h
    face_h = chin_y - eye_y
    forehead_y = eye_y - 0.5 * face_h

    # Horizontal extent from jaw outline (points 0 and 16)
    left_x = pts[0][0]
    right_x = pts[16][0]

    return {
        "chin_y": float(chin_y),
        "forehead_y": float(forehead_y),
        "left_x": float(left_x),
        "right_x": float(right_x),
        "face_height": float(face_h),
    }


def detect_face_geometry(image_rgb: np.ndarray):
    """
    Returns dict with chin_y, forehead_y, left_x, right_x, face_height (pixels).
    Tries MediaPipe first, then dlib.
    """
    # --- MediaPipe ---
    if LANDMARKER_PATH.exists():
        try:
            lms = _landmarks_mediapipe(image_rgb)
            if lms:
                # Chin: landmark 152, forehead: landmark 10
                chin_y = lms[152][1]
                forehead_y = lms[10][1]

                # Widest x across ALL landmarks approximates ear-to-ear span
                all_x = [p[0] for p in lms]
                left_x = min(all_x)
                right_x = max(all_x)

                face_height = chin_y - forehead_y
                return {
                    "chin_y": chin_y,
                    "forehead_y": forehead_y,
                    "left_x": left_x,
                    "right_x": right_x,
                    "face_height": face_height,
                    "detector": "MediaPipe",
                }
        except Exception as e:
            print(f"  [!] MediaPipe failed ({e}), trying dlib ...")

    # --- dlib fallback ---
    if DLIB_MODEL_PATH.exists():
        try:
            geo = _landmarks_dlib(image_rgb)
            if geo:
                geo["detector"] = "dlib"
                return geo
        except Exception as e:
            print(f"  [!] dlib failed ({e})")

    return None


# ---------------------------------------------------------------------------
# Crop computation
# ---------------------------------------------------------------------------


def compute_crop(
    geo: dict,
    img_w: int,
    img_h: int,
    hair_top_y: int | None = None,
    beard_bottom_y: int | None = None,
):
    """
    Returns (left, top, right, bottom) crop box, or None if the image doesn't
    have enough pixels below the chin to include a shirt area.

    hair_top_y: first non-transparent row (from find_hair_top). When provided
    this is used as the crop top; otherwise falls back to landmark extrapolation.

    beard_bottom_y: lowest non-transparent row in the face band below the chin
    (from find_beard_bottom). When present the crop bottom extends to cover the
    beard before adding the CHIN_PIXELS_AT_OUTPUT margin.
    """
    chin_y = geo["chin_y"]
    forehead_y = geo["forehead_y"]
    face_h = geo["face_height"]
    left_x = geo["left_x"]
    right_x = geo["right_x"]

    # Use the beard tip only when it extends meaningfully below the anatomical chin
    # (≥10% of face height below chin_y = real beard, not just chin-pixel rounding)
    beard_threshold = chin_y + face_h * 0.10
    if beard_bottom_y is not None and beard_bottom_y > beard_threshold:
        bottom_y = float(beard_bottom_y)
    else:
        bottom_y = chin_y

    # --- Shirt check: must have enough image below the lowest facial point ---
    pixels_below_chin = img_h - bottom_y
    if pixels_below_chin < MIN_SHIRT_PIXELS:
        return None  # no shirt visible → skip

    # --- Top of head ---
    if hair_top_y is not None:
        crop_top = float(hair_top_y)
    else:
        # Fallback: extrapolate above the forehead landmark
        head_extra = face_h * HEAD_EXTRA_FRACTION
        crop_top = forehead_y - head_extra
        crop_top = max(0, crop_top)

    # --- Bottom: chin/beard tip + a tiny margin (CHIN_PIXELS_AT_OUTPUT px at 250×250 scale) ---
    # Scale the margin by how many full-res pixels map to one output pixel.
    # Use the hair-to-bottom span as the reference vertical extent.
    vertical_span = bottom_y - crop_top
    chin_margin = vertical_span * (CHIN_PIXELS_AT_OUTPUT / OUTPUT_SIZE)
    crop_bottom = min(bottom_y + chin_margin, img_h)

    # --- Horizontal: center on face midpoint, add side padding ---
    face_center_x = (left_x + right_x) / 2
    half_face_width = (right_x - left_x) / 2
    side_pad = half_face_width * EAR_SIDE_PADDING_FRACTION

    crop_height = crop_bottom - crop_top

    # Width = height for 1:1; then ensure it covers the ears + padding
    min_half_width = half_face_width + side_pad
    half_w = max(crop_height / 2, min_half_width)

    # If ears require more width than height, expand to square based on width
    crop_width = half_w * 2
    crop_height = max(crop_height, crop_width)  # keep square (side-driven expansion)
    half_w = crop_height / 2  # recompute so it's square

    crop_left = face_center_x - half_w
    crop_right = face_center_x + half_w

    # Recenter top/bottom around the original crop centre if height expanded
    crop_center_y = (crop_top + crop_bottom) / 2
    crop_top = crop_center_y - crop_height / 2
    crop_bottom = crop_center_y + crop_height / 2

    return (
        int(round(crop_left)),
        int(round(crop_top)),
        int(round(crop_right)),
        int(round(crop_bottom)),
    )


def crop_with_padding(image: Image.Image, left, top, right, bottom) -> Image.Image:
    """Crop, padding with black where the box extends outside the image."""
    img_w, img_h = image.size
    crop_w = right - left
    crop_h = bottom - top

    # Fast path: entirely within bounds
    if left >= 0 and top >= 0 and right <= img_w and bottom <= img_h:
        return image.crop((left, top, right, bottom))

    # Clamp to image and figure out where to paste
    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(img_w, right)
    src_bottom = min(img_h, bottom)

    paste_x = src_left - left
    paste_y = src_top - top

    mode = image.mode if image.mode in ("RGB", "RGBA") else "RGB"
    canvas = Image.new(mode, (crop_w, crop_h), color=0)
    patch = image.crop((src_left, src_top, src_right, src_bottom))
    canvas.paste(patch, (paste_x, paste_y))
    return canvas


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_image(input_path: Path, output_path: Path) -> bool:
    image = Image.open(input_path)
    try:
        from PIL import ImageOps

        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    img_rgb = np.array(image.convert("RGB"))
    img_h, img_w = img_rgb.shape[:2]

    # Find the actual top of the hair from the alpha channel (if present)
    hair_top_y = find_hair_top(image)

    geo = detect_face_geometry(img_rgb)
    if geo is None:
        print(f"  [!] No face detected — skipping")
        return False

    beard_bottom_y = find_beard_bottom(
        image, geo["chin_y"], geo["left_x"], geo["right_x"], geo["face_height"]
    )

    box = compute_crop(geo, img_w, img_h, hair_top_y=hair_top_y, beard_bottom_y=beard_bottom_y)
    if box is None:
        print(f"  [skip] Not enough shirt below chin — skipping")
        return False

    left, top, right, bottom = box
    crop_size = right - left  # square

    hair_src = (
        f"alpha scan row {hair_top_y}"
        if hair_top_y is not None
        else "landmark extrapolation"
    )
    beard_threshold = geo["chin_y"] + geo["face_height"] * 0.10
    beard_info = f"  beard_bottom={beard_bottom_y}" if beard_bottom_y is not None and beard_bottom_y > beard_threshold else ""
    print(
        f"  [{geo['detector']}]  "
        f"hair_top={top} ({hair_src})  chin={geo['chin_y']:.0f}{beard_info}  "
        f"face_h={geo['face_height']:.0f}  "
        f"crop=({left},{top})→({right},{bottom})  size={crop_size}"
    )

    cropped = crop_with_padding(image, left, top, right, bottom)
    resized = cropped.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(output_path, "PNG")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Crop portraits to 250×250: top-of-head through shirt, face centered."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="input",
        help="Input image file or directory (default: input/)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output file or directory (default: output/portrait/ or same dir for single file)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    # Single file mode
    if input_path.is_file():
        if args.output:
            out_path = Path(args.output)
            if out_path.is_dir() or not out_path.suffix:
                out_path = out_path / (input_path.stem + "_portrait.png")
        else:
            out_path = input_path.parent / (input_path.stem + "_portrait.png")

        ensure_landmarker()
        print(f"Processing: {input_path.name}")
        try:
            if process_image(input_path, out_path):
                print(f"  Saved → {out_path}")
            else:
                print(f"  Not saved.")
        except Exception as e:
            print(f"  [!] Error: {e}")
        return

    # Directory mode
    if not input_path.is_dir():
        print(f"Error: '{input_path}' is not a file or directory.")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path("output") / "portrait"
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_landmarker()

    images = [
        p
        for p in sorted(input_path.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not images:
        print(f"No supported images found in {input_path}/")
        sys.exit(0)

    print(f"Found {len(images)} image(s) in {input_path}/\n")
    ok = skipped = errors = 0
    for img_path in images:
        out_path = output_dir / (img_path.stem + ".png")
        print(f"Processing: {img_path.name}")
        try:
            if process_image(img_path, out_path):
                ok += 1
                print(f"  Saved → {out_path}\n")
            else:
                skipped += 1
                print()
        except Exception as e:
            errors += 1
            print(f"  [!] Error: {e}\n")

    print(
        f"Done. {ok} saved / {skipped} skipped (no shirt) / {errors} errors  — total {len(images)}"
    )


if __name__ == "__main__":
    main()
