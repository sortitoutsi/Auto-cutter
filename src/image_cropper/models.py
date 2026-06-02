"""
Model file resolution.

Lookup order for each model:
    1. Bundled package data (image_cropper/data/)
    2. User cache dir (~/Library/Caches/image-cropper on macOS,
       %LOCALAPPDATA%\\image-cropper on Windows,
       $XDG_CACHE_HOME/image-cropper on Linux)
    3. Auto-download to the user cache dir.
"""
from __future__ import annotations

import bz2
import os
import sys
import urllib.request
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"

_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_DLIB_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
_FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def user_cache_dir() -> Path:
    """Per-OS cache directory for downloaded models."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "image-cropper"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve(name: str, url: str, *, bz2_decompress: bool = False) -> Path:
    """Return path to a model file, downloading to user cache if needed."""
    bundled = _DATA_DIR / name
    if bundled.exists():
        return bundled

    cached = user_cache_dir() / name
    if cached.exists():
        return cached

    # Also accept the legacy location (project root, for source installs)
    legacy = Path(__file__).resolve().parents[2] / name
    if legacy.exists():
        return legacy

    print(f"Downloading {name} → {cached} ...", flush=True)
    try:
        if bz2_decompress:
            bz2_path = cached.with_suffix(cached.suffix + ".bz2")
            urllib.request.urlretrieve(url, bz2_path)
            with bz2.open(bz2_path, "rb") as src, open(cached, "wb") as dst:
                dst.write(src.read())
            bz2_path.unlink()
        else:
            urllib.request.urlretrieve(url, cached)
        print(f"  Downloaded.\n", flush=True)
        return cached
    except Exception as e:
        print(f"  [!] Download failed: {e}", flush=True)
        # Return the (non-existent) cache path so callers can detect failure
        return cached


def face_landmarker_path() -> Path:
    """MediaPipe Face Landmarker (478 points + iris). Bundled with the package."""
    return _resolve("face_landmarker.task", _LANDMARKER_URL)


def dlib_model_path() -> Path:
    """dlib 68-point shape predictor (~95 MB, auto-downloaded)."""
    return _resolve("shape_predictor_68_face_landmarks.dat", _DLIB_URL, bz2_decompress=True)


def face_detector_path() -> Path:
    """MediaPipe BlazeFace short-range detector (~250 KB, auto-downloaded)."""
    return _resolve("face_detector.tflite", _FACE_DETECTOR_URL)
