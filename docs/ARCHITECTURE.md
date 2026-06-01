# Architecture

## Overview

The pipeline turns raw source images — typically scraped from the
sortitoutsi.net queue — into 250×250 transparent-background portrait PNGs
suitable for Football Manager face packs.

```
[sortitoutsi.net queue]
        │
        ▼
  download_queue.py   →  input/
        │
        ▼
    align.py          →  output/aligned/    (eye-level rotation)
        │
        ▼
  crop_source.py      →  output/cropped/    (head + shoulders bbox)
        │
        ▼
remove_background.py  →  output/transparent/ (RGBA, AI matting)
        │
        ▼
  crop_cutout.py      →  output/portrait/   (250×250 portrait)
        │
        ▼
   deglow.py          →  output/final/      (fringe correction)
```

`pipeline.sh` automates all six steps, manages a throwaway venv, and cleans
up intermediate directories on exit.

---

## Script responsibilities

### download_queue.py

Scrapes the sortitoutsi.net pending-submissions queue and downloads up to 50
images. Uses a session cookie passed via `SITSI_COOKIE` to authenticate.
Images are saved as `input/<alt-name>.<ext>` where the alt-name comes from
the HTML `alt` attribute of each image.

### align.py

Rotates the image so the eye line is horizontal. The correction angle is
computed from the centroids of the left and right eye landmarks.

Detector cascade (first success wins):
1. **MediaPipe Face Landmarker + iris** — most accurate, requires
   `face_landmarker.task` (auto-downloaded on first run)
2. **MediaPipe Eye Contour fallback** — same model, fewer landmark points
3. **dlib 68-point** — requires `shape_predictor_68_face_landmarks.dat`
   (auto-downloaded, ~100 MB)
4. **OpenCV Haar cascade** — always available, least accurate

### crop_source.py

Detects the face and expands the bounding box with fixed padding multipliers
to include hair and shoulders. A minimum output size of 500×500 px is
enforced; small faces are expanded rather than upscaled.

Detector cascade:
1. **MediaPipe BlazeFace** (short-range TFLite model, auto-downloaded)
2. **OpenCV Haar cascade** — always available

### remove_background.py

The most computationally expensive step. Strategy:

1. Both models run on the letterbox-padded image at 2048×2048
2. Optional horizontal-flip TTA (test-time augmentation) runs each model
   again on the mirrored image and flips the result back
3. The four alpha maps (2 models × 2 orientations) are mean-ensembled
4. The ensembled alpha is bicubic-upscaled to the original image resolution
   after cropping out the letterbox padding
5. A multi-channel guided filter (He et al. 2010) refines alpha edges using
   the original RGB as the guide — this is what gives clean hair edges

**Models**:
- `ZhengPeng7/BiRefNet_HR-matting` — trained for alpha matting, soft edges
- `ZhengPeng7/BiRefNet_HR` — trained for salient object detection, sharp body

Both are loaded from Hugging Face Hub (cached in `~/.cache/huggingface/`).

**Device selection**: auto-detects MPS (Apple Silicon) → CUDA → CPU.
The guided filter deliberately runs on CPU because MPS triggers an IOGPU
shared-memory assertion on macOS.

### crop_cutout.py

Uses face landmarks to find the exact crop box:
- **Top**: first non-transparent row in the alpha channel (actual hair top)
  or, if no alpha, a landmark-extrapolated estimate above the forehead
- **Bottom**: chin landmark + a fixed margin proportional to the head height
  (equivalent to ~10 px at 250 output size)
- **Horizontal**: centered on the face midpoint, wide enough to include ears
  with a small padding factor
- **Square**: the crop box is always square (longer axis wins)

Images where there are fewer than 20 pixels below the chin are skipped; they
don't show enough shirt to make a valid portrait.

Detector cascade:
1. **MediaPipe Face Landmarker** (same model as `align.py`)
2. **dlib 68-point** (same model as `align.py`)

### deglow.py

Semi-transparent alpha-edge pixels often retain light from the original
background ("glow"). The script:

1. Finds the nearest fully-opaque pixel for every fringe pixel using a
   Euclidean distance transform
2. Uses that pixel as a colour anchor
3. Blends each fringe pixel toward its anchor at a strength that scales with
   anchor luminance: dark subjects (black hair) get maximum correction;
   blond / pale subjects get a gentler touch (quadratic ramp)
4. Additionally suppresses excess brightness beyond the blend

### finalize-cutout.py

Standalone utility. Crops the tight bounding box of non-transparent pixels,
scales down if larger than 250×250, and centers on a fresh transparent canvas.
Not part of the main pipeline; used for ad-hoc re-centering of hand-edited
cutouts.

---

## Data formats

| Stage | Format | Notes |
|---|---|---|
| Input | JPEG / PNG / WebP / BMP / TIFF | Any bit depth; EXIF orientation respected |
| After align | Same as input | Rotated with bicubic resampling |
| After crop_source | Same as input | JPEG quality 95 when saved as JPEG |
| After remove_background | PNG (RGBA) | Lossless, compress level 9 |
| After crop_cutout | PNG (RGBA) | 250×250, LANCZOS resize |
| After deglow | PNG (RGBA) | Final delivery format |

---

## Fallback strategy

Every detection step has at least two fallbacks so the pipeline degrades
gracefully rather than failing hard. If neither model detects a face, the
image is skipped (printed with `[!]`) and processing continues.

Model weights are downloaded automatically on first run and cached next to
the scripts. They are gitignored so the repository stays lightweight.

---

## Performance

| Step | Typical time per image |
|---|---|
| download | network-bound |
| align | < 1 s |
| crop_source | < 1 s |
| remove_background (MPS, 2048) | 15 – 40 s |
| crop_cutout | < 1 s |
| deglow | < 1 s |

Background removal dominates. Use `--no-ensemble` or `--input-size 1024` to
trade quality for speed.
