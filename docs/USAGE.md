# Usage guide

## pipeline.sh — end-to-end

```
SITSI_COOKIE="…" ./pipeline.sh [options]
```

| Option | Effect |
|---|---|
| _(none)_ | Download from sortitoutsi queue, then process |
| `--skip-download` | Skip download; read from `input/` |
| `--input <dir>` | Use a custom input directory (implies `--skip-download`) |
| `-h / --help` | Print usage |

The pipeline:
1. Creates a throwaway venv in `tmp_pipeline/.venv`, installs `requirements.txt`
2. Runs all 6 steps in sequence
3. Writes final PNGs to `output/final/`
4. Removes `tmp_pipeline/` and the downloaded `input/` on exit (success or failure)

---

## Step scripts (standalone)

All scripts default to `output/<stage>/` when run without arguments.

### download_queue.py

Downloads up to 50 pending source images from the sortitoutsi.net submission queue.

```bash
SITSI_COOKIE="laravel_session=…" python download_queue.py
```

Output: `input/<alt-name>.<ext>`

Requires the `SITSI_COOKIE` environment variable (session cookies from your
browser). See README for how to extract them.

---

### align.py

Rotates images so both eyes land on the same horizontal line.

```bash
python align.py [input_dir] [output_dir] [--debug]
```

| Argument | Default |
|---|---|
| `input_dir` | `input/` |
| `output_dir` | `output/aligned/` |
| `--debug` | Write annotated debug overlays to `<output_dir>/debug/` |

Detector cascade: MediaPipe Iris → MediaPipe Eye Contour → dlib 68-point → OpenCV Haar.

Images where no eyes are detected are skipped (a warning is printed).

---

### crop_source.py

Detects the face and crops to include head, hair, and shoulders.
Minimum output size: 500×500 px; no downscaling.

```bash
python crop_source.py [input_dir] [output_dir]
```

| Argument | Default |
|---|---|
| `input_dir` | `input/` |
| `output_dir` | `output/cropped/` |

Detector cascade: MediaPipe BlazeFace → OpenCV Haar cascade.

Padding constants (relative to face height):

| Constant | Value | Meaning |
|---|---|---|
| `HAIR_PADDING_FACTOR` | 1.1× | Space above top of face (covers tall hair) |
| `SHOULDER_PADDING_FACTOR` | 0.8× | Space below chin |
| `SIDE_PADDING_FACTOR` | 0.35× | Horizontal margin on each side |

---

### remove_background.py

Removes the background, producing a transparent-background PNG.
Uses a two-model BiRefNet ensemble with optional flip TTA and guided-filter
edge refinement. This is the slowest step.

```bash
python remove_background.py [--input DIR] [--output DIR] [options]
```

| Flag | Default | Description |
|---|---|---|
| `--input` | `output/cropped/` | Input directory |
| `--output` | `output/transparent/` | Output directory |
| `--device` | `auto` | `auto` \| `mps` \| `cuda` \| `cpu` |
| `--no-tta` | — | Disable horizontal-flip test-time augmentation |
| `--no-ensemble` | — | Use only matting model (skip salient model) |
| `--no-refine` | — | Disable guided-filter edge refinement |
| `--refine-radius` | `4` | Guided-filter radius (px) |
| `--refine-eps` | `1e-4` | Guided-filter regularisation |
| `--input-size` | `2048` | Model input resolution (lower if OOM) |
| `--limit N` | `0` (all) | Process only first N images |
| `--overwrite` | — | Re-process even if output exists |

Tips:
- On Apple Silicon, `auto` selects MPS. If you hit memory errors, try
  `--input-size 1024`.
- `--no-ensemble` halves the number of model passes; quality drops slightly
  but throughput doubles.

---

### crop_cutout.py

Crops a transparent-background image to an exact 250×250 portrait: top of
head (from alpha channel) through shirt neckline, face horizontally centered.

```bash
python crop_cutout.py [input] [output]
```

`input` can be a file or directory; `output` can be a file or directory.

| Argument | Default |
|---|---|
| `input` | `input/` |
| `output` | `output/portrait/` (directory mode) |

Images without enough pixels below the chin to show a shirt are skipped.

Detector cascade: MediaPipe Face Landmarker → dlib 68-point.

---

### deglow.py

Removes the bright fringe ("glow" or "halo") that forms at alpha edges when a
subject was photographed against a light background.

```bash
python deglow.py <input> [output] [--strength N] [--radius N] [--overwrite]
```

| Argument | Default | Description |
|---|---|---|
| `input` | _(required)_ | PNG/WebP file or directory |
| `output` | `<input_dir>_dg/` | Output file or directory |
| `--strength` | `1.0` | Global multiplier (0 = off, 2 = aggressive) |
| `--radius` | `8` | Max fringe radius in pixels |
| `--overwrite` | — | Re-process existing outputs |

Strength is automatically adjusted per pixel based on anchor luminance: dark
hair gets maximum correction, blond/pale subjects get a gentler touch.

---

### finalize-cutout.py

Utility script: centers an existing cutout PNG onto a fresh 250×250
transparent canvas. Useful for hand-edited cutouts that need re-centering.

```bash
python finalize-cutout.py <input> [output]
```

`input` can be a single PNG or a directory of PNGs.
