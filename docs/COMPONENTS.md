# Component Reference

A module-by-module breakdown of the codebase.  Read this alongside
[ARCHITECTURE.md](ARCHITECTURE.md) (pipeline flow) and
[DECISIONS.md](DECISIONS.md) (why things are the way they are).

---

## Core modules

These modules contain no pipeline logic — they are shared infrastructure
imported by everything else.

### `types.py`

Immutable value objects shared across all pipeline stages.

| Type | Kind | Purpose |
|------|------|---------|
| `Point2D` | `tuple[float, float]` | A single (x, y) pixel coordinate |
| `EyeDetection` | frozen dataclass | Left + right eye centroids from a detector |
| `FaceBBox` | frozen dataclass | Axis-aligned face bounding box (x, y, w, h) |
| `FaceGeometry` | frozen dataclass | Landmark-derived measurements used by the portrait crop |
| `CropBox` | frozen dataclass | Crop region in LTRB coordinates with helpers |
| `MetricsDict` | TypedDict | Six quality metrics, JSON-serialisable |
| `QueueEntry` | TypedDict | One image entry scraped from the sortitoutsi queue |
| `SubmissionMeta` | TypedDict | Sidecar metadata written next to downloaded images |
| `SubmitResult` | TypedDict | Return value from `submit_cutout()` |

`EyeDetection`, `FaceBBox`, `FaceGeometry`, and `CropBox` use
`@dataclass(frozen=True, slots=True)` so they are immutable and
memory-efficient.  `MetricsDict`, `QueueEntry`, `SubmissionMeta`, and
`SubmitResult` use `TypedDict` so they serialise directly with `json.dump()`
without a custom encoder.

`CropBox` notable methods:
- `CropBox.from_ltrb(left, top, right, bottom)` — preferred constructor
- `.clamp_to_image(img_w, img_h)` — clip box to image bounds
- `.as_tuple()` — returns `(left, top, right, bottom)` for PIL

---

### `errors.py`

Single exception hierarchy; every error raised inside the package inherits
from `ImageCropperError`.

```
ImageCropperError
├── ValidationError        — bad input (file missing, wrong format, zero-area crop)
├── DetectionError         — detector found nothing (no face, no eyes)
├── ModelError             — model file missing or failed to load
│   └── BackgroundRemovalError  — BiRefNet pipeline failure
└── PipelineError          — prior stage output missing or malformed
```

CLI entry points catch `ImageCropperError` and exit with code 1.  The GUI
`PipelineWorker` lets the subprocess propagate the exit code and logs
stderr.

---

### `validation.py`

Boundary validators that raise typed errors rather than letting raw
exceptions bubble up from deep in Pillow or numpy.

| Function | Raises | What it checks |
|----------|--------|----------------|
| `validate_input_path(p, supported_exts)` | `ValidationError` | file exists, readable, correct extension |
| `validate_output_path(p)` | `ValidationError` | parent dir exists and is writable (creates it if needed) |
| `ensure_rgba(img)` | `ValidationError` | converts PIL Image to RGBA, rejects non-Image inputs |
| `validate_image_array(arr, channels, dtype)` | `ValidationError` | numpy array is (H, W, channels) with correct dtype |
| `validate_crop_box(box, img_w, img_h)` | `ValidationError` | box is non-degenerate and intersects the image |

---

### `models.py`

Resolves ML model file paths using a three-level lookup:

1. **Bundled** — `src/image_cropper/data/` (ships with the package)
2. **User cache** — OS-specific cache directory:
   - macOS: `~/Library/Caches/image-cropper/`
   - Linux: `$XDG_CACHE_HOME/image-cropper/` (fallback: `~/.cache/image-cropper/`)
   - Windows: `%LOCALAPPDATA%\image-cropper\`
3. **Auto-download** — fetched to user cache on first run

Public functions:

| Function | Returns | Notes |
|----------|---------|-------|
| `user_cache_dir()` | `Path` | Per-OS cache location |
| `face_landmarker_path()` | `Path` | MediaPipe 478-point face landmarker (bundled) |
| `dlib_model_path()` | `Path` | dlib 68-point shape predictor (~95 MB, auto-downloaded) |
| `face_detector_path()` | `Path` | MediaPipe BlazeFace short-range detector (~250 KB, auto-downloaded) |

---

### `sitsi_client.py`

Centralised, security-hardened HTTP client for sortitoutsi.net.
All network calls go through here so SSRF guards live in one place.

| Function | Purpose |
|----------|---------|
| `validate_cookie_string(raw)` | Parse and validate a browser cookie string (RFC 6265) |
| `validate_image_url(url)` | Reject URLs not on `sortitoutsi.net` (SSRF guard) |
| `validate_sitsi_url(url)` | Same check for non-image sortitoutsi pages |
| `get_session(cookie_str)` | Return a `requests.Session` with cookies, User-Agent, Referer pre-set |
| `get_csrf_token(session, page_url)` | Fetch a page and extract the Laravel CSRF token |
| `get_hidden_form_fields(soup, form_selector)` | Return all hidden `<input>` values from the first matching form |

`get_session("")` returns an unauthenticated session (empty cookie string),
used by `DownloadWorker` when no cookie is provided.

---

## Pipeline modules

Run in this order.  Each module is a standalone CLI script and can be invoked
as `python -m image_cropper.pipeline.<module>` or via its installed entry point.

```
download_queue  →  align  →  crop_source  →  remove_background
                                                    ↓
                          finalize_cutout  ←  deglow  ←  crop_cutout
                                ↓
                          submit_cutout   (optional, uploads to sortitoutsi)
```

---

### Step 0 — `download_queue.py` (`ic-download`)

**Purpose**: Scrape pending submissions from sortitoutsi.net and download
source images.

**Two modes**:

| Mode | Trigger | What it does |
|------|---------|--------------|
| Queue | default | Scrapes the pending-submissions queue, downloads up to `MAX_IMAGES` (50) |
| Collection | `--collection URL` | Scrapes a specific collection page; writes `.sitsi.json` sidecars |

**Key functions**:

- `strip_size_params(url)` — removes `?width=N&height=N` from image URLs to get full resolution
- `safe_filename(name)` — sanitises alt-text into a valid filename (replaces illegal chars with `_`)
- `guess_extension(url, content_type)` — infers `.jpg`/`.png`/etc from URL path then Content-Type
- `collect_image_entries(session)` — queue mode scraper, paginates until `MAX_IMAGES` reached
- `collect_collection_entries(session, collection_url)` — collection mode scraper; returns `(QueueEntry, SubmissionMeta)` pairs
- `download_images(session, entries, out_dir, metas=None)` — bulk downloader; writes sidecars when `metas` is provided

**Authentication**: Requires `SITSI_COOKIE` env var (browser session cookie).

**Sidecar format** (`.sitsi.json` next to each image):
```json
{
  "submission_id": 12345,
  "person_id": 6789,
  "alt": "Player Name",
  "status": "pending",
  "image_type": "source",
  "collection_url": "https://sortitoutsi.net/...",
  "downloaded_at": "2026-01-01T12:00:00+00:00"
}
```

---

### Step 1 — `align.py` (`ic-align`)

**Purpose**: Rotate images so both eyes are level (eye-line is horizontal).

**Detector cascade** (first success wins):
1. MediaPipe Face Landmarker with iris refinement (478 points, most accurate)
2. MediaPipe eye contour fallback (same model, fewer points)
3. dlib 68-point shape predictor
4. OpenCV Haar cascade (always available, least accurate)

**Key functions**:

- `detect_eyes(image_rgb)` → `EyeDetection | None` — runs cascade, returns image-left/right centroid pair
- `compute_rotation_angle(left_eye, right_eye)` → `float` — degrees to rotate, in `[-180, 180]`
- `process_image(input_path, output_path, debug_dir=None)` → `bool` — aligns one image; writes debug overlay when `debug_dir` is set

**Skipping**: Images where no detector finds eyes are skipped (no output written).

**Debug mode** (`--debug`): writes `<output_dir>/debug/<stem>_debug.jpg` with annotated eye positions and rotation angle.

---

### Step 2 — `crop_source.py` (`ic-crop-face`)

**Purpose**: Detect the face and crop to include full hair and shoulders.
Minimum output size is 500×500 px.

**Detector cascade**:
1. MediaPipe BlazeFace short-range TFLite model
2. OpenCV Haar cascade fallback

**Padding multipliers** applied to the detected face bounding box:

| Direction | Multiplier | Rationale |
|-----------|-----------|-----------|
| Above (hair) | 1.1× face height | Includes full hair |
| Below (shoulders) | 0.8× face height | Includes shirt collar |
| Sides | 0.35× face width each | Includes ears with margin |

**Key functions**:

- `detect_face(image_rgb)` → `FaceBBox | None` — cascade detector
- `compute_crop_box(face_x, face_y, face_w, face_h, img_w, img_h)` → `tuple[int,int,int,int]` — applies padding, enforces `MIN_OUTPUT_SIZE`
- `process_image(input_path, output_path)` → `bool`

**Constants**: `MIN_OUTPUT_SIZE = 500`

---

### Step 3 — `remove_background.py` (`ic-remove-bg`)

**Purpose**: High-quality background removal with clean hair edges using an
ensemble of two BiRefNet models.

**Pipeline**:
1. Letterbox-pad image to a square at `--input-size` (default 2048)
2. Run both models; optionally run both again on the horizontal flip (TTA)
3. Mean-ensemble the alpha maps (2–4 maps depending on TTA setting)
4. Bicubic-upsample ensembled alpha back to original resolution
5. Multi-channel guided filter (He et al. 2010) refines edges using original RGB
6. Compose RGBA and save lossless PNG

**Models** (loaded from Hugging Face Hub, cached in `~/.cache/huggingface/`):

| Model | Trained for | Role in ensemble |
|-------|-------------|-----------------|
| `ZhengPeng7/BiRefNet_HR-matting` | Alpha matting | Soft edges, good for hair |
| `ZhengPeng7/BiRefNet_HR` | Salient object detection | Sharp body boundary |

**Device auto-detection**: MPS (Apple Silicon) → CUDA → CPU.
The guided filter always runs on CPU because MPS triggers an IOGPU
shared-memory assertion on macOS.

**Key CLI flags**:

| Flag | Default | Effect |
|------|---------|--------|
| `--device` | auto | `cpu`, `cuda`, or `mps` |
| `--no-tta` | off | Disable horizontal-flip TTA (halves model calls) |
| `--no-ensemble` | off | Use only the matting model |
| `--no-refine` | off | Skip guided filter (faster, rougher edges) |
| `--input-size N` | 2048 | Resize to N×N before model inference |

**Key functions**:

- `predict_alpha(image_rgb, models, device, tta_flip, size)` → alpha tensor — full ensemble
- `guided_filter_multichannel(guide_chw, src_hw, radius, eps)` — 3-channel guided filter
- `guided_refine(alpha_hw, image_rgb, device, radius, eps)` — edge-aware alpha refinement
- `process_image(...)` — removes background from a single file

---

### Step 4 — `crop_cutout.py` (`ic-crop-portrait`)

**Purpose**: Crop from the top of the head through the shirt neckline into a
250×250 square PNG.

**Crop box logic**:

| Edge | Determined by |
|------|--------------|
| Top | First non-transparent row in the alpha channel (actual hair top) or landmark extrapolation above forehead |
| Bottom | Chin landmark + `--chin-pixels` (default 10 px at 250 output) |
| Horizontal | Centred on face midpoint, wide enough to include ears + `EAR_SIDE_PADDING_FRACTION` (0.08) |
| Aspect | Always square — longer axis wins |

**Detector cascade**:
1. MediaPipe Face Landmarker (478-point, same model as `align.py`)
2. dlib 68-point shape predictor

**Skipping**: Images with fewer than `MIN_SHIRT_PIXELS` (20 px) below the chin
are skipped — they don't show enough shirt for a valid portrait.

**Constants**: `OUTPUT_SIZE = 250`, `CHIN_PIXELS_AT_OUTPUT = 10`, `MIN_SHIRT_PIXELS = 20`

**Key functions**:

- `find_hair_top(image)` → `int | None` — scans top rows for first non-transparent pixel
- `detect_face_geometry(image_rgb)` → `FaceGeometry | None` — MediaPipe → dlib cascade
- `compute_crop(geo, img_w, img_h, hair_top_y, chin_pixels)` → `tuple | None` — crop box or None if insufficient shirt
- `crop_with_padding(image, left, top, right, bottom)` — crops with black fill where box extends outside image

---

### Step 5 — `deglow.py` (`ic-deglow`)

**Purpose**: Remove the light halo that semi-transparent fringe pixels retain
from the original (usually light) background.

**Algorithm**:
1. Identify fringe pixels: alpha ∈ `[FRINGE_MIN, FRINGE_MAX]` = `[5, 219]`
2. For each fringe pixel, find the nearest fully-opaque anchor pixel (alpha ≥ `OPAQUE_MIN` = 220) using a Euclidean distance transform
3. Compute correction strength from anchor luminance (BT.601):
   `strength = STRENGTH_MIN + (STRENGTH_MAX - STRENGTH_MIN) × (1 - anchor_luminance)^1.5`
   — dark subjects (black hair) get maximum correction; blond/pale subjects get a gentler touch
4. Blend fringe pixel toward anchor colour with the computed strength
5. Subtract any remaining excess brightness beyond the blend

**Key functions**:

- `luminance(rgb)` → ndarray — BT.601 perceptual luminance
- `build_anchor_map(rgb, alpha, radius)` → `(anchor_rgb, dist_to_opaque)` — distance transform pass
- `deglow_image(img, strength_scale, radius)` → `Image` — apply deglow to one RGBA image

**CLI flags**: `--strength 0–2` (scale factor), `--radius N` (max search distance in px)

---

### Step 6 — `finalize_cutout.py` (`ic-center`)

**Purpose**: Center the non-transparent content on a fresh 250×250 transparent
canvas, ensuring pixel-perfect centering regardless of any shifts from earlier
steps.

**Algorithm**:
1. `img.getbbox()` — tight bounding box of non-transparent pixels
2. Scale down if content exceeds `CANVAS_SIZE` (250×250)
3. Paste centered on a new 250×250 RGBA canvas

Returns `False` (no output) for fully transparent inputs.

**Key function**: `center_on_canvas(src, dst)` → `bool`

---

### Step 7 (optional) — `submit_cutout.py` (`ic-submit`)

**Purpose**: Post a finished cutout back to sortitoutsi.net as the response to
a source submission.

**Process**:
1. Load `.sitsi.json` sidecar to get `submission_id`
2. GET the create-response form to extract hidden fields (CSRF token, etc.)
3. POST multipart/form-data with the image + all hidden fields
4. Parse the response for success/failure indicators

**Authentication**: Requires `SITSI_COOKIE` env var.

**Key functions**:

- `load_metadata(image_path)` → `SubmissionMeta | None` — reads sidecar, also checks stem-prefix matches
- `submit_cutout(session, image_path, submission_id)` → `SubmitResult` — dynamic form POST

---

### Utility — `benchmark.py`

**Purpose**: Compute deterministic, reproducible quality metrics on PNG outputs
(no ML models required) and detect regressions against a baseline.

**Six metrics** (all stored in `MetricsDict`):

| Metric | Meaning | Good value |
|--------|---------|-----------|
| `fg_coverage_pct` | % pixels with alpha > 127 | Stable near baseline |
| `fringe_density_pct` | % semi-transparent fringe pixels (5 ≤ α ≤ 219) | Low |
| `mean_fringe_brightness` | Mean luminance of fringe pixels | Low (dark fringes) |
| `alpha_edge_sharpness` | RMS gradient of alpha at edges | High (crisp edges) |
| `h_center_of_mass` | Horizontal CoM as fraction of width | ~0.5 (centred) |
| `v_center_of_mass` | Vertical CoM as fraction of height | ~0.4–0.5 |

**CLI**: `python -m image_cropper.pipeline.benchmark DIR [--compare baseline.json] [--update-baseline baseline.json]`

---

## GUI

### `gui.py`

PySide6 desktop application.  All heavy work runs in background QThreads
(see below); Qt signals carry results back to the main thread.

**Class hierarchy**:

```
QMainWindow
└── MainWindow           — orchestrator; owns all state

QThread
├── PipelineWorker       — runs pipeline steps via subprocesses
├── SubmitWorker         — uploads finished cutout to sortitoutsi
└── DownloadWorker       — scrapes + downloads source images

QListWidgetItem
└── ImageListItem        — per-image row with step-status icons

QLabel
└── ImagePreview         — aspect-ratio-preserving image viewer

QWidget
└── StepRow              — one pipeline step row (checkbox + Run button)

dataclass (plain Python)
└── ImageEntry           — per-image state (paths, statuses, debug overlay)
```

**Thread model**:

```
Main thread (Qt event loop)
│
├─ PipelineWorker.run()  ← subprocess per step per image
│    signals: log, progress, image_step_done, finished_all
│
├─ SubmitWorker.run()    ← HTTP POST via requests
│    signals: log, finished
│
└─ DownloadWorker.run()  ← HTTP GET via requests
     signals: log, progress, finished
```

Signals are Qt queued connections so they are always delivered on the main
thread even though they are emitted from worker threads.

**Session directory**: `MainWindow` creates a `tempfile.mkdtemp(prefix="imgcrop_")`
directory at startup.  `PipelineWorker` writes all intermediate step outputs
here.  The directory is deleted in `closeEvent`.

**Output directory**: Configurable via the toolbar text field (default
`~/image-cropper-output`).  At the end of every pipeline run, `center` (or
`deglow` if `center` was not run) outputs are copied here.

**Step ID → module mapping** (`STEP_MODULE`):

| Step ID | Module |
|---------|--------|
| `align` | `image_cropper.pipeline.align` |
| `crop_face` | `image_cropper.pipeline.crop_source` |
| `remove_bg` | `image_cropper.pipeline.remove_background` |
| `crop_portrait` | `image_cropper.pipeline.crop_cutout` |
| `deglow` | `image_cropper.pipeline.deglow` |
| `center` | `image_cropper.pipeline.finalize_cutout` |

**Submit button prerequisites** (all must be true):
1. `center` step status is `done`
2. `center` or `deglow` output file exists on disk
3. `.sitsi.json` sidecar found for the original image
4. Cookie field is non-empty
