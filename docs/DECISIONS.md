# Architecture Decisions

Why the codebase is shaped the way it is.  Each entry explains the trade-off
that was made so future changes can be judged against the same criteria.

---

## Immutable frozen dataclasses for geometry objects

**Decision**: `EyeDetection`, `FaceBBox`, `FaceGeometry`, and `CropBox` are
`@dataclass(frozen=True, slots=True)`.

**Why**: Geometry values flow through several pipeline functions.  A mutable
object lets any callee silently mutate fields that callers still hold
references to — a class of bug that is hard to trace.  Frozen dataclasses make
every field read-only and give the type checker field-name visibility, so typos
are caught at analysis time rather than at runtime.

`slots=True` eliminates the per-instance `__dict__`, reducing memory for
objects that are created thousands of times in batch processing.

---

## TypedDict for JSON-serialisable shapes

**Decision**: `MetricsDict`, `QueueEntry`, `SubmissionMeta`, and `SubmitResult`
are `TypedDict` rather than dataclasses.

**Why**: These shapes need to round-trip through `json.dump()` / `json.load()`
(benchmark baselines, sidecar files, HTTP responses).  `json.dump()` accepts
plain dicts natively.  Converting a dataclass with `dataclasses.asdict()`
works but adds a call site at every serialisation point, and `asdict()`
recursively converts nested objects in ways that can surprise.  TypedDict gives
type-checked field access on plain dicts with zero serialisation overhead.

---

## Subprocess spawning in the GUI

**Decision**: `PipelineWorker` runs each pipeline step by spawning a subprocess
(`python -m image_cropper.pipeline.<module>`) rather than calling the module
functions directly.

**Why**:

1. **GPU/ML memory isolation** — PyTorch and the transformers library allocate
   GPU memory (VRAM) when models are loaded.  That memory is not reliably
   released inside a long-running process even after `del model`.  Spawning a
   subprocess means the OS reclaims VRAM as soon as the subprocess exits,
   preventing an out-of-memory crash on a machine running multiple images.

2. **Crash containment** — If `remove_background.py` crashes (e.g. a CUDA
   assertion), the GUI process stays alive and the user can retry.

3. **Simplicity** — The pipeline scripts already have correct `main()` entry
   points and handle their own argument parsing, error messages, and exit
   codes.  Reusing them as subprocesses means the GUI does not duplicate any
   of that logic.

**Trade-off**: Per-image subprocess startup adds ~0.5 s overhead.  For the
`remove_background` step this is negligible (15–40 s per image).  For fast
steps (`align`, `center`) it is noticeable but acceptable.

---

## BiRefNet dual-model ensemble + TTA

**Decision**: Background removal uses two BiRefNet models and optional
horizontal-flip test-time augmentation (TTA), mean-ensembled into one alpha.

**Why**: The two models are complementary:

- `BiRefNet_HR-matting` was trained for alpha matting — it produces soft,
  semi-transparent edges that preserve individual hair strands.
- `BiRefNet_HR` was trained for salient object detection — it produces a
  sharp, confident body boundary.

Using only the matting model can leave the body boundary too soft.  Using only
the detection model can clip hair too aggressively.  Their mean ensemble gives
sharp body edges and soft hair simultaneously.

TTA (running each model a second time on the horizontally mirrored image, then
flipping the alpha back) reduces directional bias in the model's predictions
and smooths asymmetric artifacts around the face.

**Trade-off**: Four model forward passes instead of one.  `--no-tta` halves
the passes; `--no-ensemble` reduces to one model.  Use these flags when
throughput matters more than quality.

---

## Guided filter on CPU, not MPS

**Decision**: `guided_refine()` forces `device = "cpu"` even when the rest of
the pipeline runs on Apple Silicon MPS.

**Why**: The multi-channel guided filter (He et al. 2010) uses large sliding-
window convolutions via `torch.nn.functional.unfold`.  On macOS, calling
`unfold` on an MPS tensor with certain input sizes triggers an IOGPU shared-
memory assertion crash:

```
Assertion failed: (size <= count), function iokit_mmap_region, ...
```

This is a macOS/MPS bug, not a code bug.  The guided filter runs fast enough
on CPU (< 1 s for a 2048-px image) that falling back does not meaningfully
affect total pipeline time.

---

## Detector cascades (MediaPipe → dlib → OpenCV)

**Decision**: Every detection stage tries multiple detectors in priority order
and uses the first that succeeds.

**Why**: No single detector is universally reliable.

- **MediaPipe** is the most accurate but requires a model file to be present
  (downloaded on first run; ~3 MB for face landmarker).
- **dlib** handles unusual head poses and partial occlusions well but needs
  the large shape predictor (~95 MB).
- **OpenCV Haar** cascades are always available (bundled with OpenCV) and
  never fail to load, making them a guaranteed last resort.

Falling back gracefully rather than hard-failing means the pipeline processes
as many images as possible.  Images where all detectors fail are skipped with a
warning so the batch can continue.

---

## Model file lookup order (bundled → cache → download)

**Decision**: Model files are resolved through three locations in order
(see `models.py`):
1. `src/image_cropper/data/` — bundled with the package
2. OS-specific user cache directory — persists across sessions
3. Auto-download to user cache on first use

**Why**:

- **Bundled** covers the smallest critical models (MediaPipe face landmarker,
  ~3 MB) so the package works fully offline after `pip install`.
- **User cache** avoids re-downloading the large dlib model (~95 MB) on every
  run.  The cache survives reinstalls and virtualenv rebuilds.
- **Auto-download** means users do not need to manually manage model files.

The repository stays lightweight (no large binaries in git) because only the
tiny bundled models ship with the source.

---

## Sidecar `.sitsi.json` metadata files

**Decision**: Downloaded images get a `<stem>.sitsi.json` sidecar file written
next to them.  The submit step reads this sidecar instead of asking the user
for the submission ID.

**Why**:

- **Decouples download from submit** — the two operations can run in separate
  sessions, on different machines, or with a manual review step between them.
- **Survives file moves** — as long as the sidecar travels with the image (both
  share the same stem), the metadata is available at submit time regardless of
  how the images were organised.
- **Plain JSON** — readable and editable in any text editor; no proprietary
  format or database dependency.

---

## Benchmark baseline + regression tolerances

**Decision**: Quality is tracked via `benchmarks/baseline.json` (six
deterministic metrics per image) rather than visual review only.

**Why**: ML-based pipeline steps produce subtly different outputs when models
are updated or algorithm parameters change.  A numerical baseline with per-
metric tolerances catches unintended regressions automatically in CI, without
requiring GPU hardware or model files in the test environment.

All six metrics are computed from the PNG's pixel values (no model inference),
so the benchmark job runs cheaply on any CPU.  Tolerances are tuned to pass
minor algorithm tweaks while failing significant quality regressions (e.g. a
2 percentage-point drop in foreground coverage).
