# image-cropper

An automated pipeline that turns raw football-player source images into clean
250×250 transparent-background portrait cutouts for
[sortitoutsi.net](https://sortitoutsi.net) / Football Manager face packs.

> **AI Notice** — See [AI-NOTICE.md](AI-NOTICE.md) for disclosure about how
> this project was developed.

---

## What it does

```
download → align eyes → crop head+shoulders → remove background → portrait crop → deglow
```

Each step is a standalone Python script; `pipeline.sh` wires them together
from end to end and handles venv creation, cleanup, and intermediate
directories automatically.

Final output: `output/final/*.png` — 250×250 RGBA PNGs, transparent
background, glow-free edges.

---

## Quick start

### Full pipeline (download + process)

```bash
SITSI_COOKIE="laravel_session=…; remember_web_…=…" ./pipeline.sh
```

### Process images you already have

```bash
./pipeline.sh --skip-download                    # reads from input/
./pipeline.sh --skip-download --input my_imgs/   # custom directory
```

### Run a single step

```bash
source .venv/bin/activate      # or let the pipeline create one for you

python align.py input/ output/aligned/
python crop_source.py output/aligned/ output/cropped/
python remove_background.py --input output/cropped/ --output output/transparent/
python crop_cutout.py output/transparent/ output/portrait/
python deglow.py output/portrait/ output/final/ --overwrite
```

See [docs/USAGE.md](docs/USAGE.md) for all flags and per-script options.

---

## Requirements

- macOS or Linux (tested on macOS 15+)
- Python 3.11 – 3.13 (PyTorch has no wheels for 3.14 yet)
- ~4 GB disk for ML model weights (downloaded automatically on first run)
- GPU strongly recommended for background removal (MPS on Apple Silicon,
  CUDA on Nvidia); CPU works but is slow

Install Python dependencies:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The pipeline creates and tears down its own throwaway venv automatically,
so you only need a manual venv when running scripts individually.

---

## Project layout

```
image-cropper/
├── pipeline.sh          # end-to-end orchestrator
├── download_queue.py    # step 1 – download from sortitoutsi queue
├── align.py             # step 2 – eye-level rotation
├── crop_source.py       # step 3 – face + shoulder crop
├── remove_background.py # step 4 – AI background removal (BiRefNet)
├── crop_cutout.py       # step 5 – 250×250 portrait crop
├── deglow.py            # step 6 – remove glow/halo on alpha edges
├── finalize-cutout.py   # utility – center cutout on blank canvas
├── requirements.txt
├── docs/
│   ├── USAGE.md         # detailed per-script usage
│   └── ARCHITECTURE.md  # design decisions and data flow
├── CONTRIBUTING.md
└── AI-NOTICE.md
```

Output (gitignored):

```
output/
├── aligned/      # after step 2
├── cropped/      # after step 3
├── transparent/  # after step 4
├── portrait/     # after step 5
└── final/        # after step 6 (delivery)
```

---

## Getting your SITSI cookie

1. Log in to [sortitoutsi.net](https://sortitoutsi.net) in your browser
2. Open DevTools → Application → Cookies → `sortitoutsi.net`
3. Copy all `name=value` pairs, joined by `; `
4. Export before running: `export SITSI_COOKIE="laravel_session=abc; …"`

---

## License

Personal / community use. Not affiliated with Sports Interactive or
sortitoutsi.net.
