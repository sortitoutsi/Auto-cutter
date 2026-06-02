# image-cropper

An automated pipeline + GUI that turns raw player source images into clean
250×250 transparent-background portrait cutouts.

```
align eyes → crop head+shoulders → remove background → portrait crop → deglow → center on canvas
```

Comes with a cross-platform GUI (PySide6) where you can load images,
preview each pipeline step, see the eye-detection debug overlay, and tune
the portrait crop on the fly.

> **AI Notice** — See [AI-NOTICE.md](AI-NOTICE.md) for disclosure about how
> this project was developed.

---

## Install (macOS, Linux, Windows)

Requires Python **3.11, 3.12, 3.13, or 3.14** on PATH. The installers create
a local virtualenv at `.venv/` and install everything into it.

### macOS / Linux

```bash
git clone https://github.com/Krissmed/image-cropper.git
cd image-cropper
./install.sh
```

Launch:

```bash
./launch_gui.sh
# or
.venv/bin/image-cropper
```

### Windows (PowerShell)

```powershell
git clone https://github.com/Krissmed/image-cropper.git
cd image-cropper
.\install.ps1
```

Launch:

```powershell
.\launch_gui.bat
# or
.venv\Scripts\image-cropper.exe
```

### Direct pip (any platform)

```bash
pip install .
image-cropper            # GUI
python -m image_cropper  # same thing
```

> First launch downloads ~95 MB of additional model weights (dlib face
> landmarks) to a per-user cache directory (`~/Library/Caches/image-cropper`
> on macOS, `%LOCALAPPDATA%\image-cropper` on Windows,
> `$XDG_CACHE_HOME/image-cropper` on Linux). Background removal also
> downloads BiRefNet weights from Hugging Face on first run (~2 GB).

---

## What the GUI does

- **Add Files / Add Folder** — load any mix of images
- **Per-image preview** — toggle between original, debug overlay, or latest output
- **Per-step Run buttons** — process one stage at a time and inspect the result
- **Show debug overlay** — saves and displays the eye-detection overlay from
  the Align step
- **Chin pixels** — live-tune how many output pixels (out of 250) sit below
  the chin in the portrait crop
- **Run all checked steps** — batch the whole pipeline across every loaded image
- **Final outputs** — copied to the output directory shown in the toolbar
  (default: `~/image-cropper-output`)

---

## CLI usage

After installation each pipeline step is also exposed as a console script:

```bash
ic-align         input/  output/aligned/      [--debug]
ic-crop-face     output/aligned/  output/cropped/
ic-remove-bg     --input output/cropped/  --output output/transparent/
ic-crop-portrait output/transparent/  output/portrait/  [--chin-pixels 10]
ic-deglow        output/portrait/  output/deglowed/  --overwrite
ic-center        output/deglowed/  output/final/
ic-download      # (downloads from sortitoutsi — needs SITSI_COOKIE env var)
```

Or invoke as Python modules: `python -m image_cropper.pipeline.align …`

The end-to-end wrapper still works:

```bash
SITSI_COOKIE="laravel_session=…" ./pipeline.sh
./pipeline.sh --skip-download                   # use existing input/
./pipeline.sh --skip-download --input my_imgs/  # custom directory
```

---

## Optional: standalone binary build

For users who want a single double-clickable bundle (~3 GB because of
PyTorch + MediaPipe + dlib):

```bash
# macOS / Linux
./build_binary.sh
# → dist/image-cropper/        (folder bundle)
# → dist/image-cropper.app/    (macOS only)

# Windows (PowerShell)
.\build_binary.ps1
# → dist\image-cropper\
```

PyInstaller cannot cross-compile, so run the build on the target OS.

For most users, `./install.sh` (or `.\install.ps1`) is faster and lighter.

---

## Project layout

```
image-cropper/
├── pyproject.toml              # package metadata + dependencies
├── install.sh / install.ps1    # cross-platform installers
├── launch_gui.sh / .bat        # GUI launchers
├── build_binary.sh / .ps1      # optional PyInstaller bundle
├── image_cropper.spec          # PyInstaller config
├── pipeline.sh                 # end-to-end CLI orchestrator
├── src/image_cropper/
│   ├── gui.py                  # PySide6 GUI
│   ├── models.py               # model file resolution (bundled / cache)
│   ├── data/
│   │   └── face_landmarker.task (3.6 MB, bundled)
│   └── pipeline/
│       ├── align.py            # step 1 – eye-level rotation
│       ├── crop_source.py      # step 2 – face + shoulder crop
│       ├── remove_background.py# step 3 – AI background removal (BiRefNet)
│       ├── crop_cutout.py      # step 4 – 250×250 portrait crop
│       ├── deglow.py           # step 5 – remove glow/halo on alpha edges
│       ├── finalize_cutout.py  # step 6 – center on 250×250 canvas
│       └── download_queue.py   # sortitoutsi downloader
├── docs/
│   ├── USAGE.md
│   └── ARCHITECTURE.md
└── AI-NOTICE.md
```

Output (gitignored):

```
output/                  # intermediate and final pipeline output
~/image-cropper-output/  # default GUI output location
```

---

## Platform notes

| Platform        | Status      | Notes                                                                    |
|-----------------|-------------|--------------------------------------------------------------------------|
| macOS (Apple)   | Primary     | Uses MPS for background removal. Tested on macOS 15+.                    |
| macOS (Intel)   | Should work | CPU-only background removal (slow).                                      |
| Linux x86_64    | Should work | CUDA on Nvidia, otherwise CPU. May need `libgl1` (`apt install libgl1`). |
| Linux ARM64     | Should work | dlib-bin wheels available; CPU only.                                     |
| Windows x86_64  | Should work | CUDA on Nvidia, otherwise CPU.                                           |

> "Should work" = the build targets have wheels, but I personally only
> dev on macOS. File an issue if something breaks on your platform.

---

## License

Personal / community use. Not affiliated with Sports Interactive or
sortitoutsi.net.
