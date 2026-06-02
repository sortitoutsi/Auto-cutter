#!/usr/bin/env bash
# Full image processing pipeline:
#   download → align eyes → crop faces → remove background → crop portrait → deglow → center on canvas
#
# Usage:
#   SITSI_COOKIE="..." ./pipeline.sh           # download + process
#   ./pipeline.sh --skip-download              # process existing input/ images
#   ./pipeline.sh --skip-download --input dir/ # use a custom input directory
#
# Final output lands in output-final/
# A temporary venv is created, used, and deleted automatically.
# All intermediate files are removed on exit (success or failure).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- arg parsing ---
SKIP_DOWNLOAD=0
INPUT_DIR="$SCRIPT_DIR/input"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        --input)         INPUT_DIR="$(realpath "$2")"; SKIP_DOWNLOAD=1; shift 2 ;;
        -h|--help)
            sed -n '2,11p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

TMP_DIR="$SCRIPT_DIR/tmp_pipeline"
VENV_DIR="$TMP_DIR/.venv"
TMP_ALIGNED="$TMP_DIR/aligned"
TMP_CROPPED="$TMP_DIR/cropped"
TMP_TRANSPARENT="$TMP_DIR/transparent"
TMP_PORTRAIT="$TMP_DIR/portrait"
TMP_DEGLOWED="$TMP_DIR/deglowed"
OUTPUT_DIR="$SCRIPT_DIR/output/final"

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    rm -rf "$TMP_DIR"
    echo "Removed: $TMP_DIR (includes venv)"
    if [[ $SKIP_DOWNLOAD -eq 0 && -d "$INPUT_DIR" ]]; then
        rm -rf "$INPUT_DIR"
        echo "Removed: $INPUT_DIR"
    fi
}
trap cleanup EXIT

mkdir -p "$TMP_ALIGNED" "$TMP_CROPPED" "$TMP_TRANSPARENT" "$TMP_PORTRAIT" "$TMP_DEGLOWED" "$OUTPUT_DIR"

# --- venv setup ---
echo "=== Setting up virtual environment ==="
# PyTorch has no wheels for Python 3.14 yet; prefer 3.13
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON_BIN="$(command -v "$candidate")"
        break
    fi
done
if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: No Python 3 found on PATH." >&2
    exit 1
fi
echo "Using: $PYTHON_BIN ($($PYTHON_BIN --version))"
"$PYTHON_BIN" -m venv "$VENV_DIR"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

echo "Installing image-cropper package..."
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet "$SCRIPT_DIR"
echo "Venv ready."

# --- step 1: download ---
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
    echo ""
    echo "=== Step 1/7: Downloading images ==="
    if [[ -z "${SITSI_COOKIE:-}" ]]; then
        echo "ERROR: SITSI_COOKIE is not set." >&2
        echo "Export it before running: SITSI_COOKIE='...' ./pipeline.sh" >&2
        exit 1
    fi
    "$PYTHON" -m image_cropper.pipeline.download_queue
    INPUT_DIR="$SCRIPT_DIR/input"
else
    echo ""
    echo "=== Step 1/7: Skipping download, using $INPUT_DIR ==="
fi

# --- step 2: align eyes ---
echo ""
echo "=== Step 2/7: Aligning eyes ==="
"$PYTHON" -m image_cropper.pipeline.align "$INPUT_DIR" "$TMP_ALIGNED"

# --- step 3: crop faces ---
echo ""
echo "=== Step 3/7: Cropping faces ==="
"$PYTHON" -m image_cropper.pipeline.crop_source "$TMP_ALIGNED" "$TMP_CROPPED"

# --- step 4: remove background ---
echo ""
echo "=== Step 4/7: Removing backgrounds ==="
"$PYTHON" -m image_cropper.pipeline.remove_background \
    --input "$TMP_CROPPED" \
    --output "$TMP_TRANSPARENT"

# --- step 5: crop portrait ---
echo ""
echo "=== Step 5/7: Cropping to portrait (250×250) ==="
"$PYTHON" -m image_cropper.pipeline.crop_cutout "$TMP_TRANSPARENT" "$TMP_PORTRAIT"

# --- step 6: deglow ---
echo ""
echo "=== Step 6/7: Removing glow/halo ==="
"$PYTHON" -m image_cropper.pipeline.deglow "$TMP_PORTRAIT" "$TMP_DEGLOWED" --overwrite

# --- step 7: center on canvas ---
echo ""
echo "=== Step 7/7: Centering on canvas ==="
"$PYTHON" -m image_cropper.pipeline.finalize_cutout "$TMP_DEGLOWED" "$OUTPUT_DIR"

echo ""
echo "=== Pipeline complete ==="
echo "Final images: $OUTPUT_DIR"
count=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.png" | wc -l | tr -d ' ')
echo "Output count: $count PNG(s)"
