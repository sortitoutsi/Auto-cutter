#!/usr/bin/env bash
# Build a standalone image-cropper binary using PyInstaller.
# Run on the target OS (PyInstaller cannot cross-compile).
#
# Usage:
#   ./build_binary.sh             # uses .venv/
#   ./build_binary.sh /opt/venv   # custom venv

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="${1:-.venv}"
if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv $VENV not found. Run ./install.sh first." >&2
    exit 1
fi

PY="$VENV/bin/python"
"$PY" -m pip install --quiet "pyinstaller>=6.0"

echo "Building standalone binary (this is slow and the result is ~3 GB)..."
"$PY" -m PyInstaller --clean --noconfirm image_cropper.spec

if [[ -d "dist/image-cropper" ]]; then
    SIZE=$(du -sh dist/image-cropper | cut -f1)
    echo ""
    echo "Done. Bundle at: dist/image-cropper/ ($SIZE)"
    if [[ -d "dist/image-cropper.app" ]]; then
        echo "macOS app:      dist/image-cropper.app/"
    fi
fi
