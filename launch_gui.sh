#!/usr/bin/env bash
# Launch the Image Cropper GUI.
# Prefers ./.venv, then ./venv_bg, then a `image-cropper` console script on PATH.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for cand in "$SCRIPT_DIR/.venv/bin/image-cropper" \
            "$SCRIPT_DIR/venv_bg/bin/image-cropper" \
            "$SCRIPT_DIR/.venv/bin/python" \
            "$SCRIPT_DIR/venv_bg/bin/python"; do
    if [[ -x "$cand" ]]; then
        case "$cand" in
            *image-cropper) exec "$cand" "$@" ;;
            *python)        exec "$cand" -m image_cropper "$@" ;;
        esac
    fi
done

if command -v image-cropper >/dev/null 2>&1; then
    exec image-cropper "$@"
fi

echo "ERROR: image-cropper not installed. Run ./install.sh first." >&2
exit 1
