#!/usr/bin/env bash
# Install image-cropper on macOS or Linux.
# Creates a venv, installs the package and all dependencies.
#
# Usage:
#   ./install.sh                 # default: .venv/
#   ./install.sh /opt/imgcrop    # custom venv location
#   ./install.sh --user          # install into user site-packages (no venv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
USER_INSTALL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --user) USER_INSTALL=1; shift ;;
        -h|--help)
            sed -n '2,9p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) VENV_DIR="$1"; shift ;;
    esac
done

# --- find a compatible Python (3.11-3.13; PyTorch has no 3.14 wheels yet) ---
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v "$candidate")"
        break
    fi
done
# Fall back to python3 and check its version
if [[ -z "$PYTHON_BIN" ]] && command -v python3 >/dev/null 2>&1; then
    pyver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    case "$pyver" in
        3.11|3.12|3.13) PYTHON_BIN="$(command -v python3)" ;;
    esac
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: Need Python 3.11, 3.12, or 3.13 on PATH (PyTorch has no 3.14 wheels yet)." >&2
    echo "Install one via your package manager (brew, apt, dnf, etc.) and try again." >&2
    exit 1
fi
echo "Using: $PYTHON_BIN ($($PYTHON_BIN --version))"

# --- install ---
if [[ $USER_INSTALL -eq 1 ]]; then
    echo "Installing image-cropper into user site-packages..."
    "$PYTHON_BIN" -m pip install --user --upgrade pip
    "$PYTHON_BIN" -m pip install --user "$SCRIPT_DIR"
    echo ""
    echo "Done. Launch with:"
    echo "    python3 -m image_cropper"
else
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "Creating virtual environment at $VENV_DIR ..."
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        echo "Reusing virtual environment at $VENV_DIR"
    fi
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    echo "Installing image-cropper and dependencies (this can take several minutes — pytorch is large)..."
    "$VENV_DIR/bin/pip" install "$SCRIPT_DIR"
    echo ""
    echo "Done. Launch the GUI with:"
    echo "    $VENV_DIR/bin/image-cropper"
    echo "or:"
    echo "    $VENV_DIR/bin/python -m image_cropper"
fi
