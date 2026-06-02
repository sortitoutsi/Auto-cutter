# PyInstaller spec for building a standalone image-cropper binary.
#
# Build (after `pip install pyinstaller` in your venv):
#     pyinstaller image_cropper.spec
#
# Output:
#     dist/image-cropper/        (one-folder bundle)
#     dist/image-cropper.app/    (macOS .app, when on macOS)
#
# NOTES:
#   * Output is large (~3 GB) because it bundles PyTorch, MediaPipe, dlib,
#     PySide6, etc. There's no way to shrink this much without giving up
#     features.
#   * Build on each target OS — PyInstaller cannot cross-compile.
#   * The dlib 95 MB shape predictor still auto-downloads to the user
#     cache dir on first run; only the small (3.6 MB) MediaPipe model is
#     bundled into the binary.
#   * For most users `pip install .` is a better distribution path.

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Resolve package data: the bundled MediaPipe model.
datas = collect_data_files("image_cropper", subdir="data")

# MediaPipe ships data files (model graphs, etc.) that must be bundled.
datas += collect_data_files("mediapipe")

# Heavy ML libs sometimes need explicit hidden imports.
hiddenimports = (
    collect_submodules("mediapipe")
    + collect_submodules("kornia")
    + collect_submodules("transformers")
    + collect_submodules("timm")
)

a = Analysis(
    ["src/image_cropper/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Reduce size by excluding optional deps we don't use
        "tkinter",
        "test",
        "tests",
        "unittest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="image-cropper",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="image-cropper",
)

# macOS .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="image-cropper.app",
        icon=None,
        bundle_identifier="com.krissmed.imagecropper",
        info_plist={
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "LSMinimumSystemVersion": "11.0",
            "CFBundleShortVersionString": "0.1.0",
        },
    )
