#!/usr/bin/env python3
"""
Image Cropper GUI — PySide6 frontend for the processing pipeline.

Launch:
    ./venv_bg/bin/python gui.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    Qt, QThread, Signal, QSize, QTimer, QMutex, QMutexLocker,
)
from PySide6.QtGui import (
    QPixmap, QColor, QPainter, QBrush, QFont, QIcon, QAction,
    QPalette, QDragEnterEvent, QDropEvent,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QTextEdit,
    QSplitter, QFileDialog, QSpinBox, QGroupBox, QCheckBox,
    QProgressBar, QScrollArea, QTabBar, QStatusBar, QToolBar,
    QSizePolicy, QFrame, QLineEdit, QToolButton, QAbstractItemView,
    QScrollBar, QComboBox, QSpacerItem,
)

# ── paths ──────────────────────────────────────────────────────────────────────
PYTHON = sys.executable   # GUI must be launched from the same venv as the scripts

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Default output dir: ~/image-cropper-output (works on any OS,
# no assumption about a writable repo root).
DEFAULT_OUTPUT_DIR = Path.home() / "image-cropper-output"

# ── pipeline step definitions ──────────────────────────────────────────────────
STEPS = ["align", "crop_face", "remove_bg", "crop_portrait", "deglow"]
STEP_LABEL = {
    "align":        "1. Align Eyes",
    "crop_face":    "2. Crop Face",
    "remove_bg":    "3. Remove Background",
    "crop_portrait":"4. Crop Portrait",
    "deglow":       "5. Deglow",
}
# Module names — invoked via `python -m image_cropper.pipeline.<module>`
STEP_MODULE = {
    "align":        "image_cropper.pipeline.align",
    "crop_face":    "image_cropper.pipeline.crop_source",
    "remove_bg":    "image_cropper.pipeline.remove_background",
    "crop_portrait":"image_cropper.pipeline.crop_cutout",
    "deglow":       "image_cropper.pipeline.deglow",
}

STATUS_PENDING  = "pending"
STATUS_RUNNING  = "running"
STATUS_DONE     = "done"
STATUS_SKIPPED  = "skipped"
STATUS_ERROR    = "error"

STATUS_ICON = {
    STATUS_PENDING:  "○",
    STATUS_RUNNING:  "⟳",
    STATUS_DONE:     "✓",
    STATUS_SKIPPED:  "⊘",
    STATUS_ERROR:    "✗",
}
STATUS_COLOR = {
    STATUS_PENDING:  "#888",
    STATUS_RUNNING:  "#4a9fd4",
    STATUS_DONE:     "#4caf50",
    STATUS_SKIPPED:  "#ff9800",
    STATUS_ERROR:    "#f44336",
}


# ── image entry ────────────────────────────────────────────────────────────────
class ImageEntry:
    def __init__(self, path: Path):
        self.original = path
        self.name = path.name
        # step output paths (set after each step runs)
        self.outputs: dict[str, Path] = {}
        # step statuses
        self.statuses: dict[str, str] = {s: STATUS_PENDING for s in STEPS}
        # align debug image path
        self.debug_align: Optional[Path] = None

    def latest_output(self) -> Optional[Path]:
        """Most recent step output, or original."""
        for step in reversed(STEPS):
            if step in self.outputs and self.outputs[step].exists():
                return self.outputs[step]
        return self.original


# ── worker thread ──────────────────────────────────────────────────────────────
class PipelineWorker(QThread):
    log = Signal(str)
    progress = Signal(int, int)          # current, total
    image_step_done = Signal(str, str, str)  # image_name, step, status
    finished_all = Signal()

    def __init__(
        self,
        entries: list[ImageEntry],
        steps_to_run: list[str],
        session_dir: Path,
        chin_pixels: int,
        show_debug: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.entries = entries
        self.steps_to_run = steps_to_run
        self.session_dir = session_dir
        self.chin_pixels = chin_pixels
        self.show_debug = show_debug
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total = len(self.entries) * len(self.steps_to_run)
        done = 0

        for step in self.steps_to_run:
            if self._cancelled:
                break
            self.log.emit(f"\n── {STEP_LABEL[step]} ──────────────────")
            step_out = self.session_dir / step
            step_out.mkdir(parents=True, exist_ok=True)

            for entry in self.entries:
                if self._cancelled:
                    break

                # Determine input for this step
                if step == "align":
                    in_path = entry.original
                else:
                    prev_step = STEPS[STEPS.index(step) - 1]
                    if prev_step in entry.outputs and entry.outputs[prev_step].exists():
                        in_path = entry.outputs[prev_step]
                    else:
                        self.log.emit(f"  [{entry.name}] previous step output missing — skipping")
                        entry.statuses[step] = STATUS_SKIPPED
                        self.image_step_done.emit(entry.name, step, STATUS_SKIPPED)
                        done += 1
                        self.progress.emit(done, total)
                        continue

                entry.statuses[step] = STATUS_RUNNING
                self.image_step_done.emit(entry.name, step, STATUS_RUNNING)
                self.log.emit(f"  [{entry.name}]")

                success, out_path, debug_path = self._run_step(
                    step, in_path, step_out, entry
                )

                if success and out_path and out_path.exists():
                    entry.outputs[step] = out_path
                    entry.statuses[step] = STATUS_DONE
                    if debug_path:
                        entry.debug_align = debug_path
                    self.image_step_done.emit(entry.name, step, STATUS_DONE)
                elif success is None:
                    entry.statuses[step] = STATUS_SKIPPED
                    self.image_step_done.emit(entry.name, step, STATUS_SKIPPED)
                else:
                    entry.statuses[step] = STATUS_ERROR
                    self.image_step_done.emit(entry.name, step, STATUS_ERROR)

                done += 1
                self.progress.emit(done, total)

        self.finished_all.emit()

    def _run_step(
        self,
        step: str,
        in_path: Path,
        out_dir: Path,
        entry: ImageEntry,
    ) -> tuple[bool | None, Optional[Path], Optional[Path]]:
        """Run one step for one image. Returns (success, out_path, debug_path)."""
        module = STEP_MODULE[step]
        debug_path = None

        # Build expected output filename
        if step in ("align", "crop_face"):
            out_name = in_path.name
        else:
            out_name = in_path.stem + ".png"

        out_path = out_dir / out_name

        # All scripts process whole directories — copy this one image into a temp dir
        tmp_in = self._make_single_file_dir(in_path, step)

        base_cmd = [PYTHON, "-m", module]

        if step == "align":
            cmd = base_cmd + [str(tmp_in), str(out_dir)]
            if self.show_debug:
                cmd.append("--debug")
        elif step == "crop_face":
            cmd = base_cmd + [str(tmp_in), str(out_dir)]
        elif step == "remove_bg":
            cmd = base_cmd + ["--input", str(tmp_in), "--output", str(out_dir)]
        elif step == "crop_portrait":
            cmd = base_cmd + [str(tmp_in), str(out_dir), "--chin-pixels", str(self.chin_pixels)]
        elif step == "deglow":
            cmd = base_cmd + [str(tmp_in), str(out_dir), "--overwrite"]
        else:
            return False, None, None

        env = os.environ.copy()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            for line in (stdout + "\n" + stderr).strip().splitlines():
                if line.strip():
                    self.log.emit(f"    {line}")

            if tmp_in and tmp_in.exists():
                shutil.rmtree(tmp_in, ignore_errors=True)

            if result.returncode != 0:
                return False, None, None

            # For align, check debug
            if step == "align" and self.show_debug:
                debug_candidate = out_dir / "debug" / (in_path.stem + "_debug.jpg")
                if debug_candidate.exists():
                    debug_path = debug_candidate

            # Check if output was produced
            if out_path.exists():
                return True, out_path, debug_path

            # Sometimes output name differs (e.g. stem changes) — scan dir
            candidates = list(out_dir.glob(in_path.stem + "*"))
            candidates = [p for p in candidates if p.is_file() and "debug" not in p.parts]
            if candidates:
                return True, candidates[0], debug_path

            # Script may have skipped the image legitimately
            self.log.emit(f"    (no output — image may have been skipped)")
            return None, None, debug_path

        except Exception as e:
            self.log.emit(f"    ERROR: {e}")
            if tmp_in and tmp_in.exists():
                shutil.rmtree(tmp_in, ignore_errors=True)
            return False, None, None

    def _make_single_file_dir(self, in_path: Path, step: str) -> Path:
        """
        Scripts process entire directories. Copy this one image into a temp dir
        so we can run them per-file.
        """
        tmp_dir = self.session_dir / f"_tmp_{step}_{in_path.stem}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dst = tmp_dir / in_path.name
        if not dst.exists():
            shutil.copy2(in_path, dst)
        return tmp_dir


# ── image list item ────────────────────────────────────────────────────────────
class ImageListItem(QListWidgetItem):
    def __init__(self, entry: ImageEntry):
        super().__init__()
        self.entry = entry
        self._update()

    def _update(self):
        # Build status string
        parts = []
        for step in STEPS:
            s = self.entry.statuses.get(step, STATUS_PENDING)
            parts.append(STATUS_ICON[s])
        status_str = " ".join(parts)
        self.setText(f"{self.entry.name}\n{status_str}")
        self.setToolTip(self.entry.name)


# ── preview widget ─────────────────────────────────────────────────────────────
class ImagePreview(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #1a1a1a; border-radius: 6px;")
        self._pixmap: Optional[QPixmap] = None
        self.setText("No image selected")
        self.setStyleSheet(
            "background: #1a1a1a; color: #555; border-radius: 6px; font-size: 14px;"
        )

    def set_image(self, path: Optional[Path]):
        if path is None or not path.exists():
            self._pixmap = None
            self.setText("No image")
            return
        px = QPixmap(str(path))
        if px.isNull():
            self.setText(f"Cannot load\n{path.name}")
            return
        self._pixmap = px
        self._fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
            self.setText("")


# ── step row widget ────────────────────────────────────────────────────────────
class StepRow(QWidget):
    run_requested = Signal(str)   # step id

    def __init__(self, step_id: str, parent=None):
        super().__init__(parent)
        self.step_id = step_id
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self.check = QCheckBox(STEP_LABEL[step_id])
        self.check.setChecked(True)
        layout.addWidget(self.check)
        layout.addStretch()

        btn = QPushButton("Run")
        btn.setFixedWidth(50)
        btn.setFixedHeight(24)
        btn.clicked.connect(lambda: self.run_requested.emit(step_id))
        layout.addWidget(btn)

    def is_checked(self) -> bool:
        return self.check.isChecked()


# ── main window ────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Cropper")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 800)

        self._entries: list[ImageEntry] = []
        self._session_dir = Path(tempfile.mkdtemp(prefix="imgcrop_"))
        self._worker: Optional[PipelineWorker] = None
        self._current_entry: Optional[ImageEntry] = None
        self._preview_mode = "original"  # original | debug | output

        self._setup_ui()
        self._apply_dark_theme()

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Toolbar row
        root.addLayout(self._build_toolbar())

        # Main splitter: image list | preview | pipeline
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self._build_image_list())
        splitter.addWidget(self._build_preview_area())
        splitter.addWidget(self._build_pipeline_panel())

        splitter.setSizes([220, 620, 240])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        root.addWidget(splitter, 1)

        # Bottom: progress + log
        root.addLayout(self._build_bottom_bar())

    def _build_toolbar(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        add_files_btn = QPushButton("+ Add Files")
        add_files_btn.clicked.connect(self._add_files)
        row.addWidget(add_files_btn)

        add_folder_btn = QPushButton("📁 Add Folder")
        add_folder_btn.clicked.connect(self._add_folder)
        row.addWidget(add_folder_btn)

        clear_btn = QPushButton("✕ Clear All")
        clear_btn.clicked.connect(self._clear_all)
        row.addWidget(clear_btn)

        row.addStretch()

        row.addWidget(QLabel("Output:"))
        self._output_edit = QLineEdit(str(DEFAULT_OUTPUT_DIR))
        self._output_edit.setMinimumWidth(260)
        row.addWidget(self._output_edit)

        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(30)
        browse_btn.clicked.connect(self._browse_output)
        row.addWidget(browse_btn)

        return row

    def _build_image_list(self) -> QWidget:
        box = QGroupBox("Images")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 8, 4, 4)

        self._image_list = QListWidget()
        self._image_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._image_list.currentItemChanged.connect(self._on_image_selected)
        self._image_list.setAcceptDrops(True)
        self._image_list.setIconSize(QSize(48, 48))
        layout.addWidget(self._image_list)

        self._image_count_label = QLabel("0 images")
        self._image_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_count_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._image_count_label)

        return box

    def _build_preview_area(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Tab-like buttons for preview mode
        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self._prev_btns: dict[str, QPushButton] = {}
        for mode, label in [("original", "Original"), ("debug", "Debug (Align)"), ("output", "Latest Output")]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(mode == "original")
            btn.clicked.connect(lambda checked, m=mode: self._set_preview_mode(m))
            btn_row.addWidget(btn)
            self._prev_btns[mode] = btn

        btn_row.addStretch()
        self._preview_name_label = QLabel("")
        self._preview_name_label.setStyleSheet("color: #888; font-size: 11px;")
        btn_row.addWidget(self._preview_name_label)

        layout.addLayout(btn_row)

        self._preview = ImagePreview()
        layout.addWidget(self._preview, 1)

        return container

    def _build_pipeline_panel(self) -> QWidget:
        box = QGroupBox("Pipeline")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(4)

        self._step_rows: dict[str, StepRow] = {}
        for step in STEPS:
            row = StepRow(step)
            row.run_requested.connect(self._run_single_step)
            layout.addWidget(row)
            self._step_rows[step] = row

            if step == "align":
                indent = QHBoxLayout()
                indent.setContentsMargins(20, 0, 0, 0)
                self._debug_check = QCheckBox("Show debug overlay")
                self._debug_check.setChecked(True)
                indent.addWidget(self._debug_check)
                layout.addLayout(indent)

            if step == "crop_portrait":
                indent = QHBoxLayout()
                indent.setContentsMargins(20, 0, 0, 4)
                indent.addWidget(QLabel("Chin pixels:"))
                self._chin_spin = QSpinBox()
                self._chin_spin.setRange(0, 100)
                self._chin_spin.setValue(10)
                self._chin_spin.setFixedWidth(60)
                self._chin_spin.setToolTip("Output pixels (at 250×250) included below the chin")
                indent.addWidget(self._chin_spin)
                indent.addStretch()
                layout.addLayout(indent)

        layout.addSpacing(8)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #444;")
        layout.addWidget(sep)

        layout.addSpacing(4)

        run_selected_btn = QPushButton("Run Checked Steps\n(selected image)")
        run_selected_btn.clicked.connect(self._run_checked_steps_selected)
        layout.addWidget(run_selected_btn)

        run_all_btn = QPushButton("Run Checked Steps\n(all images)")
        run_all_btn.clicked.connect(self._run_checked_steps_all)
        layout.addWidget(run_all_btn)

        layout.addSpacing(4)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_worker)
        layout.addWidget(self._cancel_btn)

        layout.addStretch()

        # Status per-step for selected image
        layout.addWidget(QLabel("Selected image step status:"))
        self._step_status_labels: dict[str, QLabel] = {}
        for step in STEPS:
            lbl = QLabel(f"  {STEP_LABEL[step]}: —")
            lbl.setStyleSheet("font-size: 11px; color: #888;")
            layout.addWidget(lbl)
            self._step_status_labels[step] = lbl

        return box

    def _build_bottom_bar(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        layout.setSpacing(3)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(14)
        layout.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(160)
        self._log.setFont(QFont("Menlo", 11))
        self._log.setStyleSheet("background: #111; color: #ccc; border-radius: 4px;")
        layout.addWidget(self._log)

        return layout

    # ── dark theme ─────────────────────────────────────────────────────────────

    def _apply_dark_theme(self):
        palette = QPalette()
        bg = QColor("#1e1e1e")
        surface = QColor("#252525")
        border = QColor("#3a3a3a")
        text = QColor("#e0e0e0")
        accent = QColor("#4a9fd4")

        palette.setColor(QPalette.ColorRole.Window, bg)
        palette.setColor(QPalette.ColorRole.WindowText, text)
        palette.setColor(QPalette.ColorRole.Base, surface)
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#2a2a2a"))
        palette.setColor(QPalette.ColorRole.Text, text)
        palette.setColor(QPalette.ColorRole.Button, QColor("#303030"))
        palette.setColor(QPalette.ColorRole.ButtonText, text)
        palette.setColor(QPalette.ColorRole.Highlight, accent)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#fff"))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#666"))
        QApplication.instance().setPalette(palette)

        self.setStyleSheet("""
            QMainWindow { background: #1e1e1e; }
            QGroupBox {
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                margin-top: 6px;
                font-weight: bold;
                color: #bbb;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QPushButton {
                background: #303030;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px 10px;
                color: #e0e0e0;
            }
            QPushButton:hover { background: #3a3a3a; border-color: #555; }
            QPushButton:pressed { background: #252525; }
            QPushButton:checked { background: #1a4060; border-color: #4a9fd4; color: #7ec8f0; }
            QPushButton:disabled { color: #555; }
            QListWidget {
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 4px;
                color: #ccc;
                font-size: 11px;
            }
            QListWidget::item { padding: 4px; border-bottom: 1px solid #2a2a2a; }
            QListWidget::item:selected { background: #1a4060; color: #eee; }
            QListWidget::item:hover { background: #2a2a2a; }
            QTextEdit {
                background: #111;
                border: 1px solid #333;
                border-radius: 4px;
                color: #ccc;
            }
            QLineEdit {
                background: #252525;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 3px 6px;
                color: #e0e0e0;
            }
            QSpinBox {
                background: #252525;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 2px 4px;
                color: #e0e0e0;
            }
            QCheckBox { color: #ccc; }
            QLabel { color: #ccc; }
            QProgressBar {
                background: #252525;
                border: 1px solid #333;
                border-radius: 3px;
                text-align: center;
                color: #ccc;
            }
            QProgressBar::chunk { background: #4a9fd4; border-radius: 3px; }
            QSplitter::handle { background: #333; }
        """)

    # ── file management ────────────────────────────────────────────────────────

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", str(Path.home()),
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.tif)",
        )
        for f in files:
            self._add_entry(Path(f))
        self._refresh_list()

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", str(Path.home())
        )
        if not folder:
            return
        d = Path(folder)
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in SUPPORTED_EXTS:
                self._add_entry(p)
        self._refresh_list()

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self._output_edit.text()
        )
        if folder:
            self._output_edit.setText(folder)

    def _add_entry(self, path: Path):
        # Avoid duplicates
        if any(e.original == path for e in self._entries):
            return
        self._entries.append(ImageEntry(path))

    def _clear_all(self):
        self._entries.clear()
        self._image_list.clear()
        self._current_entry = None
        self._preview.set_image(None)
        self._image_count_label.setText("0 images")
        self._preview_name_label.setText("")

    def _refresh_list(self):
        self._image_list.clear()
        for entry in self._entries:
            item = ImageListItem(entry)
            # Thumbnail
            px = QPixmap(str(entry.original))
            if not px.isNull():
                item.setIcon(QIcon(px.scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)))
            self._image_list.addItem(item)
        self._image_count_label.setText(f"{len(self._entries)} image(s)")
        if self._entries and not self._current_entry:
            self._image_list.setCurrentRow(0)

    def _refresh_item(self, entry: ImageEntry):
        for i in range(self._image_list.count()):
            item = self._image_list.item(i)
            if isinstance(item, ImageListItem) and item.entry is entry:
                item._update()
                break

    # ── preview ────────────────────────────────────────────────────────────────

    def _on_image_selected(self, current: QListWidgetItem, _prev):
        if not isinstance(current, ImageListItem):
            self._current_entry = None
            return
        self._current_entry = current.entry
        self._update_preview()
        self._update_step_statuses()

    def _set_preview_mode(self, mode: str):
        self._preview_mode = mode
        for m, btn in self._prev_btns.items():
            btn.setChecked(m == mode)
        self._update_preview()

    def _update_preview(self):
        entry = self._current_entry
        if entry is None:
            self._preview.set_image(None)
            self._preview_name_label.setText("")
            return

        if self._preview_mode == "original":
            path = entry.original
        elif self._preview_mode == "debug":
            path = entry.debug_align or entry.original
        else:  # output
            path = entry.latest_output()

        self._preview.set_image(path)
        self._preview_name_label.setText(path.name if path else "")

    def _update_step_statuses(self):
        entry = self._current_entry
        if entry is None:
            for lbl in self._step_status_labels.values():
                lbl.setText("  —")
                lbl.setStyleSheet("font-size: 11px; color: #888;")
            return
        for step, lbl in self._step_status_labels.items():
            s = entry.statuses.get(step, STATUS_PENDING)
            icon = STATUS_ICON[s]
            color = STATUS_COLOR[s]
            lbl.setText(f"  {STEP_LABEL[step]}: {icon}")
            lbl.setStyleSheet(f"font-size: 11px; color: {color};")

    # ── pipeline running ───────────────────────────────────────────────────────

    def _checked_steps(self) -> list[str]:
        return [s for s in STEPS if self._step_rows[s].is_checked()]

    def _run_single_step(self, step: str):
        """Run one specific step on all loaded images."""
        if not self._entries:
            self._log_msg("No images loaded.")
            return
        self._start_worker(self._entries, [step])

    def _run_checked_steps_selected(self):
        if not self._current_entry:
            self._log_msg("No image selected.")
            return
        steps = self._checked_steps()
        if not steps:
            self._log_msg("No steps selected.")
            return
        self._start_worker([self._current_entry], steps)

    def _run_checked_steps_all(self):
        if not self._entries:
            self._log_msg("No images loaded.")
            return
        steps = self._checked_steps()
        if not steps:
            self._log_msg("No steps selected.")
            return
        self._start_worker(self._entries, steps)

    def _start_worker(self, entries: list[ImageEntry], steps: list[str]):
        if self._worker and self._worker.isRunning():
            self._log_msg("Already running — cancel first.")
            return

        chin = self._chin_spin.value()
        debug = self._debug_check.isChecked()

        self._worker = PipelineWorker(
            entries=entries,
            steps_to_run=steps,
            session_dir=self._session_dir,
            chin_pixels=chin,
            show_debug=debug,
        )
        self._worker.log.connect(self._log_msg)
        self._worker.progress.connect(self._on_progress)
        self._worker.image_step_done.connect(self._on_step_done)
        self._worker.finished_all.connect(self._on_worker_finished)

        total = len(entries) * len(steps)
        self._progress.setVisible(True)
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._cancel_btn.setEnabled(True)

        self._worker.start()

    def _cancel_worker(self):
        if self._worker:
            self._worker.cancel()
            self._log_msg("Cancelling…")

    def _on_progress(self, done: int, total: int):
        self._progress.setValue(done)

    def _on_step_done(self, image_name: str, step: str, status: str):
        entry = next((e for e in self._entries if e.name == image_name), None)
        if entry:
            self._refresh_item(entry)
            if entry is self._current_entry:
                self._update_step_statuses()
                if status == STATUS_DONE:
                    # Auto-switch preview to latest output when step finishes
                    if self._preview_mode == "output":
                        self._update_preview()
                    elif step == "align" and self._preview_mode == "debug" and entry.debug_align:
                        self._update_preview()

    def _on_worker_finished(self):
        self._progress.setVisible(False)
        self._cancel_btn.setEnabled(False)
        self._log_msg("── Done ──────────────────────────────────")
        self._update_preview()

        # Copy final deglow outputs to the configured output dir
        out_dir = Path(self._output_edit.text())
        copied = 0
        for entry in self._entries:
            src = entry.outputs.get("deglow")
            if src and src.exists():
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out_dir / src.name)
                copied += 1
        if copied:
            self._log_msg(f"Copied {copied} final image(s) to {out_dir}")

    # ── logging ────────────────────────────────────────────────────────────────

    def _log_msg(self, msg: str):
        self._log.append(msg)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── cleanup ────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        try:
            shutil.rmtree(self._session_dir, ignore_errors=True)
        except Exception:
            pass
        super().closeEvent(event)


# ── entry point ────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Image Cropper")
    app.setOrganizationName("imagecropper")

    win = MainWindow()
    win.show()

    # If paths passed as CLI args, load them
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in SUPPORTED_EXTS:
                    win._add_entry(f)
            win._refresh_list()
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            win._add_entry(p)
            win._refresh_list()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
