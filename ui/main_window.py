import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.ai_advisor import AdvisorError, suggest_adjustments
from core.image_ops import apply_exposure_contrast, auto_white_balance, resize_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_rgb(path: Path) -> tuple[np.ndarray, bytes | None]:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    # exif_transpose already normalizes the Orientation tag to match the
    # rotated pixels, so these bytes are safe to reattach on export as-is.
    exif_bytes = img.info.get("exif")
    img = img.convert("RGB")
    return np.array(img), exif_bytes


def rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class AdvisorWorker(QThread):
    """Runs the OpenAI vision call off the UI thread (it's a network call,
    so blocking the UI for it would be worse than blocking it for local
    pixel math)."""

    succeeded = Signal(dict)
    failed = Signal(str)

    def __init__(self, image: np.ndarray, api_key: str, parent=None):
        super().__init__(parent)
        self._image = image
        self._api_key = api_key

    def run(self):
        try:
            result = suggest_adjustments(self._image, api_key=self._api_key)
        except AdvisorError as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sol-Room beta 0.0.1026")
        self.resize(1200, 800)

        self.current_path: Path | None = None
        self.original_rgb: np.ndarray | None = None
        self.working_rgb: np.ndarray | None = None
        self.original_exif: bytes | None = None
        self._updating_aspect = False

        self.ai_exposure = 0.0
        self.ai_contrast = 0.0
        self._ai_worker: AdvisorWorker | None = None
        self._session_api_key: str | None = None

        self._build_ui()

    def _build_preview_pane(self, title: str) -> tuple[QWidget, QLabel]:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #ccc; font-weight: bold;")

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(250, 400)
        image_label.setStyleSheet("background-color: #222; color: #aaa;")

        layout.addWidget(title_label)
        layout.addWidget(image_label, stretch=1)
        return container, image_label

    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)

        self.file_list = QListWidget()
        self.file_list.setFixedWidth(220)
        self.file_list.currentItemChanged.connect(self._on_select_image)
        root.addWidget(self.file_list)

        compare_widget = QWidget()
        compare_layout = QHBoxLayout(compare_widget)
        compare_layout.setContentsMargins(0, 0, 0, 0)

        before_container, self.before_preview = self._build_preview_pane("Before")
        after_container, self.after_preview = self._build_preview_pane("After")
        self.before_preview.setText("Open a folder to begin")
        compare_layout.addWidget(before_container, stretch=1)
        compare_layout.addWidget(after_container, stretch=1)

        root.addWidget(compare_widget, stretch=1)

        controls = QWidget()
        controls.setFixedWidth(260)
        clayout = QVBoxLayout(controls)

        self.wb_checkbox = QCheckBox("Auto White Balance")
        self.wb_checkbox.toggled.connect(self._on_toggle_wb)
        clayout.addWidget(self.wb_checkbox)

        ai_box = QGroupBox("AI Suggestion (exposure / contrast)")
        ai_layout = QVBoxLayout(ai_box)
        self.ai_suggest_btn = QPushButton("Get AI Suggestion")
        self.ai_suggest_btn.clicked.connect(self._on_ai_suggest)
        self.ai_apply_checkbox = QCheckBox("Apply suggestion")
        self.ai_apply_checkbox.setEnabled(False)
        self.ai_apply_checkbox.toggled.connect(self._on_toggle_ai)
        self.ai_status = QLabel("")
        self.ai_status.setWordWrap(True)
        self.ai_status.setStyleSheet("color: #8ab4f8;")
        ai_layout.addWidget(self.ai_suggest_btn)
        ai_layout.addWidget(self.ai_apply_checkbox)
        ai_layout.addWidget(self.ai_status)
        clayout.addWidget(ai_box)

        resize_box = QGroupBox("Resize for Export")
        form = QFormLayout(resize_box)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 20000)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 20000)
        self.keep_aspect = QCheckBox("Maintain aspect ratio")
        self.keep_aspect.setChecked(True)
        form.addRow("Width", self.width_spin)
        form.addRow("Height", self.height_spin)
        form.addRow(self.keep_aspect)
        clayout.addWidget(resize_box)

        self.width_spin.valueChanged.connect(self._on_width_changed)
        self.height_spin.valueChanged.connect(self._on_height_changed)

        export_btn = QPushButton("Export...")
        export_btn.clicked.connect(self._on_export)
        clayout.addWidget(export_btn)
        clayout.addStretch()

        root.addWidget(controls)
        self.setCentralWidget(central)

        open_action = QAction("Open Folder...", self)
        open_action.triggered.connect(self._on_open_folder)
        menu = self.menuBar().addMenu("File")
        menu.addAction(open_action)

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        self.file_list.clear()
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                item = QListWidgetItem(p.name)
                item.setData(Qt.UserRole, str(p))
                self.file_list.addItem(item)
        if self.file_list.count():
            self.file_list.setCurrentRow(0)

    def _on_select_image(self, current, _previous):
        if current is None:
            return
        path = Path(current.data(Qt.UserRole))
        self.current_path = path
        self.original_rgb, self.original_exif = load_rgb(path)
        self.wb_checkbox.blockSignals(True)
        self.wb_checkbox.setChecked(False)
        self.wb_checkbox.blockSignals(False)

        self.ai_exposure = 0.0
        self.ai_contrast = 0.0
        self.ai_apply_checkbox.blockSignals(True)
        self.ai_apply_checkbox.setChecked(False)
        self.ai_apply_checkbox.setEnabled(False)
        self.ai_apply_checkbox.blockSignals(False)
        self.ai_status.setText("")

        h, w = self.original_rgb.shape[:2]
        self._updating_aspect = True
        self.width_spin.setValue(w)
        self.height_spin.setValue(h)
        self._updating_aspect = False
        self._recompute_working()

    def _on_toggle_wb(self, _checked: bool):
        self._recompute_working()

    def _recompute_working(self):
        if self.original_rgb is None:
            return
        img = self.original_rgb
        if self.wb_checkbox.isChecked():
            img = auto_white_balance(img)
        if self.ai_apply_checkbox.isChecked() and (self.ai_exposure or self.ai_contrast):
            img = apply_exposure_contrast(img, self.ai_exposure, self.ai_contrast)
        self.working_rgb = img
        self._refresh_preview()

    def _on_ai_suggest(self):
        if self.original_rgb is None:
            return
        api_key = os.environ.get("OPENAI_API_KEY") or self._session_api_key
        if not api_key:
            key, ok = QInputDialog.getText(
                self,
                "OpenAI API Key",
                "Enter your OpenAI API key (kept in memory for this session only, never saved to disk):",
                QLineEdit.Password,
            )
            if not ok or not key.strip():
                return
            self._session_api_key = key.strip()
            api_key = self._session_api_key

        self.ai_suggest_btn.setEnabled(False)
        self.ai_status.setStyleSheet("color: #8ab4f8;")
        self.ai_status.setText("Analyzing with AI...")

        self._ai_worker = AdvisorWorker(self.original_rgb, api_key)
        self._ai_worker.succeeded.connect(self._on_ai_result)
        self._ai_worker.failed.connect(self._on_ai_error)
        self._ai_worker.finished.connect(lambda: self.ai_suggest_btn.setEnabled(True))
        self._ai_worker.start()

    def _on_ai_result(self, result: dict):
        self.ai_exposure = result["exposure"]
        self.ai_contrast = result["contrast"]
        self.ai_status.setStyleSheet("color: #8ab4f8;")
        self.ai_status.setText(
            f"{result['reasoning']} (exposure {result['exposure']:+.2f}, contrast {result['contrast']:+.2f})"
        )
        self.ai_apply_checkbox.setEnabled(True)
        self.ai_apply_checkbox.blockSignals(True)
        self.ai_apply_checkbox.setChecked(True)
        self.ai_apply_checkbox.blockSignals(False)
        self._recompute_working()

    def _on_ai_error(self, message: str):
        self.ai_status.setStyleSheet("color: #f28b82;")
        self.ai_status.setText(f"AI suggestion failed: {message}")

    def _on_toggle_ai(self, _checked: bool):
        self._recompute_working()

    def _on_width_changed(self, value):
        if self._updating_aspect or self.original_rgb is None or not self.keep_aspect.isChecked():
            return
        h, w = self.original_rgb.shape[:2]
        self._updating_aspect = True
        self.height_spin.setValue(max(1, round(value * h / w)))
        self._updating_aspect = False

    def _on_height_changed(self, value):
        if self._updating_aspect or self.original_rgb is None or not self.keep_aspect.isChecked():
            return
        h, w = self.original_rgb.shape[:2]
        self._updating_aspect = True
        self.width_spin.setValue(max(1, round(value * w / h)))
        self._updating_aspect = False

    def _refresh_preview(self):
        if self.original_rgb is None or self.working_rgb is None:
            return
        self._set_pane_image(self.before_preview, self.original_rgb)
        self._set_pane_image(self.after_preview, self.working_rgb)

    @staticmethod
    def _set_pane_image(label: QLabel, rgb: np.ndarray):
        pixmap = rgb_to_pixmap(rgb)
        scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def _on_export(self):
        if self.working_rgb is None or self.current_path is None:
            return
        result = resize_image(self.working_rgb, self.width_spin.value(), self.height_spin.value())

        default_name = f"{self.current_path.stem}_edited{self.current_path.suffix}"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", default_name, "JPEG (*.jpg);;PNG (*.png)"
        )
        if not save_path:
            return
        save_kwargs = {"quality": 95}
        if self.original_exif:
            save_kwargs["exif"] = self.original_exif
        Image.fromarray(result).save(save_path, **save_kwargs)
        QMessageBox.information(self, "Exported", f"Saved to {save_path}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview()
