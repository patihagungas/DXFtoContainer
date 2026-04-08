"""Main application window — three-pane layout with status bar."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
    QStatusBar, QLabel, QFileDialog, QMessageBox,
    QProgressBar,
)

from .left_panel import LeftPanel
from .right_panel import RightPanel
from .viewport import Viewport3D
from ..assembly import load_assembly
from ..packer import Packer, PackingResult
from ..container import Container
from ..exporter import export_dxf, export_json, export_csv, export_pdf
from ..utils import color_palette

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background packing worker
# ---------------------------------------------------------------------------

class PackWorker(QObject):
    """Run the packer in a background QThread to keep the UI responsive."""

    finished = pyqtSignal(object)   # PackingResult
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(
        self,
        assemblies: list,
        container: Container,
        clearance: float,
        log_cross: float,
        optimize_mode: str,
        fine_rotations: bool,
    ) -> None:
        super().__init__()
        self.assemblies = assemblies
        self.container = container
        self.clearance = clearance
        self.log_cross = log_cross
        self.optimize_mode = optimize_mode
        self.fine_rotations = fine_rotations

    def run(self) -> None:
        try:
            packer = Packer(
                self.container,
                clearance_mm=self.clearance,
                log_cross_mm=self.log_cross,
                optimize_mode=self.optimize_mode,
            )
            result = packer.pack(
                self.assemblies,
                fine_rotations=self.fine_rotations,
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
            )
            self.finished.emit(result)
        except Exception as exc:
            log.exception("Packing failed")
            self.error.emit(f"Packing failed:\n{exc}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """DXF Packer main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DXF Assembly Container Packer")
        self.resize(1400, 800)

        self._assemblies: list = []
        self._result: Optional[PackingResult] = None
        self._container: Optional[Container] = None
        self._worker: Optional[PackWorker] = None
        self._thread: Optional[QThread] = None
        self._color_map: dict = {}      # assembly name → RGBA
        self._color_idx: int = 0

        self._build_ui()
        self._build_menu()
        self._apply_dark_theme()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # Left panel
        self.left_panel = LeftPanel()
        self.left_panel.btn_add.clicked.connect(self._on_add_dxf)
        self.left_panel.assemblies_changed.connect(self._update_status)
        splitter.addWidget(self.left_panel)

        # Centre viewport
        self.viewport = Viewport3D()
        self.viewport.item_selected.connect(self._on_item_selected)
        splitter.addWidget(self.viewport)

        # Right panel
        self.right_panel = RightPanel()
        self.right_panel.pack_requested.connect(self._on_pack)
        self.right_panel.btn_export_dxf.clicked.connect(self._export_dxf)
        self.right_panel.btn_export_json.clicked.connect(self._export_json)
        self.right_panel.btn_export_csv.clicked.connect(self._export_csv)
        self.right_panel.btn_export_pdf.clicked.connect(self._export_pdf)
        splitter.addWidget(self.right_panel)

        splitter.setSizes([280, 900, 260])
        root_layout.addWidget(splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.lbl_status = QLabel("Ready — load DXF files to begin.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setTextVisible(True)

        self.status_bar.addWidget(self.lbl_status, 1)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _build_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        act_open = QAction("&Add DXF…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._on_add_dxf)
        file_menu.addAction(act_open)

        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        pack_menu = menu.addMenu("&Pack")
        act_pack = QAction("&Pack Now", self)
        act_pack.setShortcut("Ctrl+Return")
        act_pack.triggered.connect(self._on_pack)
        pack_menu.addAction(act_pack)

        help_menu = menu.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2a;
                color: #dde;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
            }
            QMenuBar { background: #2a2a38; color: #dde; }
            QMenuBar::item:selected { background: #3a3a55; }
            QMenu { background: #2a2a38; color: #dde; border: 1px solid #444; }
            QMenu::item:selected { background: #3a78af; }
            QSplitter::handle { background: #333; }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 4px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QScrollBar:vertical {
                background: #2a2a38; width: 12px; border: none;
            }
            QScrollBar::handle:vertical { background: #555; border-radius: 5px; }
            QLabel { color: #ccd; }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background: #2a2a3a; color: #dde;
                border: 1px solid #555; border-radius: 3px; padding: 2px 4px;
            }
            QCheckBox { color: #ccd; }
            QStatusBar { background: #141420; color: #aab; border-top: 1px solid #333; }
            QProgressBar {
                background: #2a2a3a; border: 1px solid #555; border-radius: 3px;
                text-align: center; color: white;
            }
            QProgressBar::chunk { background: #2c6fad; border-radius: 2px; }
        """)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_add_dxf(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open DXF File(s)", "", "DXF Files (*.dxf *.DXF);;All Files (*)"
        )
        for path in paths:
            self._load_dxf_file(path)

    def _load_dxf_file(self, path: str) -> None:
        try:
            color = self._next_color()
            asm = load_assembly(path, color=color)
            self._assemblies.append(asm)
            self.left_panel.add_assembly(asm, color)
            self._set_status(f"Loaded: {asm.name} ({asm.dimensions[0]:.0f}×{asm.dimensions[1]:.0f}×{asm.dimensions[2]:.0f} mm)")
        except Exception as exc:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load:\n{path}\n\n{exc}"
            )
            log.exception("Load failed: %s", path)

    def _on_pack(self) -> None:
        assemblies = self.left_panel.get_active_assemblies()
        if not assemblies:
            QMessageBox.warning(self, "No Assemblies", "Load at least one DXF assembly first.")
            return

        container = self.right_panel.get_container()
        if container.volume <= 0:
            QMessageBox.warning(self, "Invalid Container", "Container dimensions must be > 0.")
            return

        # Disable pack button during run
        self.right_panel.btn_pack.setEnabled(False)
        self.right_panel.enable_export_buttons(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(assemblies))
        self.progress_bar.setValue(0)
        self._set_status("Packing…")

        self._worker = PackWorker(
            assemblies=assemblies,
            container=container,
            clearance=self.right_panel.get_clearance(),
            log_cross=self.right_panel.get_log_cross(),
            optimize_mode=self.right_panel.get_optimize_mode(),
            fine_rotations=self.right_panel.get_fine_rotations(),
        )
        self._container = container
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_pack_done)
        self._worker.error.connect(self._on_pack_error)
        self._worker.progress.connect(lambda cur, tot: self.progress_bar.setValue(cur))
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_pack_done(self, result: PackingResult) -> None:
        self._result = result
        self.progress_bar.setVisible(False)
        self.right_panel.btn_pack.setEnabled(True)
        self.right_panel.enable_export_buttons(True)

        # Render
        self.viewport.render_result(result, self._container)

        # Status
        total_qty = sum(a.quantity for a in self.left_panel.get_active_assemblies())
        weight_str = (
            f"  |  {result.total_weight_kg:.1f} kg" if result.total_weight_kg else ""
        )
        collision_str = (
            f"  |  ⚠ {len(result.collisions)} collision(s)" if result.collisions else ""
        )
        self._set_status(
            f"Packed {len(result.placed)}/{total_qty} items  "
            f"|  Utilization: {result.utilization*100:.1f}%  "
            f"|  {result.elapsed_sec:.2f}s"
            f"{weight_str}{collision_str}"
        )

        if result.collisions:
            QMessageBox.warning(
                self, "Collision Detected",
                "Post-packing collision check found overlapping items:\n\n" +
                "\n".join(result.collisions[:5])
            )

        if result.unpacked:
            names = ", ".join(a.name for a in result.unpacked[:5])
            if len(result.unpacked) > 5:
                names += f" (+{len(result.unpacked)-5} more)"
            QMessageBox.information(
                self, "Some Items Not Placed",
                f"The following items did not fit in the container:\n{names}"
            )

    def _on_pack_error(self, msg: str) -> None:
        self.progress_bar.setVisible(False)
        self.right_panel.btn_pack.setEnabled(True)
        QMessageBox.critical(self, "Packing Error", msg)
        self._set_status("Packing failed — see error dialog.")

    def _on_item_selected(self, item) -> None:
        if item is None:
            return
        p = item.position
        e = item.extents
        self._set_status(
            f"Selected: {item.assembly.name}  |  "
            f"Pos: ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm  |  "
            f"Dims: {e[0]:.0f}×{e[1]:.0f}×{e[2]:.0f} mm"
        )

    # ------------------------------------------------------------------
    # Export slots
    # ------------------------------------------------------------------

    def _export_dxf(self) -> None:
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export DXF", "packing_result.dxf", "DXF Files (*.dxf)"
        )
        if path:
            try:
                export_dxf(self._result, self._container, path)
                self._set_status(f"DXF exported: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", str(exc))

    def _export_json(self) -> None:
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON", "packing_result.json", "JSON Files (*.json)"
        )
        if path:
            try:
                export_json(self._result, self._container, path)
                self._set_status(f"JSON exported: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", str(exc))

    def _export_csv(self) -> None:
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "packing_result.csv", "CSV Files (*.csv)"
        )
        if path:
            try:
                export_csv(self._result, path)
                self._set_status(f"CSV exported: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", str(exc))

    def _export_pdf(self) -> None:
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF", "packing_report.pdf", "PDF Files (*.pdf)"
        )
        if path:
            try:
                screenshot = None
                img = self.viewport.grab_framebuffer()
                if img and not img.isNull():
                    sc_path = str(Path(path).with_suffix(".viewport.png"))
                    img.save(sc_path)
                    screenshot = sc_path
                export_pdf(self._result, self._container, path, screenshot)
                self._set_status(f"PDF exported: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)

    def _update_status(self) -> None:
        n = len(self.left_panel.get_active_assemblies())
        self._set_status(f"{n} assembl{'y' if n==1 else 'ies'} loaded  —  configure container and click PACK")

    def _next_color(self) -> tuple:
        palette = color_palette(20)
        color = palette[self._color_idx % len(palette)]
        self._color_idx += 1
        return color

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About DXF Packer",
            "<b>DXF Assembly Container Packer</b><br><br>"
            "Loads 3D DXF assemblies, normalises their orientation,<br>"
            "and packs them into shipping containers with dunnage.<br><br>"
            "Tech: PyQt6 · ezdxf · trimesh · pyqtgraph<br>"
            "Version 1.0.0",
        )
