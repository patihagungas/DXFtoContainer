"""Right panel: container settings, packing parameters, and Pack button."""

from __future__ import annotations

import logging
from typing import Optional, Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QSizePolicy, QFrame, QSpinBox,
)

from ..container import Container, PRESETS

log = logging.getLogger(__name__)

MM_PER_INCH = 25.4


class RightPanel(QWidget):
    """Right panel with container / packing settings and the Pack button.

    Signals:
        pack_requested: Emitted when the user clicks Pack.
    """

    pack_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._updating = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Container settings ──────────────────────────────────────
        grp_container = QGroupBox("Container")
        grp_container.setStyleSheet("QGroupBox { font-weight: bold; }")
        form_c = QFormLayout(grp_container)
        form_c.setSpacing(6)

        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(PRESETS.keys()))
        self.combo_preset.currentTextChanged.connect(self._on_preset_changed)
        form_c.addRow("Preset:", self.combo_preset)

        # Unit toggle
        unit_row = QHBoxLayout()
        self.combo_unit = QComboBox()
        self.combo_unit.addItems(["mm", "inches"])
        self.combo_unit.currentTextChanged.connect(self._on_unit_changed)
        unit_row.addWidget(self.combo_unit)
        unit_row.addStretch()
        form_c.addRow("Units:", unit_row)

        self.spin_length = self._make_dim_spin()
        self.spin_width = self._make_dim_spin()
        self.spin_height = self._make_dim_spin()
        form_c.addRow("Length:", self.spin_length)
        form_c.addRow("Width:", self.spin_width)
        form_c.addRow("Height:", self.spin_height)

        for spin in (self.spin_length, self.spin_width, self.spin_height):
            spin.valueChanged.connect(self._on_dim_changed)

        layout.addWidget(grp_container)

        # ── Packing settings ────────────────────────────────────────
        grp_pack = QGroupBox("Packing")
        grp_pack.setStyleSheet("QGroupBox { font-weight: bold; }")
        form_p = QFormLayout(grp_pack)
        form_p.setSpacing(6)

        self.spin_clearance = QDoubleSpinBox()
        self.spin_clearance.setRange(0.0, 500.0)
        self.spin_clearance.setDecimals(1)
        self.spin_clearance.setSuffix(" mm")
        self.spin_clearance.setValue(50.0)
        form_p.addRow("Clearance:", self.spin_clearance)

        self.spin_log_cross = QDoubleSpinBox()
        self.spin_log_cross.setRange(10.0, 500.0)
        self.spin_log_cross.setDecimals(1)
        self.spin_log_cross.setSuffix(" mm")
        self.spin_log_cross.setValue(50.0)
        form_p.addRow("Log section:", self.spin_log_cross)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Max items (BFFD)", "Min space"])
        form_p.addRow("Optimize:", self.combo_mode)

        self.chk_fine = QCheckBox("Try fine rotations (slower)")
        self.chk_fine.setToolTip(
            "Also try 15° Z-rotation steps for each item.\n"
            "Significantly slower — use only when needed."
        )
        form_p.addRow("", self.chk_fine)

        layout.addWidget(grp_pack)

        # ── Export buttons ──────────────────────────────────────────
        grp_export = QGroupBox("Export")
        grp_export.setStyleSheet("QGroupBox { font-weight: bold; }")
        vbox_exp = QVBoxLayout(grp_export)
        vbox_exp.setSpacing(4)

        self.btn_export_dxf = QPushButton("Export DXF…")
        self.btn_export_json = QPushButton("Export JSON…")
        self.btn_export_csv = QPushButton("Export CSV…")
        self.btn_export_pdf = QPushButton("Export PDF…")

        for btn in (self.btn_export_dxf, self.btn_export_json,
                    self.btn_export_csv, self.btn_export_pdf):
            btn.setEnabled(False)
            btn.setStyleSheet(
                "QPushButton { background: #3a3a4a; color: #bbb; padding: 4px 8px; border-radius: 3px; } "
                "QPushButton:enabled { background: #2c5f8a; color: white; } "
                "QPushButton:enabled:hover { background: #3a78af; }"
            )
            vbox_exp.addWidget(btn)

        layout.addWidget(grp_export)

        layout.addStretch()

        # ── Pack button ─────────────────────────────────────────────
        self.btn_pack = QPushButton("PACK")
        self.btn_pack.setMinimumHeight(48)
        self.btn_pack.setStyleSheet(
            "QPushButton { background: #1a7a1a; color: white; font-size: 16px; "
            "font-weight: bold; border-radius: 6px; } "
            "QPushButton:hover { background: #22a022; } "
            "QPushButton:pressed { background: #155515; } "
            "QPushButton:disabled { background: #333; color: #666; }"
        )
        self.btn_pack.clicked.connect(self.pack_requested.emit)
        layout.addWidget(self.btn_pack)

        self.setMinimumWidth(220)
        self.setMaximumWidth(280)

        # Init with first preset
        self._on_preset_changed(self.combo_preset.currentText())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def get_container(self) -> Container:
        """Return a Container from the current UI values (always in mm)."""
        if self.combo_unit.currentText() == "inches":
            l = self.spin_length.value() * MM_PER_INCH
            w = self.spin_width.value() * MM_PER_INCH
            h = self.spin_height.value() * MM_PER_INCH
        else:
            l = self.spin_length.value()
            w = self.spin_width.value()
            h = self.spin_height.value()
        name = self.combo_preset.currentText()
        return Container(length=l, width=w, height=h, name=name)

    def get_clearance(self) -> float:
        """Clearance in mm."""
        return self.spin_clearance.value()

    def get_log_cross(self) -> float:
        """Log cross-section in mm."""
        return self.spin_log_cross.value()

    def get_optimize_mode(self) -> str:
        return "max_items" if self.combo_mode.currentIndex() == 0 else "min_space"

    def get_fine_rotations(self) -> bool:
        return self.chk_fine.isChecked()

    def enable_export_buttons(self, enabled: bool) -> None:
        """Enable or disable the export buttons."""
        for btn in (self.btn_export_dxf, self.btn_export_json,
                    self.btn_export_csv, self.btn_export_pdf):
            btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_preset_changed(self, name: str) -> None:
        if name == "Custom" or name not in PRESETS:
            return
        self._updating = True
        l, w, h = PRESETS[name]
        if self.combo_unit.currentText() == "inches":
            self.spin_length.setValue(l / MM_PER_INCH)
            self.spin_width.setValue(w / MM_PER_INCH)
            self.spin_height.setValue(h / MM_PER_INCH)
        else:
            self.spin_length.setValue(l)
            self.spin_width.setValue(w)
            self.spin_height.setValue(h)
        self._updating = False

    def _on_unit_changed(self, unit: str) -> None:
        """Convert displayed values when switching units."""
        self._updating = True
        if unit == "inches":
            self.spin_length.setValue(self.spin_length.value() / MM_PER_INCH)
            self.spin_width.setValue(self.spin_width.value() / MM_PER_INCH)
            self.spin_height.setValue(self.spin_height.value() / MM_PER_INCH)
            for spin in (self.spin_length, self.spin_width, self.spin_height):
                spin.setSuffix(" in")
                spin.setMaximum(10000.0)
                spin.setDecimals(2)
        else:
            self.spin_length.setValue(self.spin_length.value() * MM_PER_INCH)
            self.spin_width.setValue(self.spin_width.value() * MM_PER_INCH)
            self.spin_height.setValue(self.spin_height.value() * MM_PER_INCH)
            for spin in (self.spin_length, self.spin_width, self.spin_height):
                spin.setSuffix(" mm")
                spin.setMaximum(99999.0)
                spin.setDecimals(1)
        self._updating = False

    def _on_dim_changed(self) -> None:
        if not self._updating:
            self.combo_preset.setCurrentText("Custom")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_dim_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 99999.0)
        spin.setDecimals(1)
        spin.setSuffix(" mm")
        spin.setSingleStep(100.0)
        return spin
