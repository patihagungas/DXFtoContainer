"""Left panel: assembly list with per-row controls."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QFrame, QSpinBox, QDoubleSpinBox, QCheckBox,
    QSizePolicy, QFileDialog, QMessageBox,
)

log = logging.getLogger(__name__)


class AssemblyRow(QFrame):
    """One row in the assembly list representing a single Assembly."""

    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)  # emits self

    def __init__(self, assembly, color: tuple, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.assembly = assembly
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"AssemblyRow {{ background: rgba({int(color[0]*255)},"
            f"{int(color[1]*255)},{int(color[2]*255)}, 40); "
            f"border: 1px solid rgba({int(color[0]*255)},"
            f"{int(color[1]*255)},{int(color[2]*255)}, 180); "
            f"border-radius: 4px; margin: 2px; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(3)

        # Row 1: checkbox + name + remove button
        row1 = QHBoxLayout()
        self.chk_enabled = QCheckBox()
        self.chk_enabled.setChecked(True)
        self.chk_enabled.setToolTip("Include in packing")
        row1.addWidget(self.chk_enabled)

        name_lbl = QLabel(f"<b>{assembly.name}</b>")
        name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        row1.addWidget(name_lbl)

        btn_remove = QPushButton("✕")
        btn_remove.setFixedSize(22, 22)
        btn_remove.setStyleSheet("QPushButton { color: #c44; border: none; font-weight: bold; }")
        btn_remove.clicked.connect(lambda: self.remove_requested.emit(self))
        row1.addWidget(btn_remove)
        layout.addLayout(row1)

        # Row 2: dimensions label
        d = assembly.dimensions
        dims_lbl = QLabel(f"{d[0]:.0f} × {d[1]:.0f} × {d[2]:.0f} mm")
        dims_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(dims_lbl)

        # Row 3: Qty + Weight + Allow tipping
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Qty:"))
        self.spin_qty = QSpinBox()
        self.spin_qty.setRange(1, 999)
        self.spin_qty.setValue(assembly.quantity)
        self.spin_qty.setFixedWidth(55)
        self.spin_qty.valueChanged.connect(self._on_changed)
        row3.addWidget(self.spin_qty)

        row3.addWidget(QLabel("kg:"))
        self.spin_weight = QDoubleSpinBox()
        self.spin_weight.setRange(0.0, 999999.0)
        self.spin_weight.setDecimals(1)
        self.spin_weight.setSpecialValueText("—")
        self.spin_weight.setValue(assembly.weight_kg if assembly.weight_kg else 0.0)
        self.spin_weight.setFixedWidth(65)
        self.spin_weight.valueChanged.connect(self._on_changed)
        row3.addWidget(self.spin_weight)

        self.chk_tip = QCheckBox("Tip OK")
        self.chk_tip.setChecked(assembly.allow_tipping)
        self.chk_tip.setToolTip("Allow non-upright orientations during packing")
        self.chk_tip.stateChanged.connect(self._on_changed)
        row3.addWidget(self.chk_tip)

        row3.addStretch()
        layout.addLayout(row3)

    def _on_changed(self) -> None:
        self.assembly.quantity = self.spin_qty.value()
        w = self.spin_weight.value()
        self.assembly.weight_kg = w if w > 0 else None
        self.assembly.allow_tipping = self.chk_tip.isChecked()
        self.changed.emit()

    def is_enabled(self) -> bool:
        return self.chk_enabled.isChecked()


class LeftPanel(QWidget):
    """Left panel containing the assembly list and load button.

    Signals:
        assemblies_changed: Emitted whenever the list or row settings change.
    """

    assemblies_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._rows: list[AssemblyRow] = []
        self._load_callback: Optional[Callable] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("<b>Assemblies</b>"))
        hdr.addStretch()
        self.btn_add = QPushButton("Add DXF…")
        self.btn_add.setStyleSheet(
            "QPushButton { background: #2c6fad; color: white; "
            "padding: 4px 10px; border-radius: 3px; } "
            "QPushButton:hover { background: #3a8fd4; }"
        )
        hdr.addWidget(self.btn_add)
        layout.addLayout(hdr)

        # Scroll area for rows
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(0)
        self.list_layout.addStretch()

        self.scroll_area.setWidget(self.list_widget)
        layout.addWidget(self.scroll_area)

        # Clear all button
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self.clear_all)
        layout.addWidget(self.btn_clear)

        self.setMinimumWidth(240)
        self.setMaximumWidth(320)

    def add_assembly(self, assembly, color: tuple) -> None:
        """Add an assembly row to the list.

        Args:
            assembly: Assembly object.
            color: RGBA float tuple for the row accent colour.
        """
        row = AssemblyRow(assembly, color)
        row.changed.connect(self.assemblies_changed.emit)
        row.remove_requested.connect(self._remove_row)
        self._rows.append(row)
        # Insert before the stretch
        self.list_layout.insertWidget(self.list_layout.count() - 1, row)
        self.assemblies_changed.emit()

    def get_active_assemblies(self) -> list:
        """Return assemblies for enabled rows only."""
        return [row.assembly for row in self._rows if row.is_enabled()]

    def clear_all(self) -> None:
        """Remove all assembly rows."""
        for row in self._rows:
            self.list_layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        self.assemblies_changed.emit()

    def _remove_row(self, row: AssemblyRow) -> None:
        self.list_layout.removeWidget(row)
        row.deleteLater()
        self._rows.remove(row)
        self.assemblies_changed.emit()
