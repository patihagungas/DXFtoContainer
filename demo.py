"""Demo launcher: opens the app pre-loaded with all 3 sample DXFs and auto-packs them."""

import sys
import logging
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer

from dxf_packer.utils import setup_logging, color_palette
from dxf_packer.ui.main_window import MainWindow
from dxf_packer.assembly import load_assembly

SAMPLES = [
    ("samples/box_simple.dxf",       3),
    ("samples/l_shape_tilted.dxf",   3),
    ("samples/thin_beam_45deg.dxf",  3),
]


def main() -> None:
    setup_logging(logging.INFO)

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("DXF Packer Demo")

    window = MainWindow()
    window.show()

    # Auto-load samples
    palette = color_palette(len(SAMPLES))
    base = Path(__file__).parent
    for (path, qty), color in zip(SAMPLES, palette):
        full_path = base / path
        if full_path.exists():
            try:
                asm = load_assembly(str(full_path), quantity=qty, color=color)
                window.left_panel.add_assembly(asm, color)
                window._assemblies.append(asm)
            except Exception as exc:
                print(f"Could not load {path}: {exc}")

    # Auto-pack after a short delay (UI must be rendered first)
    QTimer.singleShot(800, window._on_pack)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
