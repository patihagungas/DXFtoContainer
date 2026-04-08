"""DXF Assembly Container Packer — application entry point."""

import sys
import logging

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from dxf_packer.utils import setup_logging
from dxf_packer.ui.main_window import MainWindow


def main() -> None:
    """Launch the DXF Packer desktop application."""
    setup_logging(logging.INFO)
    log = logging.getLogger(__name__)
    log.info("Starting DXF Assembly Container Packer")

    # High-DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("DXF Packer")
    app.setOrganizationName("DXFPacker")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
