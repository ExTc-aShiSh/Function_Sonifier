"""
main.py — Entry point for Function Sonifier.

A desktop GUI application that lets users hear mathematical functions
as sound. Built with PyQt6, SymPy, Matplotlib, and sounddevice.
"""

import sys
import os

# Ensure the package directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from gui import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Function Sonifier")
    app.setStyle("Fusion")

    # Set default font
    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
