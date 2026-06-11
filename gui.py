"""
gui.py — Main GUI window for Function Sonifier (PyQt6).
"""

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QComboBox, QGroupBox,
    QTextEdit, QFileDialog, QMessageBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QSplitter, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QIcon

from graph_widget import GraphWidget
from audio_engine import AudioEngine, PlaybackState
from function_parser import evaluate_function, analyze_function
from export_utils import (
    export_graph_as_png, export_audio_as_wav,
    save_project_settings, load_project_settings,
)
from settings import (
    ProjectSettings, PRESET_FUNCTIONS,
    DEFAULT_X_START, DEFAULT_X_END, DEFAULT_NUM_SAMPLES,
)


# ──────────────────────────────────────────────
# Thread-safe signal bridge
# ──────────────────────────────────────────────
class _SignalBridge(QObject):
    progress_signal = pyqtSignal(float)
    finished_signal = pyqtSignal()


class MainWindow(QMainWindow):
    """Main application window for Function Sonifier."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Function Sonifier")
        self.setMinimumSize(960, 700)
        self.resize(1100, 780)

        # State
        self._audio_engine = AudioEngine()
        self._settings = ProjectSettings()
        self._current_result = None
        self._comparison_result = None

        # Signal bridge for thread-safe UI updates
        self._signals = _SignalBridge()
        self._signals.progress_signal.connect(self._on_progress)
        self._signals.finished_signal.connect(self._on_playback_finished)
        self._audio_engine.set_progress_callback(
            lambda p: self._signals.progress_signal.emit(p)
        )
        self._audio_engine.set_finished_callback(
            lambda: self._signals.finished_signal.emit()
        )

        # Cursor update timer
        self._cursor_timer = QTimer()
        self._cursor_timer.setInterval(40)  # ~25 fps
        self._cursor_timer.timeout.connect(self._update_cursor)

        self._apply_theme()
        self._build_ui()

    # ──────────────────────────────────────
    # Theme
    # ──────────────────────────────────────
    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QWidget { font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; font-size: 13px; color: #1a1a1a; }
            QGroupBox { border: 1px solid #d0d0d0; border-radius: 6px; margin-top: 14px; padding: 12px 8px 8px 8px; background: #ffffff; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #333; font-weight: 600; font-size: 13px; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { border: 1px solid #c0c0c0; border-radius: 4px; padding: 5px 8px; background: #ffffff; selection-background-color: #555; }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #666; }
            QPushButton { border: 1px solid #b0b0b0; border-radius: 4px; padding: 6px 14px; background: #ffffff; color: #1a1a1a; font-weight: 500; }
            QPushButton:hover { background: #e8e8e8; border-color: #888; }
            QPushButton:pressed { background: #d0d0d0; }
            QPushButton#playBtn { background: #1a1a1a; color: #ffffff; border: none; }
            QPushButton#playBtn:hover { background: #333; }
            QPushButton#stopBtn { background: #ffffff; color: #c0392b; border-color: #c0392b; }
            QPushButton#stopBtn:hover { background: #fdecea; }
            QSlider::groove:horizontal { height: 4px; background: #d0d0d0; border-radius: 2px; }
            QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; background: #1a1a1a; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #555; border-radius: 2px; }
            QTextEdit { border: 1px solid #d0d0d0; border-radius: 4px; background: #fafafa; font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; }
            QLabel#sectionLabel { font-weight: 600; font-size: 13px; color: #333; }
        """)

    # ──────────────────────────────────────
    # UI Construction
    # ──────────────────────────────────────
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left panel (controls) ──
        left = QWidget()
        left.setFixedWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Title
        title = QLabel("Function Sonifier")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #1a1a1a; padding: 4px 0 2px 0;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title)

        # --- Function Input ---
        grp_func = QGroupBox("Function")
        gl = QVBoxLayout(grp_func)
        self._func_input = QLineEdit("sin(x)")
        self._func_input.setPlaceholderText("e.g. sin(x), x**2, exp(-x)*cos(5*x)")
        self._func_input.returnPressed.connect(self._on_plot)
        gl.addWidget(self._func_input)

        # Presets dropdown
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("— Select a preset —", "")
        for fn in PRESET_FUNCTIONS:
            self._preset_combo.addItem(fn, fn)
        self._preset_combo.setFixedHeight(32)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        preset_row.addWidget(self._preset_combo, stretch=1)
        gl.addLayout(preset_row)
        left_layout.addWidget(grp_func)

        # --- Range ---
        grp_range = QGroupBox("Range")
        rg = QGridLayout(grp_range)
        rg.setSpacing(6)
        rg.addWidget(QLabel("x start:"), 0, 0)
        self._x_start = QDoubleSpinBox()
        self._x_start.setRange(-1000, 1000)
        self._x_start.setDecimals(2)
        self._x_start.setValue(DEFAULT_X_START)
        self._x_start.setFixedHeight(32)
        rg.addWidget(self._x_start, 0, 1)
        rg.addWidget(QLabel("x end:"), 1, 0)
        self._x_end = QDoubleSpinBox()
        self._x_end.setRange(-1000, 1000)
        self._x_end.setDecimals(2)
        self._x_end.setValue(DEFAULT_X_END)
        self._x_end.setFixedHeight(32)
        rg.addWidget(self._x_end, 1, 1)
        rg.addWidget(QLabel("Samples:"), 2, 0)
        self._num_samples = QSpinBox()
        self._num_samples.setRange(50, 10000)
        self._num_samples.setValue(DEFAULT_NUM_SAMPLES)
        self._num_samples.setFixedHeight(32)
        rg.addWidget(self._num_samples, 2, 1)
        rg.addWidget(QLabel("Duration (s):"), 3, 0)
        self._duration = QDoubleSpinBox()
        self._duration.setRange(0.5, 30.0)
        self._duration.setDecimals(1)
        self._duration.setValue(3.0)
        self._duration.setFixedHeight(32)
        rg.addWidget(self._duration, 3, 1)
        left_layout.addWidget(grp_range)



        # --- Audio Controls (single row) ---
        grp_audio = QGroupBox("Audio Controls")
        al = QVBoxLayout(grp_audio)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._btn_play = QPushButton("Play")
        self._btn_play.setObjectName("playBtn")
        self._btn_play.setFixedHeight(36)
        self._btn_play.clicked.connect(self._on_play)
        btn_row.addWidget(self._btn_play)

        self._btn_pause = QPushButton("Pause")
        self._btn_pause.setFixedHeight(36)
        self._btn_pause.clicked.connect(self._on_pause)
        btn_row.addWidget(self._btn_pause)

        self._btn_replay = QPushButton("Replay")
        self._btn_replay.setFixedHeight(36)
        self._btn_replay.clicked.connect(self._on_replay)
        btn_row.addWidget(self._btn_replay)

        al.addLayout(btn_row)

        vol_row = QHBoxLayout()
        vol_row.addWidget(QLabel("Volume:"))
        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(70)
        self._volume_slider.valueChanged.connect(self._on_volume_change)
        vol_row.addWidget(self._volume_slider)
        self._vol_label = QLabel("70%")
        self._vol_label.setFixedWidth(36)
        vol_row.addWidget(self._vol_label)
        al.addLayout(vol_row)
        left_layout.addWidget(grp_audio)

        # --- Comparison Mode ---
        grp_cmp = QGroupBox("Comparison Mode")
        cl = QVBoxLayout(grp_cmp)
        self._cmp_check = QCheckBox("Enable comparison")
        self._cmp_check.toggled.connect(self._on_comparison_toggle)
        cl.addWidget(self._cmp_check)
        self._cmp_input = QLineEdit()
        self._cmp_input.setPlaceholderText("Second function g(x)…")
        self._cmp_input.setEnabled(False)
        cl.addWidget(self._cmp_input)
        left_layout.addWidget(grp_cmp)

        # --- Export & Actions ---
        grp_export = QGroupBox("Actions")
        el = QGridLayout(grp_export)
        el.setSpacing(4)
        btn_plot = QPushButton("Plot")
        btn_plot.clicked.connect(self._on_plot)
        el.addWidget(btn_plot, 0, 0)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._on_clear)
        el.addWidget(btn_clear, 0, 1)
        btn_png = QPushButton("Export PNG")
        btn_png.clicked.connect(self._on_export_png)
        el.addWidget(btn_png, 1, 0)
        btn_wav = QPushButton("Export WAV")
        btn_wav.clicked.connect(self._on_export_wav)
        el.addWidget(btn_wav, 1, 1)
        btn_save = QPushButton("Save Project")
        btn_save.clicked.connect(self._on_save_project)
        el.addWidget(btn_save, 2, 0)
        btn_load = QPushButton("Load Project")
        btn_load.clicked.connect(self._on_load_project)
        el.addWidget(btn_load, 2, 1)
        left_layout.addWidget(grp_export)

        left_layout.addStretch()
        root.addWidget(left)

        # ── Right panel (graph + analysis) ──
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self._graph = GraphWidget(parent=right, width=7, height=4, dpi=100)
        self._graph.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self._graph, stretch=3)

        # Analysis panel
        grp_analysis = QGroupBox("Function Analysis")
        anl = QVBoxLayout(grp_analysis)
        self._analysis_text = QTextEdit()
        self._analysis_text.setReadOnly(True)
        self._analysis_text.setFixedHeight(110)
        self._analysis_text.setPlaceholderText("Plot a function to see analysis…")
        anl.addWidget(self._analysis_text)
        right_layout.addWidget(grp_analysis)

        root.addWidget(right, stretch=1)

    # ──────────────────────────────────────
    # Slots
    # ──────────────────────────────────────
    def _set_preset(self, fn: str) -> None:
        self._func_input.setText(fn)
        self._on_plot()

    def _on_preset_selected(self, index: int) -> None:
        """Handle preset dropdown selection."""
        fn = self._preset_combo.currentData()
        if fn:  # Ignore the placeholder item
            self._set_preset(fn)

    def _on_clear(self) -> None:
        """Clear graph, reset both function inputs, and stop audio."""
        self._on_stop()
        self._func_input.clear()
        self._cmp_input.clear()
        self._cmp_check.setChecked(False)
        self._graph.clear_plot()
        self._analysis_text.clear()
        self._current_result = None
        self._comparison_result = None
        self._preset_combo.setCurrentIndex(0)

    def _on_comparison_toggle(self, checked: bool) -> None:
        self._cmp_input.setEnabled(checked)

    def _on_volume_change(self, value: int) -> None:
        self._audio_engine.volume = value / 100.0
        self._vol_label.setText(f"{value}%")

    def _on_plot(self) -> None:
        expr = self._func_input.text().strip()
        if not expr:
            self._show_error("Please enter a function.")
            return

        result = evaluate_function(
            expr,
            self._x_start.value(),
            self._x_end.value(),
            self._num_samples.value(),
        )
        if not result.success:
            self._show_error(result.error_message)
            return

        self._current_result = result
        self._graph.plot_function(result.x_values, result.y_values, label=f"f(x) = {expr}")

        # Comparison
        if self._cmp_check.isChecked() and self._cmp_input.text().strip():
            cmp_expr = self._cmp_input.text().strip()
            cmp_result = evaluate_function(
                cmp_expr, self._x_start.value(),
                self._x_end.value(), self._num_samples.value(),
            )
            if cmp_result.success:
                self._comparison_result = cmp_result
                self._graph.plot_comparison(cmp_result.x_values, cmp_result.y_values, label2=f"g(x) = {cmp_expr}")
            else:
                self._comparison_result = None
                self._show_error(f"Comparison function error: {cmp_result.error_message}")
        else:
            self._comparison_result = None

        # Analysis
        analysis = analyze_function(expr, result.x_values, result.y_values)
        if analysis:
            lines = [
                f"Maximum value:    {analysis.max_value}",
                f"Minimum value:    {analysis.min_value}",
                f"Discontinuities:  {analysis.num_discontinuities}",
                f"Period:           {analysis.approximate_period if analysis.approximate_period else 'N/A (non-periodic or unknown)'}",
            ]
            self._analysis_text.setPlainText("\n".join(lines))

        # Pre-generate audio
        self._generate_audio()

    def _generate_audio(self) -> None:
        if self._current_result is None:
            return
        self._audio_engine.duration = self._duration.value()
        ch2 = self._comparison_result.y_values if self._comparison_result else None
        self._audio_engine.generate_audio(self._current_result.y_values, mode="A", y_values_ch2=ch2)

    def _on_play(self) -> None:
        if self._current_result is None:
            self._show_error("Plot a function first.")
            return
        if self._audio_engine.state == PlaybackState.STOPPED:
            self._generate_audio()
        if self._audio_engine.play():
            self._cursor_timer.start()

    def _on_pause(self) -> None:
        self._audio_engine.pause()
        self._cursor_timer.stop()

    def _on_stop(self) -> None:
        self._audio_engine.stop()
        self._cursor_timer.stop()
        self._graph.clear_cursor()

    def _on_replay(self) -> None:
        self._on_stop()
        self._on_play()

    def _on_progress(self, progress: float) -> None:
        pass  # Cursor is updated via timer for smoother motion

    def _on_playback_finished(self) -> None:
        self._cursor_timer.stop()
        self._graph.clear_cursor()

    def _update_cursor(self) -> None:
        progress = self._audio_engine.progress
        self._graph.update_cursor(progress)

    # ── Export ──
    def _on_export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Graph", "graph.png", "PNG Files (*.png)")
        if path:
            ok, msg = export_graph_as_png(self._graph.get_figure(), path)
            self._show_info(msg) if ok else self._show_error(msg)

    def _on_export_wav(self) -> None:
        audio = self._audio_engine.get_audio_data()
        if audio is None:
            self._show_error("Generate audio first (plot and play).")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Audio", "sonification.wav", "WAV Files (*.wav)")
        if path:
            ok, msg = export_audio_as_wav(audio, self._audio_engine.get_sample_rate(), path)
            self._show_info(msg) if ok else self._show_error(msg)

    def _on_save_project(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "project.json", "JSON Files (*.json)")
        if path:
            s = ProjectSettings(
                function_expr=self._func_input.text(),
                x_start=self._x_start.value(),
                x_end=self._x_end.value(),
                num_samples=self._num_samples.value(),
                duration=self._duration.value(),
                sonification_mode="A",
                volume=self._volume_slider.value() / 100.0,
                comparison_enabled=self._cmp_check.isChecked(),
                comparison_function=self._cmp_input.text(),
            )
            ok, msg = save_project_settings(s, path)
            self._show_info(msg) if ok else self._show_error(msg)

    def _on_load_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "JSON Files (*.json)")
        if path:
            ok, settings, msg = load_project_settings(path)
            if ok and settings:
                self._func_input.setText(settings.function_expr)
                self._x_start.setValue(settings.x_start)
                self._x_end.setValue(settings.x_end)
                self._num_samples.setValue(settings.num_samples)
                self._duration.setValue(settings.duration)
                self._volume_slider.setValue(int(settings.volume * 100))
                self._cmp_check.setChecked(settings.comparison_enabled)
                self._cmp_input.setText(settings.comparison_function)
                self._on_plot()
                self._show_info(msg)
            else:
                self._show_error(msg)

    # ── Dialogs ──
    def _show_error(self, msg: str) -> None:
        QMessageBox.warning(self, "Error", msg)

    def _show_info(self, msg: str) -> None:
        QMessageBox.information(self, "Success", msg)
