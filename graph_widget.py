"""
graph_widget.py — Matplotlib-based graph widget embedded in PyQt6.

Displays function plots with a clean black-and-white theme, supports
cursor tracking during audio playback, and comparison mode for
overlaying two functions.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from typing import Optional


class GraphWidget(FigureCanvas):
    """
    A Matplotlib canvas widget for PyQt6 that displays function plots
    with real-time cursor tracking.
    """

    def __init__(self, parent=None, width: int = 8, height: int = 4, dpi: int = 100):
        self._fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#ffffff')
        self._ax = self._fig.add_subplot(111)
        super().__init__(self._fig)
        self.setParent(parent)

        # Plot elements
        self._line1: Optional[Line2D] = None
        self._line2: Optional[Line2D] = None
        self._cursor_line: Optional[Line2D] = None

        # Data references
        self._x_data: Optional[np.ndarray] = None
        self._y_data: Optional[np.ndarray] = None
        self._x_data2: Optional[np.ndarray] = None
        self._y_data2: Optional[np.ndarray] = None

        # Initial styling
        self._setup_style()

    def _setup_style(self) -> None:
        """Apply minimalist black-and-white styling to the axes."""
        self._ax.set_facecolor('#ffffff')
        self._ax.tick_params(colors='#222222', labelsize=9)
        self._ax.spines['bottom'].set_color('#222222')
        self._ax.spines['top'].set_color('#cccccc')
        self._ax.spines['left'].set_color('#222222')
        self._ax.spines['right'].set_color('#cccccc')
        self._ax.grid(True, linestyle='--', alpha=0.3, color='#888888')
        self._ax.set_xlabel('x', fontsize=10, color='#222222')
        self._ax.set_ylabel('f(x)', fontsize=10, color='#222222')
        self._fig.tight_layout(pad=2.0)

    def plot_function(
        self,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        label: str = "f(x)",
        color: str = '#000000',
        clear: bool = True
    ) -> None:
        """
        Plot a function on the graph.

        Args:
            x_vals: Array of x-coordinates.
            y_vals: Array of y-coordinates.
            label:  Legend label for the function.
            color:  Line color (hex string).
            clear:  Whether to clear existing plots first.
        """
        if clear:
            self._ax.cla()
            self._setup_style()
            self._line2 = None
            self._cursor_line = None

        self._x_data = x_vals
        self._y_data = y_vals

        self._line1, = self._ax.plot(
            x_vals, y_vals,
            color=color,
            linewidth=1.8,
            label=label,
            antialiased=True,
        )

        # Set reasonable y-axis limits
        y_finite = y_vals[np.isfinite(y_vals)]
        if len(y_finite) > 0:
            y_min, y_max = np.min(y_finite), np.max(y_finite)
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
            self._ax.set_ylim(y_min - margin, y_max + margin)

        self._ax.set_title(label, fontsize=11, color='#222222', pad=8)
        self._fig.tight_layout(pad=2.0)
        self.draw()

    def plot_comparison(
        self,
        x_vals2: np.ndarray,
        y_vals2: np.ndarray,
        label2: str = "g(x)"
    ) -> None:
        """
        Overlay a second function for comparison mode.

        Args:
            x_vals2: x-coordinates for the second function.
            y_vals2: y-coordinates for the second function.
            label2:  Legend label for the second function.
        """
        self._x_data2 = x_vals2
        self._y_data2 = y_vals2

        self._line2, = self._ax.plot(
            x_vals2, y_vals2,
            color='#666666',
            linewidth=1.8,
            linestyle='--',
            label=label2,
            antialiased=True,
        )

        self._ax.legend(
            fontsize=9, framealpha=0.9,
            edgecolor='#cccccc', facecolor='#ffffff'
        )

        # Adjust y-axis to fit both functions
        all_y = np.concatenate([
            self._y_data[np.isfinite(self._y_data)],
            y_vals2[np.isfinite(y_vals2)]
        ])
        if len(all_y) > 0:
            y_min, y_max = np.min(all_y), np.max(all_y)
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
            self._ax.set_ylim(y_min - margin, y_max + margin)

        self._ax.set_title(
            f"f(x)  vs  g(x)", fontsize=11, color='#222222', pad=8
        )
        self._fig.tight_layout(pad=2.0)
        self.draw()

    def update_cursor(self, progress: float) -> None:
        """
        Update the vertical cursor line position based on playback progress.

        Args:
            progress: Playback progress as a fraction [0.0, 1.0].
        """
        if self._x_data is None:
            return

        # Calculate x position from progress
        idx = int(progress * (len(self._x_data) - 1))
        idx = max(0, min(idx, len(self._x_data) - 1))
        x_pos = self._x_data[idx]

        # Remove old cursor
        if self._cursor_line is not None:
            try:
                self._cursor_line.remove()
            except ValueError:
                pass

        # Draw new cursor
        y_lim = self._ax.get_ylim()
        self._cursor_line, = self._ax.plot(
            [x_pos, x_pos], y_lim,
            color='#ff0000',
            linewidth=1.2,
            linestyle='-',
            alpha=0.7,
        )
        self._ax.set_ylim(y_lim)  # Prevent auto-scaling from cursor
        self.draw_idle()

    def clear_cursor(self) -> None:
        """Remove the cursor line from the plot."""
        if self._cursor_line is not None:
            try:
                self._cursor_line.remove()
            except ValueError:
                pass
            self._cursor_line = None
            self.draw_idle()

    def clear_plot(self) -> None:
        """Clear all plot elements and reset the graph."""
        self._ax.cla()
        self._setup_style()
        self._line1 = None
        self._line2 = None
        self._cursor_line = None
        self._x_data = None
        self._y_data = None
        self._x_data2 = None
        self._y_data2 = None
        self.draw()

    def get_figure(self) -> Figure:
        """Return the Matplotlib Figure object (for export)."""
        return self._fig
