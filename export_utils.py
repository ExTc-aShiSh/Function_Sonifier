"""
export_utils.py — Export utilities for Function Sonifier.

Provides functions to export graphs as PNG, audio as WAV,
and save/load project settings as JSON files.
"""

import json
import numpy as np
from scipy.io import wavfile
from matplotlib.figure import Figure
from typing import Optional

from settings import ProjectSettings


def export_graph_as_png(fig: Figure, filepath: str) -> tuple[bool, str]:
    """
    Export a Matplotlib figure as a PNG image.

    Args:
        fig:      The Matplotlib Figure to export.
        filepath: Destination file path (should end in .png).

    Returns:
        Tuple of (success, message).
    """
    try:
        fig.savefig(
            filepath,
            dpi=200,
            bbox_inches='tight',
            facecolor='#ffffff',
            edgecolor='none',
        )
        return True, f"Graph exported to {filepath}"
    except Exception as e:
        return False, f"Export failed: {str(e)}"


def export_audio_as_wav(
    audio_data: np.ndarray,
    sample_rate: int,
    filepath: str
) -> tuple[bool, str]:
    """
    Export audio data as a WAV file.

    Args:
        audio_data:  NumPy array of audio samples (mono or stereo).
        sample_rate: Sample rate in Hz.
        filepath:    Destination file path (should end in .wav).

    Returns:
        Tuple of (success, message).
    """
    try:
        # Normalize to 16-bit integer range
        if audio_data.dtype != np.int16:
            peak = np.max(np.abs(audio_data))
            if peak > 0:
                normalized = audio_data / peak
            else:
                normalized = audio_data
            int_data = (normalized * 32767).astype(np.int16)
        else:
            int_data = audio_data

        wavfile.write(filepath, sample_rate, int_data)
        return True, f"Audio exported to {filepath}"
    except Exception as e:
        return False, f"Export failed: {str(e)}"


def save_project_settings(
    settings: ProjectSettings,
    filepath: str
) -> tuple[bool, str]:
    """
    Save project settings to a JSON file.

    Args:
        settings: ProjectSettings instance to serialize.
        filepath: Destination file path (should end in .json).

    Returns:
        Tuple of (success, message).
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(settings.to_json())
        return True, f"Settings saved to {filepath}"
    except Exception as e:
        return False, f"Save failed: {str(e)}"


def load_project_settings(filepath: str) -> tuple[bool, Optional[ProjectSettings], str]:
    """
    Load project settings from a JSON file.

    Args:
        filepath: Source file path to read.

    Returns:
        Tuple of (success, settings_or_None, message).
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        settings = ProjectSettings.from_json(content)
        return True, settings, f"Settings loaded from {filepath}"
    except Exception as e:
        return False, None, f"Load failed: {str(e)}"
