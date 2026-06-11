"""
settings.py — Default settings and configuration for Function Sonifier.

Contains all configurable constants and a Settings dataclass for
serialization/deserialization of project state.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# ──────────────────────────────────────────────
# Audio Mapping Defaults
# ──────────────────────────────────────────────
DEFAULT_MIN_FREQ: float = 220.0   # Hz (A3)
DEFAULT_MAX_FREQ: float = 880.0   # Hz (A5)
DEFAULT_SAMPLE_RATE: int = 44100  # CD-quality sample rate
DEFAULT_DURATION: float = 3.0     # seconds of playback

# ──────────────────────────────────────────────
# Function Evaluation Defaults
# ──────────────────────────────────────────────
DEFAULT_X_START: float = -6.28    # approx -2π
DEFAULT_X_END: float = 6.28      # approx  2π
DEFAULT_NUM_SAMPLES: int = 500

# ──────────────────────────────────────────────
# Musical Scale (Mode D) — C Major Pentatonic
# Frequencies for two octaves of C major pentatonic
# ──────────────────────────────────────────────
MUSICAL_SCALE_NOTES: list[str] = [
    "C4", "D4", "E4", "G4", "A4",
    "C5", "D5", "E5", "G5", "A5",
    "C6"
]
MUSICAL_SCALE_FREQS: list[float] = [
    261.63, 293.66, 329.63, 392.00, 440.00,
    523.25, 587.33, 659.25, 783.99, 880.00,
    1046.50
]

# ──────────────────────────────────────────────
# Sonification Modes
# ──────────────────────────────────────────────
SONIFICATION_MODES: dict[str, str] = {
    "A": "Pitch Only",
    "B": "Pitch + Volume",
    "C": "Pitch + Stereo Panning",
    "D": "Musical Scale",
}

# ──────────────────────────────────────────────
# Preset Functions
# ──────────────────────────────────────────────
PRESET_FUNCTIONS: list[str] = [
    "sin(x)",
    "cos(x)",
    "tan(x)",
    "x**2",
    "x**3",
    "exp(x)",
    "log(x)",
    "abs(x)",
]


@dataclass
class ProjectSettings:
    """Serializable project settings for save/load functionality."""
    function_expr: str = "sin(x)"
    x_start: float = DEFAULT_X_START
    x_end: float = DEFAULT_X_END
    num_samples: int = DEFAULT_NUM_SAMPLES
    min_freq: float = DEFAULT_MIN_FREQ
    max_freq: float = DEFAULT_MAX_FREQ
    duration: float = DEFAULT_DURATION
    sonification_mode: str = "A"
    volume: float = 0.7
    # Comparison mode
    comparison_enabled: bool = False
    comparison_function: str = ""

    def to_json(self) -> str:
        """Serialize settings to JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ProjectSettings":
        """Deserialize settings from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
