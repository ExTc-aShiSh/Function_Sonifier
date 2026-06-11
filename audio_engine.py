"""
audio_engine.py — Sonification engine for Function Sonifier.

Converts mathematical function values into audio signals using multiple
sonification modes. Handles real-time playback with threading to keep
the GUI responsive.
"""

import numpy as np
import sounddevice as sd
import threading
from typing import Optional, Callable
from enum import Enum

from settings import (
    DEFAULT_MIN_FREQ, DEFAULT_MAX_FREQ,
    DEFAULT_SAMPLE_RATE, DEFAULT_DURATION,
    MUSICAL_SCALE_FREQS,
)


class PlaybackState(Enum):
    """Current state of audio playback."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


class SonificationMode(Enum):
    """Available sonification mapping modes."""
    PITCH_ONLY = "A"          # y → frequency
    PITCH_VOLUME = "B"        # y → frequency + amplitude
    PITCH_PANNING = "C"       # y → frequency + stereo pan
    MUSICAL_SCALE = "D"       # y → nearest scale note


class AudioEngine:
    """
    Generates and controls playback of sonified function audio.

    Maps evaluated y-values to audio parameters (pitch, volume, panning)
    and provides threaded playback with play/pause/stop controls.
    """

    def __init__(self) -> None:
        self._sample_rate: int = DEFAULT_SAMPLE_RATE
        self._duration: float = DEFAULT_DURATION
        self._min_freq: float = DEFAULT_MIN_FREQ
        self._max_freq: float = DEFAULT_MAX_FREQ
        self._volume: float = 0.7

        # Playback state
        self._state: PlaybackState = PlaybackState.STOPPED
        self._audio_data: Optional[np.ndarray] = None
        self._audio_data_ch2: Optional[np.ndarray] = None  # For comparison mode
        self._playback_thread: Optional[threading.Thread] = None
        self._current_frame: int = 0
        self._total_frames: int = 0
        self._lock: threading.Lock = threading.Lock()

        # Callback for cursor position updates
        self._progress_callback: Optional[Callable[[float], None]] = None
        self._playback_finished_callback: Optional[Callable[[], None]] = None

    # ──────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def progress(self) -> float:
        """Current playback progress as a fraction [0.0, 1.0]."""
        if self._total_frames == 0:
            return 0.0
        return min(self._current_frame / self._total_frames, 1.0)

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        self._volume = max(0.0, min(1.0, value))

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        self._duration = max(0.5, min(30.0, value))

    # ──────────────────────────────────────────
    # Callback Registration
    # ──────────────────────────────────────────

    def set_progress_callback(self, callback: Callable[[float], None]) -> None:
        """Register a callback invoked with progress fraction during playback."""
        self._progress_callback = callback

    def set_finished_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked when playback finishes."""
        self._playback_finished_callback = callback

    # ──────────────────────────────────────────
    # Audio Generation
    # ──────────────────────────────────────────

    def generate_audio(
        self,
        y_values: np.ndarray,
        mode: str = "A",
        y_values_ch2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate audio samples from function y-values.

        Args:
            y_values:    Normalized y-values of the primary function.
            mode:        Sonification mode ('A', 'B', 'C', or 'D').
            y_values_ch2: Optional second channel y-values for comparison.

        Returns:
            NumPy array of audio samples (mono or stereo).
        """
        total_samples = int(self._sample_rate * self._duration)

        # Normalize y-values to [0, 1]
        y_norm = self._normalize(y_values)

        # Resample y_values to match total audio samples
        y_resampled = np.interp(
            np.linspace(0, len(y_norm) - 1, total_samples),
            np.arange(len(y_norm)),
            y_norm
        )

        # Generate audio based on mode
        if mode == "D":
            audio = self._generate_musical(y_resampled, total_samples)
        else:
            audio = self._generate_continuous(y_resampled, total_samples, mode)

        # Apply fade-in and fade-out to prevent clicks
        audio = self._apply_fade(audio)

        # Handle comparison mode (second channel)
        if y_values_ch2 is not None:
            y_norm2 = self._normalize(y_values_ch2)
            y_resampled2 = np.interp(
                np.linspace(0, len(y_norm2) - 1, total_samples),
                np.arange(len(y_norm2)),
                y_norm2
            )
            if mode == "D":
                audio2 = self._generate_musical(y_resampled2, total_samples)
            else:
                audio2 = self._generate_continuous(y_resampled2, total_samples, "A")
            audio2 = self._apply_fade(audio2)

            # Mix into stereo: ch1 left, ch2 right
            if audio.ndim == 1:
                stereo = np.column_stack([audio, audio2])
            else:
                stereo = np.column_stack([audio[:, 0], audio2[:, 0] if audio2.ndim > 1 else audio2])
            self._audio_data = stereo
        else:
            self._audio_data = audio

        self._total_frames = len(self._audio_data)
        self._current_frame = 0
        return self._audio_data

    def _normalize(self, y: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        y_min, y_max = np.nanmin(y), np.nanmax(y)
        if y_max - y_min < 1e-10:
            return np.full_like(y, 0.5)
        return (y - y_min) / (y_max - y_min)

    def _generate_continuous(
        self, y_norm: np.ndarray, total_samples: int, mode: str
    ) -> np.ndarray:
        """
        Generate continuous-frequency audio (Modes A, B, C).

        Args:
            y_norm:        Normalized y-values resampled to total_samples.
            total_samples: Total number of audio samples.
            mode:          'A' (pitch), 'B' (pitch+vol), 'C' (pitch+pan).

        Returns:
            Audio signal as numpy array.
        """
        # Map normalized values to frequencies
        freqs = self._min_freq + y_norm * (self._max_freq - self._min_freq)

        # Generate phase-continuous sine wave using cumulative phase
        dt = 1.0 / self._sample_rate
        phase = np.cumsum(2.0 * np.pi * freqs * dt)
        signal = np.sin(phase)

        if mode == "A":
            # Pitch only — uniform amplitude
            return signal * self._volume

        elif mode == "B":
            # Pitch + volume — amplitude follows normalized y
            amplitude = 0.3 + 0.7 * y_norm  # Range [0.3, 1.0]
            return signal * amplitude * self._volume

        elif mode == "C":
            # Pitch + stereo panning
            pan = y_norm  # 0 = full left, 1 = full right
            left = signal * np.sqrt(1.0 - pan) * self._volume
            right = signal * np.sqrt(pan) * self._volume
            return np.column_stack([left, right])

        return signal * self._volume

    def _generate_musical(
        self, y_norm: np.ndarray, total_samples: int
    ) -> np.ndarray:
        """
        Generate audio using musical scale quantization (Mode D).

        Maps each y-value to the nearest note in the predefined musical scale,
        producing a stepped, melodic output.
        """
        scale_freqs = np.array(MUSICAL_SCALE_FREQS)
        num_notes = len(scale_freqs)

        # Map normalized values to scale indices
        indices = np.clip(
            np.round(y_norm * (num_notes - 1)).astype(int),
            0, num_notes - 1
        )
        freqs = scale_freqs[indices]

        # Smooth frequency transitions to avoid harsh jumps
        # Use a small window moving average
        window = min(500, total_samples // 50)
        if window > 1:
            kernel = np.ones(window) / window
            freqs = np.convolve(freqs, kernel, mode='same')

        # Generate phase-continuous sine wave
        dt = 1.0 / self._sample_rate
        phase = np.cumsum(2.0 * np.pi * freqs * dt)
        signal = np.sin(phase)

        return signal * self._volume

    def _apply_fade(self, audio: np.ndarray, fade_ms: int = 20) -> np.ndarray:
        """Apply fade-in/fade-out to prevent audio clicks."""
        fade_samples = int(self._sample_rate * fade_ms / 1000)
        if fade_samples >= len(audio) // 2:
            fade_samples = len(audio) // 4

        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)

        if audio.ndim == 1:
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
        else:
            audio[:fade_samples, :] *= fade_in[:, np.newaxis]
            audio[-fade_samples:, :] *= fade_out[:, np.newaxis]

        return audio

    # ──────────────────────────────────────────
    # Playback Controls
    # ──────────────────────────────────────────

    def play(self) -> bool:
        """
        Start or resume audio playback.

        Returns:
            True if playback started successfully.
        """
        if self._audio_data is None:
            return False

        with self._lock:
            if self._state == PlaybackState.PLAYING:
                return True

            if self._state == PlaybackState.STOPPED:
                self._current_frame = 0

            self._state = PlaybackState.PLAYING

        # Start playback in a separate thread
        self._playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True
        )
        self._playback_thread.start()
        return True

    def pause(self) -> None:
        """Pause audio playback (can be resumed)."""
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                self._state = PlaybackState.PAUSED
                sd.stop()

    def stop(self) -> None:
        """Stop audio playback and reset position."""
        with self._lock:
            self._state = PlaybackState.STOPPED
            self._current_frame = 0
            sd.stop()

    def _playback_worker(self) -> None:
        """
        Background thread that streams audio and reports progress.
        Uses a block-based approach for responsive pause/stop.
        """
        block_size = 2048
        channels = 2 if self._audio_data.ndim > 1 else 1

        try:
            stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=channels,
                dtype='float32',
                blocksize=block_size,
            )
            stream.start()

            while True:
                with self._lock:
                    if self._state != PlaybackState.PLAYING:
                        break
                    start = self._current_frame
                    end = min(start + block_size, self._total_frames)

                if start >= self._total_frames:
                    break

                # Write audio block
                block = self._audio_data[start:end].astype(np.float32)
                if block.ndim == 1:
                    block = block.reshape(-1, 1)
                stream.write(block)

                with self._lock:
                    self._current_frame = end

                # Report progress
                if self._progress_callback:
                    progress = end / self._total_frames
                    self._progress_callback(progress)

            stream.stop()
            stream.close()

        except Exception as e:
            print(f"Playback error: {e}")

        # Signal completion
        with self._lock:
            if self._state == PlaybackState.PLAYING:
                self._state = PlaybackState.STOPPED
                self._current_frame = 0

        if self._playback_finished_callback:
            self._playback_finished_callback()

    # ──────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────

    def get_audio_data(self) -> Optional[np.ndarray]:
        """Return the current generated audio data."""
        return self._audio_data

    def get_sample_rate(self) -> int:
        """Return the current sample rate."""
        return self._sample_rate
