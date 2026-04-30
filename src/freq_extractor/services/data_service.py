"""Data service — signal generation and dataset construction.

Contains ``SignalGenerator`` for producing clean and noisy sinusoids,
and ``DatasetBuilder`` for creating sliding-window dataset entries.

References
----------
- PRD FR-1, FR-2, PRD_sig §§2–5, PLAN §4
"""

from __future__ import annotations

import logging

import numpy as np

from freq_extractor.constants import FREQUENCY_INDEX, NUM_CLASSES

logger = logging.getLogger("freq_extractor.data_service")


class SignalGenerator:
    """Generate clean and noisy sinusoidal signals.

    Parameters
    ----------
    sampling_rate : int
        Sampling frequency in Hz.
    duration : float
        Signal duration in seconds.
    amplitude : float
        Peak amplitude of the sinusoid.
    """

    def __init__(self, sampling_rate: int, duration: float, amplitude: float) -> None:
        """Validate parameters and store configuration."""
        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be > 0, got {sampling_rate}")
        if duration <= 0:
            raise ValueError(f"duration must be > 0, got {duration}")
        if amplitude < 0:
            raise ValueError(f"amplitude must be >= 0, got {amplitude}")
        self.fs = sampling_rate
        self.duration = duration
        self.amplitude = amplitude
        self.t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    def validate_nyquist(self, frequencies: list[float]) -> None:
        """Raise ``ValueError`` if any frequency violates the Nyquist criterion."""
        f_max = max(frequencies)
        if f_max > self.fs / 2:
            raise ValueError(
                f"Nyquist violation: max freq {f_max} Hz > fs/2 = {self.fs / 2} Hz"
            )

    def generate_clean(self, freq: float, phase: float = 0.0) -> np.ndarray:
        """Return a clean sinusoidal signal ``A·sin(2π·f·t + φ)``."""
        if freq <= 0:
            raise ValueError(f"Frequency must be > 0, got {freq}")
        return self.amplitude * np.sin(2 * np.pi * freq * self.t + phase)

    def generate_noisy(
        self, clean: np.ndarray, noise_std_ratio: float, rng: np.random.Generator,
    ) -> np.ndarray:
        """Add Gaussian noise ``ε ~ N(0, (σ·A)²)`` to a clean signal."""
        if noise_std_ratio < 0:
            raise ValueError(f"noise_std_ratio must be >= 0, got {noise_std_ratio}")
        if noise_std_ratio == 0:
            return clean.copy()
        std = noise_std_ratio * self.amplitude
        return clean + rng.normal(0.0, std, size=clean.shape)


class DatasetBuilder:
    """Construct sliding-window dataset entries from signals."""

    @staticmethod
    def build_windows(
        clean: np.ndarray, noisy: np.ndarray, freq: int, window_size: int,
    ) -> list[dict[str, np.ndarray]]:
        """Create sliding-window entries with one-hot labels.

        Returns list of dicts with keys:
        ``frequency_label``, ``noisy_samples``, ``clean_samples``, ``target_output``.
        """
        n = len(clean)
        if window_size >= n:
            raise ValueError(f"window_size ({window_size}) must be < signal length ({n})")
        one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
        one_hot[FREQUENCY_INDEX[freq]] = 1.0
        entries: list[dict[str, np.ndarray]] = []
        for i in range(n - window_size):
            entries.append({
                "frequency_label": one_hot.copy(),
                "noisy_samples": noisy[i : i + window_size].astype(np.float32),
                "clean_samples": clean[i : i + window_size].astype(np.float32),
                "target_output": np.array([clean[i + window_size]], dtype=np.float32),
            })
        return entries
