"""Tests for UI callback helpers — signal generation, noise, BPF.

Validates the internal helper functions used by Dash callbacks.
"""

from __future__ import annotations

import numpy as np

from freq_extractor.services.ui_callbacks import _add_noise, _apply_bpf, _gen_signal


class TestGenSignal:
    """Signal generation helper tests."""

    def test_returns_time_and_signal(self) -> None:
        """_gen_signal returns (time_array, signal_array) of same length."""
        t, sig = _gen_signal(200.0, 2.0, 10.0, 0.0, 1.0)
        assert len(t) == len(sig)
        assert len(t) > 0

    def test_frequency_in_signal(self) -> None:
        """Generated signal has correct dominant frequency."""
        t, sig = _gen_signal(1000.0, 5.0, 25.0, 0.0, 1.0)
        freqs = np.fft.rfftfreq(len(sig), d=1 / 1000.0)
        mags = np.abs(np.fft.rfft(sig))
        dominant = freqs[np.argmax(mags)]
        assert abs(dominant - 25.0) < 1.0

    def test_amplitude_scaling(self) -> None:
        """Amplitude parameter scales signal peak."""
        _, sig = _gen_signal(200.0, 2.0, 10.0, 0.0, 2.0)
        assert np.max(np.abs(sig)) <= 2.0 + 0.01

    def test_zero_freq_uses_fallback(self) -> None:
        """Zero frequency falls back to 0.1 Hz."""
        t, sig = _gen_signal(200.0, 2.0, 0.0, 0.0, 1.0)
        assert len(t) > 0


class TestAddNoise:
    """Noise addition tests."""

    def test_none_returns_clean(self) -> None:
        """No noise returns signal unchanged."""
        sig = np.sin(np.linspace(0, 1, 100))
        rng = np.random.default_rng(42)
        result = _add_noise(sig, "None", rng)
        np.testing.assert_array_equal(result, sig)

    def test_gaussian_modifies_signal(self) -> None:
        """6K.8: Gaussian noise changes the signal."""
        sig = np.sin(np.linspace(0, 1, 100))
        rng = np.random.default_rng(42)
        result = _add_noise(sig, "Gaussian", rng)
        assert not np.array_equal(result, sig)

    def test_uniform_modifies_signal(self) -> None:
        """Uniform noise changes the signal."""
        sig = np.sin(np.linspace(0, 1, 100))
        rng = np.random.default_rng(42)
        result = _add_noise(sig, "Uniform", rng)
        assert not np.array_equal(result, sig)


class TestApplyBPF:
    """Bandpass filter tests."""

    def test_filter_returns_same_length(self) -> None:
        """6K.7: Filtered signal has same length as input."""
        sig = np.sin(np.linspace(0, 1, 1000))
        result = _apply_bpf(sig, 10.0, 5.0, 200.0)
        assert len(result) == len(sig)

    def test_invalid_band_returns_original(self) -> None:
        """Invalid frequency band returns signal unchanged."""
        sig = np.sin(np.linspace(0, 1, 100))
        result = _apply_bpf(sig, 0.0, 0.0, 200.0)
        np.testing.assert_array_equal(result, sig)
