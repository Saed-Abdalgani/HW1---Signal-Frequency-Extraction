"""Tests for UI signal helpers used by Dash callbacks."""

from __future__ import annotations

import numpy as np

from freq_extractor.services.ui_signal_utils import (
    add_noise,
    apply_bpf,
    apply_filter,
    compute_shared_grid,
    gen_signal,
    signal_on_grid,
)


class TestGenSignal:
    """Signal generation helper tests."""

    def test_returns_time_and_signal(self) -> None:
        """gen_signal returns aligned arrays."""
        t, sig = gen_signal(200.0, 2.0, 10.0, 0.0, 1.0)
        assert len(t) == len(sig)
        assert len(t) > 0

    def test_frequency_in_signal(self) -> None:
        """Dominant FFT bin near configured frequency."""
        t, sig = gen_signal(1000.0, 5.0, 4.0, 0.0, 1.0)
        freqs = np.fft.rfftfreq(len(sig), d=1 / 1000.0)
        mags = np.abs(np.fft.rfft(sig))
        dominant = freqs[np.argmax(mags)]
        assert abs(dominant - 4.0) < 1.0

    def test_amplitude_scaling(self) -> None:
        """Amplitude scales peak."""
        _, sig = gen_signal(200.0, 2.0, 4.0, 0.0, 2.0)
        assert np.max(np.abs(sig)) <= 2.0 + 0.01

    def test_zero_freq_uses_fallback(self) -> None:
        """Zero frequency uses internal fallback."""
        t, sig = gen_signal(200.0, 2.0, 0.0, 0.0, 1.0)
        assert len(t) > 0


class TestSharedGrid:
    """Shared time grid."""

    def test_signal_on_shared_grid(self) -> None:
        """Same *t* length for two frequencies."""
        t = compute_shared_grid(200, 3, 2.0)
        s1 = signal_on_grid(t, 4.0, 0.0, 1.0)
        s2 = signal_on_grid(t, 3.0, 0.0, 1.0)
        assert len(s1) == len(s2) == len(t)

    def test_grid_symmetric_about_origin(self) -> None:
        """Time samples pair as ``t[k] ~= -t[-1-k]``."""
        t = compute_shared_grid(200, 3, 2.0)
        assert np.allclose(t + t[::-1], 0.0)

    def test_sine_pi2_phase_zero_reflects_about_t0(self) -> None:
        """sin(2πft + π/2) on symmetric grid mirrors left–right when φ is 0."""
        t = compute_shared_grid(200, 3, 2.0)
        sig = signal_on_grid(t, 4.2, 0.0, 1.0)
        assert np.allclose(sig, sig[::-1])


class TestAddNoise:
    """Noise addition."""

    def test_none_returns_clean(self) -> None:
        """None type leaves signal unchanged."""
        sig = np.sin(np.linspace(0, 1, 100))
        rng = np.random.default_rng(42)
        result = add_noise(sig, "None", rng)
        np.testing.assert_array_equal(result, sig)

    def test_gaussian_modifies_signal(self) -> None:
        """Gaussian changes values."""
        sig = np.sin(np.linspace(0, 1, 100))
        rng = np.random.default_rng(42)
        result = add_noise(sig, "Gaussian", rng)
        assert not np.array_equal(result, sig)

    def test_uniform_modifies_signal(self) -> None:
        """Uniform changes values."""
        sig = np.sin(np.linspace(0, 1, 100))
        rng = np.random.default_rng(42)
        result = add_noise(sig, "Uniform", rng)
        assert not np.array_equal(result, sig)

    def test_none_noise_zeros_unchanged(self) -> None:
        """Noise type None keeps a zero baseline."""
        sig = np.zeros(50)
        rng = np.random.default_rng(0)
        out = add_noise(sig, "None", rng)
        np.testing.assert_array_equal(out, sig)


class TestApplyBPF:
    """BPF checkbox path."""

    def test_filter_returns_same_length(self) -> None:
        """Output length matches input."""
        sig = np.sin(np.linspace(0, 1, 1000))
        result = apply_bpf(sig, 4.0, 2.0, 200.0)
        assert len(result) == len(sig)

    def test_invalid_band_returns_original(self) -> None:
        """Degenerate band returns original."""
        sig = np.sin(np.linspace(0, 1, 100))
        result = apply_bpf(sig, 0.0, 0.0, 200.0)
        np.testing.assert_array_equal(result, sig)


class TestApplyFilterDropdown:
    """Dropdown filter."""

    def test_none_passthrough(self) -> None:
        """None filter is identity."""
        sig = np.sin(np.linspace(0, 1, 200))
        out = apply_filter(sig, 3.0, 5.0, 200.0, "None")
        np.testing.assert_array_equal(out, sig)
