"""Tests for SignalGenerator — SG-T1, SG-T2, SG-T5, SG-T11..SG-T15.

Validates FFT correctness, noise statistics, Nyquist guards,
and edge-case input validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import rfft, rfftfreq

from freq_extractor.services.data_service import SignalGenerator


class TestSignalGeneratorClean:
    """Clean signal generation tests."""

    def test_fft_peak_matches_frequency(self, signal_gen: SignalGenerator) -> None:
        """SG-T1: FFT dominant frequency matches a configured class frequency."""
        sig = signal_gen.generate_clean(4.0)
        freqs = rfftfreq(len(sig), d=1.0 / signal_gen.fs)
        magnitudes = np.abs(rfft(sig))
        dominant = freqs[np.argmax(magnitudes)]
        assert abs(dominant - 4.0) < 0.5, f"Expected ~4 Hz, got {dominant}"

    def test_fft_peak_all_frequencies(self, signal_gen: SignalGenerator) -> None:
        """SG-T1 extended: verify FFT peak for all 4 configured frequencies."""
        from freq_extractor.constants import FREQUENCY_LABELS

        for freq in FREQUENCY_LABELS:
            sig = signal_gen.generate_clean(float(freq))
            freqs_ax = rfftfreq(len(sig), d=1.0 / signal_gen.fs)
            dominant = freqs_ax[np.argmax(np.abs(rfft(sig)))]
            assert abs(dominant - freq) < 0.5

    def test_signal_length(self, signal_gen: SignalGenerator) -> None:
        """Signal length equals fs * duration."""
        sig = signal_gen.generate_clean(10.0)
        assert len(sig) == int(signal_gen.fs * signal_gen.duration)

    def test_amplitude_bounded(self, signal_gen: SignalGenerator) -> None:
        """Peak amplitude does not exceed configured amplitude."""
        sig = signal_gen.generate_clean(6.0)
        assert np.max(np.abs(sig)) <= signal_gen.amplitude + 1e-6


class TestSignalGeneratorNoisy:
    """Noisy signal generation tests."""

    def test_snr_at_sigma_01(self, signal_gen: SignalGenerator, rng) -> None:
        """SG-T2: SNR ≈ 20 dB at σ = 0.1."""
        clean = signal_gen.generate_clean(15.0)
        noisy = signal_gen.generate_noisy(clean, 0.1, rng)
        noise = noisy - clean
        snr = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
        assert 15 < snr < 25, f"Expected SNR ~20 dB, got {snr:.1f}"

    def test_zero_noise_returns_clean(self, signal_gen: SignalGenerator, rng) -> None:
        """SG-T5: σ=0 → noisy == clean exactly."""
        clean = signal_gen.generate_clean(5.0)
        noisy = signal_gen.generate_noisy(clean, 0.0, rng)
        np.testing.assert_array_equal(noisy, clean)

    def test_noise_mean_near_zero(self, signal_gen: SignalGenerator, rng) -> None:
        """SG-T14: Noise has zero mean (within statistical tolerance)."""
        clean = signal_gen.generate_clean(15.0)
        noisy = signal_gen.generate_noisy(clean, 0.1, rng)
        noise = noisy - clean
        tol = 3 * 0.1 * signal_gen.amplitude / np.sqrt(len(noise))
        assert abs(np.mean(noise)) < tol

    def test_noise_std_matches_config(self, rng) -> None:
        """SG-T15: Noise std matches σ·A within 15 % (statistical, N=2000)."""
        sigma = 0.1
        gen = SignalGenerator(sampling_rate=200, duration=10.0, amplitude=1.0)
        clean = gen.generate_clean(15.0)
        noisy = gen.generate_noisy(clean, sigma, rng)
        noise_std = np.std(noisy - clean)
        expected = sigma * gen.amplitude
        assert abs(noise_std - expected) / expected < 0.15

    def test_all_zero_noise_vector_tolerated(self, signal_gen: SignalGenerator, rng) -> None:
        """EC.12: Degenerate zero-noise Gaussian returns finite clean samples."""
        clean = signal_gen.generate_clean(15.0)
        noisy = signal_gen.generate_noisy(clean, 0.0, rng)
        np.testing.assert_array_equal(noisy, clean)
        assert np.isfinite(noisy).all()


class TestSignalGeneratorValidation:
    """Input validation edge cases."""

    def test_negative_noise_raises(self, signal_gen: SignalGenerator, rng) -> None:
        """SG-T11: Negative noise_std_ratio raises ValueError."""
        clean = signal_gen.generate_clean(10.0)
        with pytest.raises(ValueError, match="noise_std_ratio"):
            signal_gen.generate_noisy(clean, -0.1, rng)

    def test_nyquist_violation_raises(self) -> None:
        """SG-T12: Frequency exceeding Nyquist raises ValueError."""
        gen = SignalGenerator(sampling_rate=100, duration=1.0, amplitude=1.0)
        with pytest.raises(ValueError, match="Nyquist"):
            gen.validate_nyquist([60.0])

    def test_zero_duration_raises(self) -> None:
        """SG-T13: Zero or negative duration raises ValueError."""
        with pytest.raises(ValueError, match="duration"):
            SignalGenerator(sampling_rate=200, duration=0.0, amplitude=1.0)

    def test_negative_sampling_rate_raises(self) -> None:
        """Negative sampling rate raises ValueError."""
        with pytest.raises(ValueError, match="sampling_rate"):
            SignalGenerator(sampling_rate=-100, duration=1.0, amplitude=1.0)

    def test_negative_amplitude_raises(self) -> None:
        """Negative amplitude raises ValueError."""
        with pytest.raises(ValueError, match="amplitude"):
            SignalGenerator(sampling_rate=200, duration=1.0, amplitude=-1.0)

    def test_negative_frequency_raises(self, signal_gen: SignalGenerator) -> None:
        """SG-EC6: Negative frequency raises ValueError."""
        with pytest.raises(ValueError, match="Frequency"):
            signal_gen.generate_clean(-5.0)

    def test_zero_amplitude_allowed(self, rng) -> None:
        """EC.7: Zero amplitude is allowed and produces zero targets."""
        gen = SignalGenerator(sampling_rate=200, duration=1.0, amplitude=0.0)
        clean = gen.generate_clean(15.0)
        noisy = gen.generate_noisy(clean, 10.0, rng)
        assert np.all(clean == 0.0)
        assert np.all(noisy == 0.0)
