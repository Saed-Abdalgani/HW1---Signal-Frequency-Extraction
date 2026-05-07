"""Tests for UI callback helpers — signal generation, noise, BPF.

Validates the internal helper functions used by Dash callbacks.
"""

from __future__ import annotations

import numpy as np

from freq_extractor.services.ui_callbacks import _add_noise, _apply_bpf, _apply_filter, _gen_signal


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

    def test_filter_dropdown_modes_return_same_length(self) -> None:
        """Filter dropdown modes all produce a signal with stable length."""
        sig = np.sin(np.linspace(0, 10, 1000))
        for mode in ["None", "Bandpass", "Lowpass", "Highpass"]:
            result = _apply_filter(sig, 10.0, 5.0, 200.0, mode)
            assert len(result) == len(sig)
class FakeApp:
    """Mock Dash app to capture registered callbacks."""
    def __init__(self):
        self.callbacks = {}

    def callback(self, *args, **kwargs):
        def decorator(func):
            self.callbacks[func.__name__] = func
            return func
        return decorator

class TestUICallbacks:
    """Dash callback logic tests."""

    def test_update_metrics(self, sample_config) -> None:
        """update_metrics returns correctly formatted strings."""
        from freq_extractor.services.ui_callbacks import register_callbacks
        app = FakeApp()
        register_callbacks(app, sample_config)
        res = app.callbacks["update_metrics"](200, 2, 5, 0, 0, 0)
        assert res[0] == " 200 Hz"
        assert res[1] == " 2"
        assert res[3] == " 2.50 Hz"
        assert res[5] == " 100.0 Hz"

    def test_constrain_freq_ranges(self, sample_config) -> None:
        """6K.13: Frequency sliders clamp to the current Nyquist value."""
        from freq_extractor.services.ui_callbacks import register_callbacks
        app = FakeApp()
        register_callbacks(app, sample_config)
        res = app.callbacks["constrain_freq_ranges"](80, 5, 30, 50, 100)
        assert res == [40, 40, 40, 40, 5, 30, 40, 40]

    def test_update_sin_vals(self, sample_config) -> None:
        """update_sin_vals formats sinusoid parameters."""
        from freq_extractor.services.ui_callbacks import register_callbacks
        app = FakeApp()
        register_callbacks(app, sample_config)
        res = app.callbacks["update_sin_vals"](5.0, 0.5, 1.0, 0.1)
        assert res == ("5.0 Hz", "0.50 rad", "1.00", "0.10")

    def test_update_signals(self, sample_config) -> None:
        """update_signals returns individual and combined plots."""
        from freq_extractor.services.ui_callbacks import register_callbacks
        app = FakeApp()
        register_callbacks(app, sample_config)
        sin_vals = [
            ["mix"], ["bpf"], 5.0, 0.0, 1.0, 0.1,
            [], [], 0.0, 0.0, 0.0, 0.0,
            ["mix"], [], 30.0, 0.0, 0.5, 0.0,
            [], [], 0.0, 0.0, 0.0, 0.0,
        ]
        ind, comb = app.callbacks["update_signals"](200, 2, 5, "DOTS", "Gaussian", "Bandpass", *sin_vals)
        assert ind is not None
        assert comb is not None

    def test_toggle_sweep(self, sample_config) -> None:
        """toggle_sweep inverts the disabled state."""
        from freq_extractor.services.ui_callbacks import register_callbacks
        app = FakeApp()
        register_callbacks(app, sample_config)
        assert app.callbacks["toggle_sweep"](1, True) is False
        assert app.callbacks["toggle_sweep"](1, False) is True

    def test_sweep_noise_changes_sigma(self, sample_config) -> None:
        """6K.12: Sweep interval drives sigma through 0->1->0."""
        from freq_extractor.services.ui_callbacks import register_callbacks
        app = FakeApp()
        register_callbacks(app, sample_config)
        assert app.callbacks["sweep_noise"](0, False) == [0.0, 0.0, 0.0, 0.0]
        assert app.callbacks["sweep_noise"](20, False) == [1.0, 1.0, 1.0, 1.0]
        assert app.callbacks["sweep_noise"](40, False) == [0.0, 0.0, 0.0, 0.0]
