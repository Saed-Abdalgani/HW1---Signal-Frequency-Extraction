"""Tests for Dash Sinusoid Explorer callback registration."""

from __future__ import annotations

from freq_extractor.services.ui_callbacks import register_callbacks


class FakeApp:
    """Mock Dash app to capture registered callbacks."""

    def __init__(self) -> None:
        self.callbacks: dict = {}

    def callback(self, *args, **kwargs):
        def decorator(func):
            self.callbacks[func.__name__] = func
            return func
        return decorator


class TestUICallbacks:
    """Dash callback wiring."""

    def test_update_metrics(self, sample_config) -> None:
        """update_metrics formats header strings."""
        app = FakeApp()
        register_callbacks(app, sample_config)
        res = app.callbacks["update_metrics"](200, 2, 3.0, 0, 0, 0)
        assert res[0] == " 200 Hz"
        assert res[1] == " 2"

    def test_update_sin_vals(self, sample_config) -> None:
        """update_sin_vals formats readouts."""
        app = FakeApp()
        register_callbacks(app, sample_config)
        res = app.callbacks["update_sin_vals"](3.0, 0.5, 1.0, 0.1)
        assert res == ("3.0 Hz", "0.50 rad", "1.00", "0.10")

    def test_update_signals(self, sample_config) -> None:
        """update_signals returns two figures."""
        app = FakeApp()
        register_callbacks(app, sample_config)
        sin_vals = [
            ["mix"], ["bpf"], 3.0, 0.0, 1.0, 0.1,
            [], [], 0.0, 0.0, 0.0, 0.0,
            ["mix"], [], 4.0, 0.0, 0.5, 0.0,
            [], [], 0.0, 0.0, 0.0, 0.0,
        ]
        ind, comb = app.callbacks["update_signals"](
            200, 3, 5, "DOTS", "Gaussian", "Bandpass", *sin_vals,
        )
        assert ind is not None
        assert comb is not None

    def test_sweep_zero_sigmas(self, sample_config) -> None:
        """SWEEP NOISE clears all four per-sin sigma sliders."""
        app = FakeApp()
        register_callbacks(app, sample_config)
        assert app.callbacks["sweep_zero_sigmas"](1) == (0.0, 0.0, 0.0, 0.0)
