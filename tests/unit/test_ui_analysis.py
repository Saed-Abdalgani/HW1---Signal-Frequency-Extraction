"""Tests for UI analysis callbacks and shared signal grids."""

from __future__ import annotations

from freq_extractor.services.ui_analysis import register_analysis_callbacks
from freq_extractor.services.ui_signal_utils import (
    compute_shared_grid,
    gen_signal,
    signal_on_grid,
)


class TestUiSharedSignals:
    """Shared grid helpers."""

    def test_gen_signal_length(self) -> None:
        """gen_signal returns aligned arrays."""
        t, sig = gen_signal(200.0, 2.0, 4.0, 0.0, 1.0)
        assert len(t) == len(sig)
        assert len(t) > 0

    def test_signal_on_grid_matches_length(self) -> None:
        """signal_on_grid follows *t* length."""
        t = compute_shared_grid(200, 3, 2.0)
        sig = signal_on_grid(t, 4.0, 0.0, 1.0)
        assert len(sig) == len(t)

    def test_negative_freq_on_grid_still_finite(self) -> None:
        """Fallback frequency keeps samples finite."""
        t = compute_shared_grid(200, 2, 1.0)
        sig = signal_on_grid(t, -5.0, 0.0, 1.0)
        assert len(sig) > 0


class FakeApp:
    """Mock Dash app."""

    def __init__(self) -> None:
        self.callbacks: dict = {}

    def callback(self, *args, **kwargs):
        def decorator(func):
            self.callbacks[func.__name__] = func
            return func
        return decorator


class TestUIAnalysisCallbacks:
    """Analysis tab callbacks."""

    def test_update_tsne(self) -> None:
        """TSNE figure builds."""
        app = FakeApp()
        register_analysis_callbacks(app, [])
        sin_vals = [
            ["mix"], [], 3.0, 0.0, 1.0, 0.1,
            [], [], 0.0, 0.0, 0.0, 0.0,
            ["mix"], [], 4.0, 0.0, 0.5, 0.0,
            [], [], 0.0, 0.0, 0.0, 0.0,
        ]
        fig = app.callbacks["update_tsne"](
            "tsne", 5, 200, 3, 5, "LINE", "None", "None", *sin_vals)
        assert fig is not None

    def test_update_pca(self) -> None:
        """PCA figure builds."""
        app = FakeApp()
        register_analysis_callbacks(app, [])
        sin_vals = [
            ["mix"], [], 2.0, 0.0, 1.0, 0.1,
            ["mix"], [], 3.0, 0.0, 0.0, 0.0,
            ["mix"], [], 4.0, 0.0, 0.5, 0.0,
            [], [], 0.0, 0.0, 0.0, 0.0,
        ]
        fig = app.callbacks["update_pca"](
            "pca", 200, 3, 5, "LINE", "None", "None", *sin_vals)
        assert fig is not None

    def test_update_fft(self) -> None:
        """FFT figure builds."""
        app = FakeApp()
        register_analysis_callbacks(app, [])
        sin_vals = [
            ["mix"], ["bpf"], 2.0, 0.0, 1.0, 0.1,
            [], [], 0.0, 0.0, 0.0, 0.0,
            ["mix"], [], 4.0, 0.0, 0.5, 0.0,
            [], [], 0.0, 0.0, 0.0, 0.0,
        ]
        fig = app.callbacks["update_fft"](
            "fft", ["log"], 200, 3, 5, "LINE", "None", "None", *sin_vals)
        assert fig is not None
