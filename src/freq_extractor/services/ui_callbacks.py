"""Dash callbacks for the Sinusoid Explorer — signals and controls."""

from __future__ import annotations

from typing import Any

from dash import Input, Output

from freq_extractor.services.ui_analysis import register_analysis_callbacks
from freq_extractor.services.ui_callback_inputs import (
    analysis_inputs,
    build_sin_inputs,
    signals_callback_inputs,
)
from freq_extractor.services.ui_callbacks_plots import build_signal_figures
from freq_extractor.services.ui_signal_utils import clamp_n_cycles


def register_callbacks(app: Any, config: dict[str, Any]) -> None:
    """Register all Dash callbacks on *app*."""
    _ = config
    sin_inputs = build_sin_inputs()
    a_in = analysis_inputs(sin_inputs)
    s_in = signals_callback_inputs(sin_inputs)

    @app.callback(
        [Output(f"metric-{m}", "children") for m in ["Fs", "N-CYC", "T", "F_MIN", "N", "NYQ"]],
        [Input("fs-slider", "value"), Input("n-cycles", "value")]
        + [Input(f"freq-{i}", "value") for i in range(1, 5)],
    )
    def update_metrics(fs, n_cyc, *freqs):
        fs = fs or 200
        n_cyc = clamp_n_cycles(n_cyc)
        f_min = min(f for f in freqs if f and f > 0) if any(f and f > 0 for f in freqs) else 1
        dur = n_cyc / f_min
        n_samp = int(fs * dur)
        return (f" {fs} Hz", f" {n_cyc}", f" {dur:.2f}s",
                f" {f_min:.1f} Hz", f" {n_samp}", f" {fs / 2:.1f} Hz")

    for i in range(1, 5):
        @app.callback(
            [Output(f"f-val-{i}", "children"), Output(f"phi-val-{i}", "children"),
             Output(f"a-val-{i}", "children"), Output(f"sig-val-{i}", "children")],
            [Input(f"freq-{i}", "value"), Input(f"phase-{i}", "value"),
             Input(f"amp-{i}", "value"), Input(f"sigma-{i}", "value")],
        )
        def update_sin_vals(freq, phase, amp, sigma, _i=i):
            return (f"{freq or 0:.1f} Hz", f"{phase or 0:.2f} rad",
                    f"{amp or 0:.2f}", f"{sigma or 0:.2f}")

    @app.callback([Output("individual-plot", "figure"), Output("combined-plot", "figure")], s_in)
    def update_signals(fs, n_cyc, bw, display, noise_type, filt, *sin_vals):
        return build_signal_figures(
            fs, n_cyc, bw, display, noise_type, filt, tuple(sin_vals),
        )

    @app.callback(
        [Output(f"sigma-{n}", "value") for n in range(1, 5)],
        Input("sweep-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def sweep_zero_sigmas(_n_clicks):
        """SWEEP NOISE: set each per-sin σ slider to zero."""
        return (0.0, 0.0, 0.0, 0.0)

    register_analysis_callbacks(app, a_in)
