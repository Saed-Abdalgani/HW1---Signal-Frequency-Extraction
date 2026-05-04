"""Dash callbacks for the Sinusoid Explorer — signals and controls.

Registers header metrics, SIGNALS tab, and SWEEP NOISE callbacks.
Analysis tab callbacks (T-SNE, PCA, FFT) are in ``ui_analysis.py``.
Signal generation/processing utilities are in ``ui_signal_utils.py``.

References
----------
- PRD FR-9
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State

from freq_extractor.services.ui_analysis import register_analysis_callbacks
from freq_extractor.services.ui_layout import SIN_COLOURS
from freq_extractor.services.ui_signal_utils import (
    add_noise,
    apply_bpf,
    gen_signal,
)

_DARK_LAYOUT = {
    "paper_bgcolor": "#0d1117", "plot_bgcolor": "#0d1117",
    "font": {"color": "#e6edf3", "family": "Inter"},
    "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
    "xaxis": {"gridcolor": "#1e2632", "zerolinecolor": "#1e2632"},
    "yaxis": {"gridcolor": "#1e2632", "zerolinecolor": "#1e2632"},
}


def _build_sin_inputs() -> list:
    """Build input list for 4 sinusoids (mix, bpf, freq, phase, amp, sigma)."""
    sin_inputs = []
    for i in range(1, 5):
        sin_inputs += [
            Input(f"mix-{i}", "value"), Input(f"bpf-{i}", "value"),
            Input(f"freq-{i}", "value"), Input(f"phase-{i}", "value"),
            Input(f"amp-{i}", "value"), Input(f"sigma-{i}", "value"),
        ]
    return sin_inputs


def register_callbacks(app: Any, config: dict[str, Any]) -> None:
    """Register all Dash callbacks on *app*."""
    sin_inputs = _build_sin_inputs()
    all_inputs = ([Input("fs-slider", "value"), Input("n-cycles", "value"),
                   Input("bw-slider", "value"), Input("display-toggle", "value"),
                   Input("noise-dropdown", "value"), Input("filter-dropdown", "value")]
                  + sin_inputs)

    @app.callback(
        [Output(f"metric-{m}", "children") for m in ["Fs", "N-CYC", "T", "F_MIN", "N", "NYQ"]],
        [Input("fs-slider", "value"), Input("n-cycles", "value")]
        + [Input(f"freq-{i}", "value") for i in range(1, 5)],
    )
    def update_metrics(fs, n_cyc, *freqs):
        fs = fs or 200
        n_cyc = n_cyc or 2
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

    @app.callback(
        [Output("individual-plot", "figure"), Output("combined-plot", "figure")],
        all_inputs,
    )
    def update_signals(fs, n_cyc, bw, display, noise_type, filt, *sin_vals):
        fs, n_cyc = fs or 200, n_cyc or 2
        mode = "markers" if display == "DOTS" else "lines"
        rng = np.random.default_rng(0)

        ind_fig = go.Figure(layout={**_DARK_LAYOUT, "title": {"text": "Individual Sinusoids", "font": {"color": "#e6edf3"}}})
        combined, t_ref = None, None

        for i in range(4):
            mix, bpf_v, freq, phase, amp, sigma = sin_vals[i * 6:(i + 1) * 6]
            if not freq or freq <= 0:
                continue
            t, sig = gen_signal(fs, n_cyc, freq, phase or 0, amp or 1)
            if sigma and sigma > 0:
                sig = sig + rng.normal(0, sigma, size=sig.shape)
            if bpf_v and "bpf" in bpf_v:
                sig = apply_bpf(sig, freq, bw or 5, fs)

            ind_fig.add_trace(go.Scatter(x=t, y=sig, mode=mode, name=f"Sin {i + 1}",
                visible=True if (mix and "mix" in mix) else "legendonly",
                line={"color": SIN_COLOURS[i]}, marker={"color": SIN_COLOURS[i]}))

            if mix and "mix" in mix:
                if combined is None:
                    combined, t_ref = np.zeros_like(sig), t
                elif len(sig) != len(combined):
                    ml = min(len(sig), len(combined))
                    sig, combined, t_ref = sig[:ml], combined[:ml], t_ref[:ml]
                combined += sig

        comb_title = "Combined Signal (Noisy)" if noise_type and noise_type != "None" else "Combined Signal (Clean)"
        comb_fig = go.Figure(layout={**_DARK_LAYOUT, "title": {"text": comb_title, "font": {"color": "#e6edf3"}}})
        if combined is not None:
            combined = add_noise(combined, noise_type or "None", rng)
            comb_fig.add_trace(go.Scatter(x=t_ref, y=combined, mode=mode,
                name="Noisy Mixed" if noise_type and noise_type != "None" else "Clean Mixed",
                line={"color": "#ffffff"}))

        ind_fig.update_layout(xaxis_title="t (s)", hovermode="x unified")
        comb_fig.update_layout(xaxis_title="t (s)", hovermode="closest")
        return ind_fig, comb_fig

    @app.callback(
        Output("sweep-interval", "disabled"),
        Input("sweep-btn", "n_clicks"),
        State("sweep-interval", "disabled"),
        prevent_initial_call=True,
    )
    def toggle_sweep(n_clicks, is_disabled):
        return not is_disabled

    register_analysis_callbacks(app, all_inputs)
