"""Plotly figure builders for SIGNALS tab Dash callbacks."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from freq_extractor.services.ui_layout import SIN_COLOURS
from freq_extractor.services.ui_signal_utils import (
    add_noise,
    apply_bpf,
    apply_filter,
    clamp_n_cycles,
    compute_shared_grid,
    f_min_from_sin_bundle,
    signal_on_grid,
    weighted_mean_freq,
)

def _pristine_symmetric_context(
    sin_vals: tuple[Any, ...], noise_type: str | None, filt: str | None,
) -> bool:
    """True when traces have no stochastic noise / dropdown filter / BPF to preserve symmetry."""
    if (noise_type or "None") != "None":
        return False
    if (filt or "None") != "None":
        return False
    for i in range(4):
        b = i * 6
        freq = sin_vals[b + 2]
        if not freq or freq <= 0:
            continue
        bpf_val = sin_vals[b + 1] or []
        if "bpf" in bpf_val:
            return False
        sigma = sin_vals[b + 5]
        if sigma and float(sigma) > 0:
            return False
    return True


_DARK_LAYOUT = {
    "paper_bgcolor": "#0d1117", "plot_bgcolor": "#0d1117",
    "font": {"color": "#e6edf3", "family": "Inter"},
    "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
    "xaxis": {"gridcolor": "#1e2632", "zerolinecolor": "#1e2632"},
    "yaxis": {"gridcolor": "#1e2632", "zerolinecolor": "#1e2632"},
}


def build_signal_figures(
    fs: float | None,
    n_cyc: float | int | None,
    bw: float | None,
    display: str | None,
    noise_type: str | None,
    filt: str | None,
    sin_vals: tuple[Any, ...],
) -> tuple[go.Figure, go.Figure]:
    """Build individual and combined sinusoid figures.

    Returns
    -------
    tuple[plotly.graph_objects.Figure, plotly.graph_objects.Figure]
        ``(individual_figure, combined_figure)``.
    """
    fs = fs or 200
    n_cyc = clamp_n_cycles(n_cyc)
    mode = "markers" if display == "DOTS" else "lines"
    rng = np.random.default_rng(0)
    f_min = f_min_from_sin_bundle(sin_vals)
    t = compute_shared_grid(fs, n_cyc, f_min)
    comb_center = weighted_mean_freq(sin_vals)
    pristine_sym = _pristine_symmetric_context(sin_vals, noise_type or "None", filt)

    ind_fig = go.Figure(layout={
        **_DARK_LAYOUT,
        "title": {"text": "Individual Sinusoids", "font": {"color": "#e6edf3"}},
    })
    combined = None

    for i in range(4):
        mix, bpf_v, freq, phase, amp, sigma = sin_vals[i * 6:(i + 1) * 6]
        if not freq or freq <= 0:
            continue
        sig = signal_on_grid(t, float(freq), float(phase or 0), float(amp or 1))
        if sigma and float(sigma) > 0:
            sig = sig + rng.normal(0, float(sigma), size=sig.shape)
        if not pristine_sym:
            sig = apply_filter(sig, filt or "None", fs, float(freq), bw or 5)
            if bpf_v and "bpf" in bpf_v:
                sig = apply_bpf(sig, float(freq), bw or 5, fs)

        ind_fig.add_trace(go.Scatter(
            x=t, y=sig, mode=mode, name=f"Sin {i + 1}",
            visible=True if (mix and "mix" in mix) else "legendonly",
            line={"color": SIN_COLOURS[i]}, marker={"color": SIN_COLOURS[i]},
        ))

        if mix and "mix" in mix:
            combined = sig if combined is None else combined + sig

    comb_title = (
        "Combined Signal (Noisy)" if noise_type and noise_type != "None" else "Combined Signal (Clean)"
    )
    comb_fig = go.Figure(layout={
        **_DARK_LAYOUT,
        "title": {"text": comb_title, "font": {"color": "#e6edf3"}},
    })
    if combined is not None:
        if not pristine_sym:
            combined = apply_filter(combined, filt or "None", fs, comb_center, bw or 5)
        combined = add_noise(combined, noise_type or "None", rng)
        comb_fig.add_trace(go.Scatter(
            x=t, y=combined, mode=mode,
            name="Noisy Mixed" if noise_type and noise_type != "None" else "Clean Mixed",
            line={"color": "#ffffff"},
        ))

    ind_fig.update_layout(xaxis_title="t (s)", hovermode="x unified")
    comb_fig.update_layout(xaxis_title="t (s)", hovermode="closest")
    return ind_fig, comb_fig
