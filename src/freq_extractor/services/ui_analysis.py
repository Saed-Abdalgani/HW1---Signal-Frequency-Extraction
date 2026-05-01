"""Dash callbacks for analysis tabs (T-SNE, PCA, FFT).
Separated from ``ui_callbacks.py`` to comply with ≤ 145 code-line limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from freq_extractor.services.ui_layout import SIN_COLOURS

_DARK = {
    "paper_bgcolor": "#0d1117", "plot_bgcolor": "#161b22",
    "font": {"color": "#e6edf3"}, "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
}


def _gen_sig(fs: float, n_cyc: float, freq: float, phase: float, amp: float) -> tuple:
    """Generate time array and sinusoidal signal."""
    if freq <= 0:
        freq = 0.1
    dur = n_cyc / freq
    n = max(int(fs * dur), 2)
    t = np.linspace(0, dur, n, endpoint=False)
    return t, amp * np.sin(2 * np.pi * freq * t + phase)


def register_analysis_callbacks(app: Any, all_inputs: list) -> None:
    """Register T-SNE, PCA, and FFT callbacks."""

    @app.callback(Output("tsne-plot", "figure"),
                  [Input("tsne-perp", "value")] + all_inputs)
    def update_tsne(perp, fs, n_cyc, bw, display, noise_type, filt, *sin_vals):
        fs = fs or 200
        n_cyc = n_cyc or 2
        window = 10
        features, labels, colours = [], [], []
        for i in range(4):
            mix = sin_vals[i * 6]
            freq = sin_vals[i * 6 + 2]
            if not (mix and "mix" in mix and freq and freq > 0):
                continue
            phase = sin_vals[i * 6 + 3] or 0
            amp = sin_vals[i * 6 + 4] or 1
            _, sig = _gen_sig(fs, n_cyc, freq, phase, amp)
            for j in range(len(sig) - window):
                features.append(sig[j:j + window])
                labels.append(f"Sin {i + 1}")
                colours.append(SIN_COLOURS[i])
        fig = go.Figure(layout={**_DARK, "title": "T-SNE 3D"})
        if len(features) >= 4:
            feat = np.array(features)
            perp = min(perp or 30, len(feat) - 1)
            emb = TSNE(n_components=3, perplexity=max(perp, 2), random_state=42).fit_transform(feat)
            fig.add_trace(go.Scatter3d(x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
                mode="markers", marker={"size": 3, "color": colours, "opacity": 0.7},
                text=labels, hoverinfo="text"))
        return fig

    @app.callback(Output("pca-plot", "figure"), all_inputs)
    def update_pca(fs, n_cyc, bw, display, noise_type, filt, *sin_vals):
        fs = fs or 200
        n_cyc = n_cyc or 2
        window = 10
        features, labels, colours = [], [], []
        for i in range(4):
            mix = sin_vals[i * 6]
            freq = sin_vals[i * 6 + 2]
            if not (mix and "mix" in mix and freq and freq > 0):
                continue
            phase = sin_vals[i * 6 + 3] or 0
            amp = sin_vals[i * 6 + 4] or 1
            _, sig = _gen_sig(fs, n_cyc, freq, phase, amp)
            for j in range(len(sig) - window):
                features.append(sig[j:j + window])
                labels.append(f"Sin {i + 1}")
                colours.append(SIN_COLOURS[i])
        fig = go.Figure(layout={**_DARK, "title": "PCA 3D"})
        if len(features) >= 3:
            feat = np.array(features)
            pca = PCA(n_components=3)
            emb = pca.fit_transform(feat)
            evr = pca.explained_variance_ratio_ * 100
            fig.add_trace(go.Scatter3d(x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
                mode="markers", marker={"size": 3, "color": colours, "opacity": 0.7},
                text=labels, hoverinfo="text"))
            fig.update_layout(scene={
                "xaxis_title": f"PC1 ({evr[0]:.1f} %)",
                "yaxis_title": f"PC2 ({evr[1]:.1f} %)",
                "zaxis_title": f"PC3 ({evr[2]:.1f} %)"})
        return fig

    @app.callback(Output("fft-plot", "figure"),
                  [Input("fft-log", "value")] + all_inputs)
    def update_fft(log_scale, fs, n_cyc, bw, display, noise_type, filt, *sin_vals):
        fs = fs or 200
        n_cyc = n_cyc or 2
        combined = None
        active_freqs = []
        for i in range(4):
            mix = sin_vals[i * 6]
            freq = sin_vals[i * 6 + 2]
            if not (mix and "mix" in mix and freq and freq > 0):
                continue
            phase = sin_vals[i * 6 + 3] or 0
            amp = sin_vals[i * 6 + 4] or 1
            _, sig = _gen_sig(fs, n_cyc, freq, phase, amp)
            bpf_val = sin_vals[i * 6 + 1] or []
            active_freqs.append((freq, SIN_COLOURS[i], bw or 5, "bpf" in bpf_val))
            if combined is None:
                combined = np.zeros_like(sig)
            ml = min(len(sig), len(combined))
            combined = combined[:ml] + sig[:ml]
        fig = go.Figure(layout={**_DARK, "title": "FFT Spectrum"})
        if combined is not None and len(combined) > 1:
            spectrum = np.abs(np.fft.rfft(combined))
            freqs_ax = np.fft.rfftfreq(len(combined), d=1 / fs)
            fig.add_trace(go.Scatter(x=freqs_ax, y=spectrum, mode="lines",
                line={"color": "#58a6ff"}, name="Magnitude"))
            for freq, colour, bw_val, has_bpf in active_freqs:
                fig.add_vline(x=freq, line={"color": colour, "dash": "dash", "width": 1.5})
                if has_bpf:
                    fig.add_vrect(x0=freq - bw_val / 2, x1=freq + bw_val / 2,
                        fillcolor=colour, opacity=0.1, line_width=0)
            if log_scale and "log" in log_scale:
                fig.update_xaxes(type="log")
            fig.update_xaxes(title="Frequency (Hz)")
            fig.update_yaxes(title="Magnitude")
        return fig
