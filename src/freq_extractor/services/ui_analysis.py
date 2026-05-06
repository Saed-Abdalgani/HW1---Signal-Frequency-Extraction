"""Dash callbacks for analysis tabs (T-SNE, PCA, FFT)."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, no_update
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from freq_extractor.services.ui_analysis_prep import (
    MAX_TSNE_ROWS,
    TSNE_ITER,
    safe_tsne_perplexity,
    stack_window_features,
    subsample_feature_rows,
)
from freq_extractor.services.ui_layout import SIN_COLOURS
from freq_extractor.services.ui_signal_utils import (
    clamp_n_cycles,
    compute_shared_grid,
    f_min_from_sin_bundle,
    signal_on_grid,
)

_DARK = {
    "paper_bgcolor": "#0d1117", "plot_bgcolor": "#0d1117",
    "font": {"color": "#e6edf3"}, "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
}


def register_analysis_callbacks(app: Any, all_inputs: list) -> None:
    """Register T-SNE, PCA, and FFT callbacks."""

    @app.callback(
        Output("tsne-plot", "figure"),
        [Input("main-tabs", "value"), Input("tsne-perp", "value")] + all_inputs,
    )
    def update_tsne(tab, perp, fs, n_cyc, bw, display, noise_type, filt, *sin_vals):
        if tab != "tsne":
            return no_update
        window = 10
        features, labels, colours = stack_window_features(fs, n_cyc, sin_vals, window)
        fig = go.Figure(layout={**_DARK, "title": "T-SNE 3D Visualization"})
        if len(features) >= 4:
            feat_full = np.array(features)
            fb, lu, cu = subsample_feature_rows(feat_full, labels, colours, MAX_TSNE_ROWS)
            emb = TSNE(
                n_components=3,
                perplexity=safe_tsne_perplexity(perp, len(fb)),
                random_state=42,
                learning_rate="auto",
                max_iter=TSNE_ITER,
                early_exaggeration=8.0,
                n_jobs=-1,
            ).fit_transform(fb)
            fig.add_trace(go.Scatter3d(
                x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
                mode="markers", marker={"size": 4, "color": cu, "opacity": 0.8},
                text=lu, hoverinfo="text"))
            fig.update_layout(scene={
                "xaxis_title": "Dim 1", "yaxis_title": "Dim 2", "zaxis_title": "Dim 3"})
            if feat_full.shape[0] > MAX_TSNE_ROWS:
                txt = f"T-SNE 3D (preview; {fb.shape[0]:d} rows of {feat_full.shape[0]:d})"
                fig.update_layout(title={"text": txt})
        else:
            txt = ("T-SNE — need ≥4 MIX-enabled windows (turn MIX on, set frequency > 0)")
            fig.update_layout(title={"text": txt})
        return fig

    @app.callback(Output("pca-plot", "figure"), [Input("main-tabs", "value")] + all_inputs)
    def update_pca(tab, fs, n_cyc, bw, display, noise_type, filt, *sin_vals):
        if tab != "pca":
            return no_update
        window = 10
        features, labels, colours = stack_window_features(fs, n_cyc, sin_vals, window)
        fig = go.Figure(layout={**_DARK, "title": "PCA 3D Visualization"})
        if len(features) >= 3:
            feat = np.array(features)
            pca = PCA(n_components=min(3, feat.shape[0], feat.shape[1]))
            emb = pca.fit_transform(feat)
            evr = pca.explained_variance_ratio_ * 100
            fig.add_trace(go.Scatter3d(
                x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
                mode="markers", marker={"size": 4, "color": colours, "opacity": 0.8},
                text=labels, hoverinfo="text"))
            fig.update_layout(scene={
                "xaxis_title": f"PC1 ({evr[0]:.1f}%)",
                "yaxis_title": f"PC2 ({evr[1]:.1f}%)",
                "zaxis_title": f"PC3 ({evr[2]:.1f}%)"})
        return fig

    @app.callback(
        Output("fft-plot", "figure"),
        [Input("main-tabs", "value"), Input("fft-log", "value")] + all_inputs,
    )
    def update_fft(tab, log_scale, fs, n_cyc, bw, display, noise_type, filt, *sin_vals):
        if tab != "fft":
            return no_update
        fs = fs or 200
        n_cyc = clamp_n_cycles(n_cyc)
        t = compute_shared_grid(fs, n_cyc, f_min_from_sin_bundle(sin_vals))
        combined = None
        active_freqs = []
        for i in range(4):
            mix = sin_vals[i * 6]
            freq = sin_vals[i * 6 + 2]
            if not (mix and "mix" in mix and freq and freq > 0):
                continue
            phase = sin_vals[i * 6 + 3] or 0
            amp = sin_vals[i * 6 + 4] or 1
            sig = signal_on_grid(t, float(freq), float(phase), float(amp))
            bpf_val = sin_vals[i * 6 + 1] or []
            active_freqs.append((freq, SIN_COLOURS[i], bw or 5, "bpf" in bpf_val))
            combined = sig if combined is None else combined + sig
        fig = go.Figure(layout={**_DARK, "title": "FFT Magnitude Spectrum"})
        if combined is not None and len(combined) > 1:
            spectrum = np.abs(np.fft.rfft(combined))
            freqs_ax = np.fft.rfftfreq(len(combined), d=1 / fs)
            fig.add_trace(go.Scatter(
                x=freqs_ax, y=spectrum, mode="lines",
                line={"color": "#1f6feb"}, name="Magnitude"))
            for freq, colour, bw_val, has_bpf in active_freqs:
                fig.add_vline(x=freq, line={"color": colour, "dash": "dash", "width": 2})
                if has_bpf:
                    fig.add_vrect(x0=freq - bw_val / 2, x1=freq + bw_val / 2,
                        fillcolor=colour, opacity=0.15, line_width=0)
            if log_scale and "log" in log_scale:
                fig.update_xaxes(
                    type="log",
                    range=[np.log10(0.1), np.log10(10.0)],
                    title="Frequency (Hz)",
                )
            else:
                fig.update_xaxes(range=[0.0, 10.0], title="Frequency (Hz)")
            fig.update_yaxes(title="Magnitude")
        return fig
