"""Reusable Dash components for the Sinusoid Explorer.

Provides header metrics, global parameter controls, per-sinusoid
controls, and the four-tab panel.
"""

from __future__ import annotations

from typing import Any

from dash import dcc, html

SIN_COLOURS = ["#00d2ff", "#ff6b6b", "#51cf66", "#ffd43b"]


def header_metrics() -> list:
    """Build the live header metrics bar."""
    metrics = ["Fs", "N-CYC", "T", "N", "F_MIN"]
    items = []
    for i, m in enumerate(metrics):
        items.append(html.Span("---", id=f"metric-{m}", className="metric-text"))
        if i < len(metrics) - 1:
            items.append(html.Span("|", className="metric-divider"))
    return items


def global_controls(config: dict[str, Any]) -> html.Div:
    """Build the Global Parameters sidebar panel."""
    return html.Div([
        html.H4("GLOBAL PARAMETERS", className="sidebar-section-title"),

        html.Div([
            html.Label("Fs", className="slider-label"),
            html.Div(dcc.Slider(id="fs-slider", min=10, max=2000, step=10,
                value=100, marks=None, tooltip={"placement": "right", "always_visible": True}), className="slider-container"),
        ], className="slider-row"),

        html.Div([
            html.Label("N-cycles", className="slider-label"),
            html.Div(dcc.Input(id="n-cycles", type="number", value=1, min=1, max=20, className="styled-input"), className="slider-container"),
        ], className="slider-row"),

        html.Div([
            html.Label("BW", className="slider-label"),
            html.Div(dcc.Slider(id="bw-slider", min=0.1, max=100, step=0.1,
                value=1.0, marks=None, tooltip={"placement": "right", "always_visible": True}), className="slider-container"),
        ], className="slider-row"),

        html.Div([
            html.Label("Display", className="slider-label"),
            dcc.RadioItems(id="display-toggle",
                options=[{"label": "LINE", "value": "LINE"}, {"label": "DOTS", "value": "DOTS"}],
                value="DOTS", inline=True, className="segmented-control"),
        ], className="slider-row"),

        html.Div([
            html.Label("Noise", className="slider-label"),
            dcc.Dropdown(id="noise-dropdown",
                options=[{"label": n, "value": n} for n in ["None", "Gaussian", "Uniform"]],
                value="Gaussian", clearable=False, className="custom-dropdown"),
        ], className="slider-row"),

        html.Div([
            html.Label("Filter", className="slider-label"),
            dcc.Dropdown(id="filter-dropdown",
                options=[{"label": f, "value": f} for f in ["None", "Lowpass", "Highpass", "Bandpass"]],
                value="Bandpass", clearable=False, className="custom-dropdown"),
        ], className="slider-row"),

        html.Button("► SWEEP NOISE", id="sweep-btn", className="sweep-btn"),
        dcc.Interval(id="sweep-interval", interval=100, disabled=True, n_intervals=0),
    ], className="control-block")


def _sin_control(n: int) -> html.Div:
    """Build per-sinusoid control block for Sin *n* (1-indexed)."""
    c = SIN_COLOURS[n - 1]
    freqs = [0.8, 1.8, 8.0, 12.0]
    return html.Div([
        html.Div([
            html.Div(className="color-dot", style={"backgroundColor": c}),
            html.Div(f"Sin {n}", className="sin-name", style={"color": c}),
            dcc.Checklist(id=f"mix-{n}", options=[{"label": "MIX", "value": "mix"}],
                value=["mix"], inline=True, className="sin-checkbox"),
            dcc.Checklist(id=f"bpf-{n}", options=[{"label": "BPF", "value": "bpf"}],
                value=[] if n != 1 else ["bpf"], inline=True, className="sin-checkbox"),
        ], className="sin-header"),

        html.Div([
            html.Label("f", className="slider-label"),
            html.Div(dcc.Slider(id=f"freq-{n}", min=0.1, max=100, step=0.1, value=freqs[n - 1], marks=None, tooltip={"placement": "right", "always_visible": True}), className="slider-container"),
        ], className="slider-row"),

        html.Div([
            html.Label("φ", className="slider-label"),
            html.Div(dcc.Slider(id=f"phase-{n}", min=0, max=6.28, step=0.01, value=0.0 if n==1 else 0.93 if n==2 else 0, marks=None, tooltip={"placement": "right", "always_visible": True}), className="slider-container"),
        ], className="slider-row"),

        html.Div([
            html.Label("A", className="slider-label"),
            html.Div(dcc.Slider(id=f"amp-{n}", min=0, max=2.0, step=0.01, value=0.1 if n==1 else 0.8 if n==2 else 0.6, marks=None, tooltip={"placement": "right", "always_visible": True}), className="slider-container"),
        ], className="slider-row"),

        html.Div([
            html.Label("σ", className="slider-label"),
            html.Div(dcc.Slider(id=f"sigma-{n}", min=0, max=1.0, step=0.01, value=0.0, marks=None, tooltip={"placement": "right", "always_visible": True}), className="slider-container"),
        ], className="slider-row"),
    ], className=f"control-block sin-block-{n}")


def build_sin_controls() -> list[html.Div]:
    """Return list of 4 sinusoid control blocks."""
    return [_sin_control(i) for i in range(1, 5)]


def build_tabs() -> html.Div:
    """Build the four visualisation tabs."""
    return html.Div([
        dcc.Tabs(id="main-tabs", value="signals", className="custom-tabs-container", children=[
            dcc.Tab(label="SIGNALS", value="signals", className="custom-tab", selected_className="custom-tab--selected", children=[
                html.Div([
                    dcc.Graph(id="individual-plot", config={"displayModeBar": True}),
                    dcc.Graph(id="combined-plot", config={"displayModeBar": True}),
                ], className="tab-content")
            ]),
            dcc.Tab(label="T-SNE 3D", value="tsne", className="custom-tab", selected_className="custom-tab--selected", children=[
                html.Div([
                    html.Div([html.Label("Perplexity", className="slider-label"), dcc.Slider(id="tsne-perp", min=5, max=50,
                        step=1, value=30, marks=None, tooltip={"placement": "bottom", "always_visible": True})], style={"padding": "8px"}),
                    dcc.Graph(id="tsne-plot"),
                ], className="tab-content")
            ]),
            dcc.Tab(label="PCA 3D", value="pca", className="custom-tab", selected_className="custom-tab--selected", children=[
                html.Div([dcc.Graph(id="pca-plot")], className="tab-content")
            ]),
            dcc.Tab(label="FFT SPECTRUM", value="fft", className="custom-tab", selected_className="custom-tab--selected", children=[
                html.Div([
                    html.Div([dcc.Checklist(id="fft-log",
                        options=[{"label": "Log Scale", "value": "log"}],
                        value=[], inline=True, className="sin-checkbox")], style={"padding": "8px"}),
                    dcc.Graph(id="fft-plot"),
                ], className="tab-content")
            ]),
        ])
    ], className="main-content")
