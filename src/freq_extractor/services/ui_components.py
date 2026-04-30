"""Reusable Dash components for the Sinusoid Explorer.

Provides header metrics, global parameter controls, per-sinusoid
controls, and the four-tab panel.

References
----------
- PRD FR-9
"""

from __future__ import annotations

from typing import Any

from dash import dcc, html

SIN_COLOURS = ["#00d2ff", "#ff6b6b", "#51cf66", "#ffd43b"]


def header_metrics() -> html.Div:
    """Build the live header metrics bar."""
    metrics = ["Fs", "N-CYC", "T", "h", "F_MIN"]
    return html.Div(
        [html.Span(id=f"metric-{m}", style={"margin": "0 18px", "fontSize": "14px",
            "fontWeight": "600"}) for m in metrics],
        style={"display": "flex", "alignItems": "center", "justifyContent": "center",
            "backgroundColor": "#161b22", "padding": "10px 0", "borderRadius": "8px",
            "marginBottom": "12px", "border": "1px solid #30363d"},
    )


def global_controls(config: dict[str, Any]) -> html.Div:
    """Build the Global Parameters sidebar panel."""
    sig = config.get("signal", {})
    return html.Div([
        html.H4("Global Parameters", style={"color": "#58a6ff", "marginBottom": "12px"}),
        html.Label("Fs (Hz)"), dcc.Slider(id="fs-slider", min=10, max=2000, step=10,
            value=sig.get("sampling_rate_hz", 200), marks=None,
            tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("N-cycles"), dcc.Input(id="n-cycles", type="number", value=2,
            min=1, max=20, style={"width": "80px", "backgroundColor": "#21262d",
            "color": "#e6edf3", "border": "1px solid #30363d", "borderRadius": "4px"}),
        html.Label("BW (Hz)"), dcc.Slider(id="bw-slider", min=0.1, max=100, step=0.1,
            value=5.0, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("Display"), dcc.RadioItems(id="display-toggle",
            options=[{"label": "LINE", "value": "LINE"}, {"label": "DOTS", "value": "DOTS"}],
            value="LINE", inline=True, style={"marginBottom": "8px"}),
        html.Label("Noise"), dcc.Dropdown(id="noise-dropdown",
            options=[{"label": n, "value": n} for n in ["None", "Gaussian", "Uniform"]],
            value="None", clearable=False,
            style={"backgroundColor": "#21262d", "color": "#0d1117"}),
        html.Label("Filter"), dcc.Dropdown(id="filter-dropdown",
            options=[{"label": f, "value": f} for f in ["None", "Lowpass", "Highpass", "Bandpass"]],
            value="None", clearable=False,
            style={"backgroundColor": "#21262d", "color": "#0d1117"}),
        html.Button("SWEEP NOISE", id="sweep-btn",
            style={"marginTop": "12px", "width": "100%", "padding": "8px",
                "backgroundColor": "#238636", "color": "#fff", "border": "none",
                "borderRadius": "6px", "cursor": "pointer", "fontWeight": "bold"}),
        dcc.Interval(id="sweep-interval", interval=100, disabled=True, n_intervals=0),
    ], style={"padding": "12px", "backgroundColor": "#161b22", "borderRadius": "8px",
        "marginBottom": "12px", "border": "1px solid #30363d"})


def _sin_control(n: int) -> html.Div:
    """Build per-sinusoid control block for Sin *n* (1-indexed)."""
    c = SIN_COLOURS[n - 1]
    freqs = [5, 15, 30, 50]
    return html.Div([
        html.Div([
            html.Span("*", style={"color": c, "fontSize": "18px", "marginRight": "8px"}),
            html.Strong(f"Sin {n}", style={"marginRight": "12px"}),
            dcc.Checklist(id=f"mix-{n}", options=[{"label": "MIX", "value": "mix"}],
                value=["mix"], inline=True, style={"display": "inline"}),
            dcc.Checklist(id=f"bpf-{n}", options=[{"label": "BPF", "value": "bpf"}],
                value=[], inline=True, style={"display": "inline", "marginLeft": "8px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        html.Label("f (Hz)"), dcc.Slider(id=f"freq-{n}", min=0.1, max=100,
            step=0.1, value=freqs[n - 1], marks=None,
            tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("phase (rad)"), dcc.Slider(id=f"phase-{n}", min=0, max=6.283,
            step=0.01, value=0, marks=None,
            tooltip={"placement": "bottom", "always_visible": True}),
        html.Label("A"), dcc.Slider(id=f"amp-{n}", min=0, max=2.0,
            step=0.01, value=1.0, marks=None,
            tooltip={"placement": "bottom", "always_visible": True}),
    ], style={"padding": "10px", "backgroundColor": "#161b22", "borderRadius": "8px",
        "marginBottom": "8px", "border": f"1px solid {c}33"})


def build_sin_controls() -> list[html.Div]:
    """Return list of 4 sinusoid control blocks."""
    return [_sin_control(i) for i in range(1, 5)]


def build_tabs() -> dcc.Tabs:
    """Build the four visualisation tabs."""
    ts = {"backgroundColor": "#161b22", "color": "#8b949e", "border": "none",
        "padding": "10px 20px", "fontWeight": "600"}
    sel = {**ts, "color": "#58a6ff", "borderBottom": "2px solid #58a6ff"}
    return dcc.Tabs(id="main-tabs", value="signals", children=[
        dcc.Tab(label="SIGNALS", value="signals", style=ts, selected_style=sel, children=[
            dcc.Graph(id="individual-plot", config={"displayModeBar": True}),
            dcc.Graph(id="combined-plot", config={"displayModeBar": True}),
        ]),
        dcc.Tab(label="T-SNE 3D", value="tsne", style=ts, selected_style=sel, children=[
            html.Div([html.Label("Perplexity"), dcc.Slider(id="tsne-perp", min=5, max=50,
                step=1, value=30, marks=None,
                tooltip={"placement": "bottom", "always_visible": True})], style={"padding": "8px"}),
            dcc.Graph(id="tsne-plot"),
        ]),
        dcc.Tab(label="PCA 3D", value="pca", style=ts, selected_style=sel,
            children=[dcc.Graph(id="pca-plot")]),
        dcc.Tab(label="FFT SPECTRUM", value="fft", style=ts, selected_style=sel, children=[
            html.Div([dcc.Checklist(id="fft-log",
                options=[{"label": "Log Scale", "value": "log"}],
                value=[], inline=True, style={"padding": "8px"})]),
            dcc.Graph(id="fft-plot"),
        ]),
    ], style={"backgroundColor": "#0d1117"})
