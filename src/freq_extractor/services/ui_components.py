"""Reusable Dash components for the Sinusoid Explorer.

Provides header metrics, global parameter controls, per-sinusoid
controls, and the four-tab panel.
"""

from __future__ import annotations

from typing import Any

from dash import dcc, html

SIN_COLOURS = ["#00FFFF", "#FF7F00", "#00FF00", "#FFFF00"]
_MONO = {"fontFamily": "'JetBrains Mono', monospace", "fontSize": "13px"}
_LBL = {"color": "#8b949e", "fontStyle": "italic", "fontSize": "13px"}
_ROW = {"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}


def header_metrics() -> list:
    """Build the live header metrics bar."""
    items = []
    for mid, label in [("Fs", "FS:"), ("N-CYC", "N-CYC:"), ("T", "T:"),
                        ("F_MIN", "F_MIN:"), ("N", "SAMPLES:"), ("NYQ", "NYQUIST:")]:
        items.append(html.Span([
            html.Span(label, className="metric-label"), html.Span("---", id=f"metric-{mid}"),
        ], className="metric-text"))
    return items


def global_controls(config: dict[str, Any]) -> html.Div:
    """Build the Global Parameters sidebar panel."""
    def _labeled_slider(label, sid, **kw):
        return html.Div([
            html.Div([html.Label(label, className="slider-label",
                      style={"width": "auto", "minWidth": "auto", "whiteSpace": "nowrap"})],
                      style={"marginBottom": "4px"}),
            html.Div(dcc.Slider(id=sid, marks=None,
                                tooltip={"placement": "right", "always_visible": False}, **kw),
                     className="slider-container"),
        ], style={"marginTop": "12px"})

    return html.Div([
        html.H4("GLOBAL PARAMETERS", className="sidebar-section-title"),
        _labeled_slider("Sampling Freq (Fs)", "fs-slider", min=10, max=2000, step=10, value=200),
        _labeled_slider("N-Cycles", "n-cycles", min=1, max=20, step=1, value=5),
        _labeled_slider("BW (Hz)", "bw-slider", min=1, max=50, step=1, value=10),
        html.Div([
            html.Label("Display", className="slider-label", style={"width": "auto", "minWidth": "auto"}),
            dcc.RadioItems(id="display-toggle",
                options=[{"label": "LINE", "value": "LINE"}, {"label": "DOTS", "value": "DOTS"}],
                value="LINE", inline=True, className="segmented-control"),
        ], className="slider-row", style={"marginTop": "14px"}),
        html.Div([
            html.Label("Noise", className="slider-label", style={"width": "auto", "minWidth": "auto"}),
            html.Div(dcc.Dropdown(id="noise-dropdown",
                options=[{"label": n, "value": n} for n in ["None", "Gaussian", "Uniform"]],
                value="None", clearable=False, className="custom-dropdown",
                style={"width": "160px", "fontSize": "11px"}), style={"marginLeft": "auto"}),
        ], className="slider-row", style={"marginTop": "10px"}),
        html.Div([
            html.Label("Filter", className="slider-label", style={"width": "auto", "minWidth": "auto"}),
            html.Div(dcc.Dropdown(id="filter-dropdown",
                options=[{"label": f, "value": f} for f in ["None", "Bandpass", "Lowpass", "Highpass"]],
                value="Bandpass", clearable=False, className="custom-dropdown",
                style={"width": "160px", "fontSize": "11px"}), style={"marginLeft": "auto"}),
        ], className="slider-row", style={"marginTop": "10px"}),
        html.Button("► SWEEP NOISE", id="sweep-btn", className="sweep-btn"),
        dcc.Interval(id="sweep-interval", interval=100, disabled=True, n_intervals=0),
    ], className="control-block")


def _sin_slider(label, sid, c, val_id, **kw):
    """Single sin-parameter slider with value readout."""
    return html.Div([
        html.Div([html.Span(label, style=_LBL),
                  html.Span(id=val_id, style={**_MONO, "color": c})], style=_ROW),
        dcc.Slider(id=sid, marks=None, tooltip={"placement": "right", "always_visible": False}, **kw),
    ], style={"marginTop": "10px"})


def _sin_control(n: int) -> html.Div:
    """Build per-sinusoid control block for Sin *n* (1-indexed)."""
    c = SIN_COLOURS[n - 1]
    freqs, amps = [5.0, 5.0, 5.0, 5.0], [1.0, 0.8, 0.5, 0.2]
    mixes = [True, True, True, False]
    return html.Div([
        html.Div([
            html.Div(className="color-dot", style={"backgroundColor": c, "boxShadow": f"0 0 8px {c}"}),
            html.Div(f"Sin {n}", className="sin-name", style={"color": c}),
            dcc.Checklist(id=f"mix-{n}", options=[{"label": "MIX", "value": "mix"}],
                value=["mix"] if mixes[n - 1] else [], inline=True, className="sin-checkbox"),
            dcc.Checklist(id=f"bpf-{n}", options=[{"label": "BPF", "value": "bpf"}],
                value=[], inline=True, className="sin-checkbox"),
        ], className="sin-header"),
        _sin_slider("f", f"freq-{n}", c, f"f-val-{n}", min=0.1, max=100, step=0.1, value=freqs[n - 1]),
        _sin_slider("φ", f"phase-{n}", c, f"phi-val-{n}", min=0, max=6.28, step=0.01, value=0.0),
        _sin_slider("A", f"amp-{n}", c, f"a-val-{n}", min=0, max=2.0, step=0.01, value=amps[n - 1]),
        _sin_slider("σ", f"sigma-{n}", c, f"sig-val-{n}", min=0, max=1.0, step=0.01, value=0.0),
    ], className=f"control-block sin-block-{n}")


def build_sin_controls() -> list[html.Div]:
    """Return list of 4 sinusoid control blocks."""
    return [_sin_control(i) for i in range(1, 5)]


def _graph(gid):
    """Graph with dark-bordered styling."""
    return dcc.Graph(id=gid, config={"displayModeBar": True},
                     style={"border": "1px solid #30363d", "borderRadius": "6px", "overflow": "hidden"})


def build_tabs() -> html.Div:
    """Build the four visualisation tabs."""
    return html.Div([
        dcc.Tabs(id="main-tabs", value="signals", className="custom-tabs-container", children=[
            dcc.Tab(label="SIGNALS", value="signals", className="custom-tab",
                    selected_className="custom-tab--selected",
                    children=[html.Div([_graph("individual-plot"), _graph("combined-plot")],
                              className="tab-content")]),
            dcc.Tab(label="T-SNE 3D", value="tsne", className="custom-tab",
                    selected_className="custom-tab--selected", children=[html.Div([
                        html.Div([html.Label("Perplexity", className="slider-label"),
                                  dcc.Slider(id="tsne-perp", min=5, max=50, step=1, value=30,
                                             marks=None, tooltip={"placement": "bottom", "always_visible": True})],
                                 style={"padding": "8px"}),
                        _graph("tsne-plot")], className="tab-content")]),
            dcc.Tab(label="PCA 3D", value="pca", className="custom-tab",
                    selected_className="custom-tab--selected",
                    children=[html.Div([_graph("pca-plot")], className="tab-content")]),
            dcc.Tab(label="FFT SPECTRUM", value="fft", className="custom-tab",
                    selected_className="custom-tab--selected", children=[html.Div([
                        html.Div([dcc.Checklist(id="fft-log",
                            options=[{"label": "Log Scale", "value": "log"}],
                            value=[], inline=True, className="sin-checkbox")], style={"padding": "8px"}),
                        _graph("fft-plot")], className="tab-content")]),
        ])
    ], className="main-content")
