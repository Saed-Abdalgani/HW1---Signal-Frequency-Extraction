"""Dash ``Input`` / ``State`` lists for Sinusoid Explorer callbacks."""

from __future__ import annotations

from dash import Input


def build_sin_inputs() -> list:
    """Build Dash inputs for four sinusoids (mix, bpf, freq, phase, amp, sigma).

    Returns
    -------
    list
        Ordered :class:`dash.Input` list for all per-sin controls.
    """
    sin_inputs = []
    for i in range(1, 5):
        sin_inputs += [
            Input(f"mix-{i}", "value"), Input(f"bpf-{i}", "value"),
            Input(f"freq-{i}", "value"), Input(f"phase-{i}", "value"),
            Input(f"amp-{i}", "value"), Input(f"sigma-{i}", "value"),
        ]
    return sin_inputs


def analysis_inputs(sin_inputs: list) -> list:
    """Combine global controls with *sin_inputs* for analysis tabs.

    Returns
    -------
    list
        Dash dependency list without sweep interval.
    """
    return (
        [Input("fs-slider", "value"), Input("n-cycles", "value"),
         Input("bw-slider", "value"), Input("display-toggle", "value"),
         Input("noise-dropdown", "value"), Input("filter-dropdown", "value")]
        + sin_inputs
    )


def signals_callback_inputs(sin_inputs: list) -> list:
    """Dependency list for SIGNALS-tab figures (same signals as analysis inputs).

    Returns
    -------
    list
        Dash ``Input`` list for per-sin and global controls.
    """
    return analysis_inputs(sin_inputs)
