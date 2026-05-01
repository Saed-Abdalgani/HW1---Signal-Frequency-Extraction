"""Dash layout builder for the Sinusoid Explorer.

Constructs the sidebar, header metrics bar, and assembles the
full dashboard layout using components from ``ui_components.py``.

References
----------
- PRD FR-9, PLAN ADR-6
"""

from __future__ import annotations

from typing import Any

from dash import html

from freq_extractor.services.ui_components import (
    SIN_COLOURS,
    build_sin_controls,
    build_tabs,
    global_controls,
    header_metrics,
)

# Re-export SIN_COLOURS for callbacks
__all__ = ["SIN_COLOURS", "build_layout"]


def build_layout(config: dict[str, Any]) -> html.Div:
    """Assemble the full dashboard layout.

    Parameters
    ----------
    config : dict
        Parsed ``config/setup.json``.

    Returns
    -------
    html.Div
        Top-level Dash container.
    """
    return html.Div([
        html.Div([
            html.Div([
                html.Div("(*)", className="app-logo"),
                html.H1("SINUSOID EXPLORER", className="app-title"),
            ], className="title-group"),
            html.Div(header_metrics(), className="metrics-bar"),
        ], className="app-header"),
        html.Div(
            [
                html.Div(
                    [global_controls(config)] + build_sin_controls(),
                    className="sidebar",
                ),
                html.Div(
                    [build_tabs()],
                    className="main-content",
                ),
            ],
            className="main-grid",
        ),
    ], className="dashboard-container")
