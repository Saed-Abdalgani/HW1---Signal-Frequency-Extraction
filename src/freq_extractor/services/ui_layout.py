"""Dash layout builder for the Sinusoid Explorer.

Constructs the full-height sidebar + header-bar layout matching
the React reference design.

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
        # ── Left Sidebar ──
        html.Div([
            # Title area
            html.Div([
                html.H1("Sinusoid Explorer"),
                html.P("HW1 — Signal Frequency Extraction"),
            ], className="sidebar-title-block"),
            # Global controls
            global_controls(config),
            # Per-sinusoid controls
            html.Div(
                build_sin_controls(),
                className="sin-controls-area",
            ),
        ], className="sidebar"),

        # ── Main Content Area ──
        html.Div([
            # Header bar with metrics + tabs
            html.Div([
                html.Div(header_metrics(), className="metrics-bar"),
            ], className="app-header"),
            # Tab panel
            build_tabs(),
        ], className="main-area"),
    ], className="dashboard-container")
