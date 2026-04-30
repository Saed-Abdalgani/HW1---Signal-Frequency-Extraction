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


def build_layout(config: dict[str, Any]) -> list:
    """Assemble the full dashboard layout.

    Parameters
    ----------
    config : dict
        Parsed ``config/setup.json``.

    Returns
    -------
    list
        Top-level Dash children list.
    """
    return [
        html.H1(
            "SINUSOID EXPLORER",
            style={
                "textAlign": "center", "padding": "16px 0 8px",
                "background": "linear-gradient(90deg, #58a6ff, #51cf66)",
                "WebkitBackgroundClip": "text",
                "WebkitTextFillColor": "transparent",
                "fontSize": "28px", "fontWeight": "800", "letterSpacing": "2px",
            },
        ),
        header_metrics(),
        html.Div(
            [
                html.Div(
                    [global_controls(config)] + build_sin_controls(),
                    style={
                        "width": "300px", "overflowY": "auto",
                        "maxHeight": "90vh", "padding": "0 12px",
                    },
                ),
                html.Div(
                    [build_tabs()],
                    style={"flex": "1", "padding": "0 12px"},
                ),
            ],
            style={"display": "flex", "gap": "8px"},
        ),
    ]
