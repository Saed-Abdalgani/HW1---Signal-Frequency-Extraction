"""UIService — Sinusoid Explorer interactive Dash dashboard.

Provides the ``UIService`` class that builds and launches a Plotly Dash
application featuring four visualisation tabs: Signals, T-SNE 3D,
PCA 3D, and FFT Spectrum.

References
----------
- PRD FR-9..FR-13, PLAN ADR-6
"""

from __future__ import annotations

import logging
from typing import Any

import dash
from dash import html

from freq_extractor.services.ui_callbacks import register_callbacks
from freq_extractor.services.ui_layout import build_layout

logger = logging.getLogger("freq_extractor.ui")


class UIService:
    """Sinusoid Explorer interactive dashboard service.

    Parameters
    ----------
    config : dict | None
        Parsed ``config/setup.json``.  Passed to layout and callbacks.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialise UIService and store config."""
        self.config = config or {}
        self._app: dash.Dash | None = None

    def build_app(self) -> dash.Dash:
        """Construct the Dash application with layout and callbacks.

        Returns
        -------
        dash.Dash
            Fully configured Dash app object.
        """
        import os
        assets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "assets")
        self._app = dash.Dash(
            __name__,
            title="SINUSOID EXPLORER",
            assets_folder=assets_path,
            suppress_callback_exceptions=True,
        )
        self._app.layout = build_layout(self.config)
        register_callbacks(self._app, self.config)
        logger.info("Sinusoid Explorer app built successfully.")
        return self._app

    def launch_ui(self, port: int = 8050) -> None:
        """Start the Dash development server.

        Parameters
        ----------
        port : int
            Port number (default 8050).
        """
        if self._app is None:
            self.build_app()
        logger.info("Launching Sinusoid Explorer on http://localhost:%d", port)
        self._app.run(debug=False, port=port)
