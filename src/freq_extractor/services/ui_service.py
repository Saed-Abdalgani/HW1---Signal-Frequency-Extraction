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
from pathlib import Path
from typing import Any

import dash

from freq_extractor.services.ui_callbacks import register_callbacks
from freq_extractor.services.ui_layout import build_layout

logger = logging.getLogger("freq_extractor.ui")


def _dash_assets_folder() -> str:
    """Path to Dash static assets bundled next to ``freq_extractor``.

    Older builds placed ``assets`` at repo root four levels above this module;
    that breaks when ``freq_extractor`` is imported from ``site-packages``.

    Returns
    -------
    str
        Absolute path containing ``style.css`` (preferred: ``freq_extractor/assets``).
    """
    bundled = Path(__file__).resolve().parent.parent / "assets"
    if bundled.is_dir() and (bundled / "style.css").is_file():
        return str(bundled)
    for anc in Path(__file__).resolve().parents:
        legacy = anc / "assets"
        if legacy.is_dir() and (legacy / "style.css").is_file():
            logger.warning("Using legacy repo-root assets folder: %s", legacy)
            return str(legacy)
    return str(bundled)


def _disable_asset_cache(app: dash.Dash) -> None:
    """Avoid stale CSS when running with ``debug=False`` (browser / proxy caches)."""
    from flask import request

    @app.server.after_request
    def _apply(resp):
        rp = getattr(request, "path", "") or ""
        if rp.startswith("/assets"):
            resp.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
        return resp


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
        assets_path = _dash_assets_folder()
        if not Path(assets_path, "style.css").is_file():
            logger.warning("Dash assets incomplete at %s (missing style.css).", assets_path)
        logger.info("Dash assets_folder=%s", assets_path)
        self._app = dash.Dash(
            __name__,
            title="SINUSOID EXPLORER",
            assets_folder=assets_path,
            suppress_callback_exceptions=True,
        )
        _disable_asset_cache(self._app)
        self._app.layout = build_layout(self.config)
        register_callbacks(self._app, self.config)
        logger.info("Sinusoid Explorer app built successfully.")
        return self._app

    def launch_ui(self, port: int = 8050, *, debug: bool = True) -> None:
        """Start the Dash development server.

        When *debug* is ``True`` (default), Flask reloads after ``.py`` / asset
        changes so the dashboard tracks saved edits without a manual restart.

        Parameters
        ----------
        port : int
            Port number (default 8050).
        debug : bool
            If ``True``, enable Dash/Flask debug reloader + hot reload.
            Use ``False`` for CI or a quieter one-shot serve.
        """
        if self._app is None:
            self.build_app()
        logger.info(
            "Launching Sinusoid Explorer on http://localhost:%d (debug=%s)",
            port,
            debug,
        )
        if debug:
            self._app.run(
                debug=True,
                port=port,
                threaded=True,
                dev_tools_hot_reload=True,
                dev_tools_hot_reload_interval=1000,
            )
        else:
            self._app.run(debug=False, port=port, threaded=True)
