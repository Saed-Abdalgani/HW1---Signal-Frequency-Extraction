"""Tests for UIService — 6K.1, 6K.2.

Validates Dash app construction and layout structure.
"""

from __future__ import annotations

import dash

from freq_extractor.services.ui_service import UIService


class TestUIService:
    """UIService tests."""

    def test_build_app_returns_dash(self, sample_config) -> None:
        """6K.1: build_app() returns a Dash app instance."""
        svc = UIService(config=sample_config)
        app = svc.build_app()
        assert isinstance(app, dash.Dash)

    def test_app_title(self, sample_config) -> None:
        """App title is 'SINUSOID EXPLORER'."""
        svc = UIService(config=sample_config)
        app = svc.build_app()
        assert app.title == "SINUSOID EXPLORER"

    def test_layout_not_none(self, sample_config) -> None:
        """App layout is set after build."""
        svc = UIService(config=sample_config)
        app = svc.build_app()
        assert app.layout is not None

    def test_default_config(self) -> None:
        """UIService works with no config (defaults)."""
        svc = UIService()
        app = svc.build_app()
        assert isinstance(app, dash.Dash)

    def test_build_idempotent(self, sample_config) -> None:
        """Calling build_app twice works without error."""
        svc = UIService(config=sample_config)
        svc.build_app()
        app2 = svc.build_app()
        assert isinstance(app2, dash.Dash)
