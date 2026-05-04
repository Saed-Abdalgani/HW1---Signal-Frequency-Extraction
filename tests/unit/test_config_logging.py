"""Logging configuration validation tests for config.py — DV.3.

Validates logging schema and level validation.
"""

from __future__ import annotations

import json

import pytest

from freq_extractor.shared.config import clear_cache, get_logging_config


class TestLoggingValidation:
    """Logging configuration validation tests."""

    def test_get_logging_config(self, tmp_config_dir) -> None:
        """get_logging_config returns dict with 'version' key."""
        lc = get_logging_config()
        assert "version" in lc

    def test_invalid_logging_level_raises(self, tmp_config_dir) -> None:
        """DV.3: logging levels must be standard Python logging levels."""
        path = tmp_config_dir / "logging_config.json"
        cfg = json.loads(path.read_text())
        cfg["logging"]["handlers"]["console"]["level"] = "NOPE"
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="level"):
            get_logging_config()

    def test_logging_missing_handlers(self, tmp_config_dir) -> None:
        """Missing handlers in logging config raises ValueError."""
        path = tmp_config_dir / "logging_config.json"
        cfg = json.loads(path.read_text())
        cfg["logging"]["handlers"] = {}
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="must be a non-empty mapping"):
            get_logging_config()

    def test_logging_invalid_root_level(self, tmp_config_dir) -> None:
        """Invalid root level in logging config raises ValueError."""
        path = tmp_config_dir / "logging_config.json"
        cfg = json.loads(path.read_text())
        cfg["logging"]["root"]["level"] = "INVALID"
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="root.level is invalid"):
            get_logging_config()
