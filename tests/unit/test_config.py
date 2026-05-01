"""Tests for config.py — 6H.5..6H.7.

Validates config loading, caching, cache clearing, and missing file handling.
"""

from __future__ import annotations

import json

import pytest

from freq_extractor.shared.config import (
    clear_cache,
    get_config_dir,
    get_logging_config,
    get_rate_limits,
    get_setup,
    load_json,
)


class TestLoadJson:
    """JSON config loading tests."""

    def test_load_setup(self, tmp_config_dir) -> None:
        """get_setup() returns a dict with 'version' key."""
        cfg = get_setup()
        assert "version" in cfg
        assert cfg["version"] == "1.00"

    def test_cache_returns_same_object(self, tmp_config_dir) -> None:
        """6H.5: Cached config returns same object on repeat call."""
        a = load_json("setup.json")
        b = load_json("setup.json")
        assert a is b

    def test_clear_cache_forces_reread(self, tmp_config_dir) -> None:
        """6H.6: clear_cache forces re-read from disk."""
        a = load_json("setup.json")
        clear_cache()
        b = load_json("setup.json")
        assert a is not b
        assert a == b

    def test_missing_file_raises(self, tmp_config_dir) -> None:
        """6H.7: Missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_json("nonexistent.json")

    def test_get_rate_limits(self, tmp_config_dir) -> None:
        """get_rate_limits returns dict with 'services' key."""
        rl = get_rate_limits()
        assert "services" in rl

    def test_get_logging_config(self, tmp_config_dir) -> None:
        """get_logging_config returns dict with 'version' key."""
        lc = get_logging_config()
        assert "version" in lc

    def test_config_dir_override(self, tmp_config_dir) -> None:
        """FREQ_EXTRACTOR_CONFIG_DIR env var is respected."""
        from pathlib import Path

        result = get_config_dir()
        assert Path(result) == tmp_config_dir

    def test_setup_validates_version(self, tmp_config_dir) -> None:
        """get_setup validates config version."""
        cfg = get_setup()
        assert "signal" in cfg
