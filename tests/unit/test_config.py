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

    def test_setup_schema_and_ranges(self, tmp_config_dir) -> None:
        """DV.1, DV.4-DV.6: setup schema and critical ranges validate."""
        cfg = get_setup()
        assert cfg["training"]["learning_rate"] > 0
        assert cfg["training"]["batch_size"] > 0
        assert cfg["training"]["max_epochs"] > 0
        assert min(cfg["signal"]["frequencies_hz"]) > 0
        assert cfg["signal"]["sampling_rate_hz"] >= 2 * max(cfg["signal"]["frequencies_hz"])

    def test_invalid_training_range_raises(self, tmp_config_dir) -> None:
        """DV.4: Bad hyperparameter ranges fail fast."""
        path = tmp_config_dir / "setup.json"
        cfg = json.loads(path.read_text())
        cfg["training"]["batch_size"] = 0
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="batch_size"):
            get_setup()

    def test_invalid_frequency_list_raises(self, tmp_config_dir) -> None:
        """DV.5: Empty or non-positive frequencies fail fast."""
        path = tmp_config_dir / "setup.json"
        cfg = json.loads(path.read_text())
        cfg["signal"]["frequencies_hz"] = [5, -1]
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="frequencies_hz"):
            get_setup()

    def test_invalid_nyquist_raises(self, tmp_config_dir) -> None:
        """DV.6: sampling_rate must satisfy Nyquist at startup."""
        path = tmp_config_dir / "setup.json"
        cfg = json.loads(path.read_text())
        cfg["signal"]["sampling_rate_hz"] = 50
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="Nyquist"):
            get_setup()

    def test_invalid_rate_limits_raise(self, tmp_config_dir) -> None:
        """DV.2: rate limit schema validates required numeric ranges."""
        path = tmp_config_dir / "rate_limits.json"
        cfg = json.loads(path.read_text())
        cfg["rate_limits"]["services"]["default"]["requests_per_minute"] = 0
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="requests_per_minute"):
            get_rate_limits()

    def test_invalid_logging_level_raises(self, tmp_config_dir) -> None:
        """DV.3: logging levels must be standard Python logging levels."""
        path = tmp_config_dir / "logging_config.json"
        cfg = json.loads(path.read_text())
        cfg["logging"]["handlers"]["console"]["level"] = "NOPE"
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="level"):
            get_logging_config()
