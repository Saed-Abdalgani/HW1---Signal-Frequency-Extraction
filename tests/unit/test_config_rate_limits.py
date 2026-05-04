"""Rate limits validation tests for config.py — DV.2.

Validates rate limit schema and numeric ranges.
"""

from __future__ import annotations

import json

import pytest

from freq_extractor.shared.config import clear_cache, get_rate_limits


class TestRateLimitsValidation:
    """Rate limits configuration validation tests."""

    def test_get_rate_limits(self, tmp_config_dir) -> None:
        """get_rate_limits returns dict with 'services' key."""
        rl = get_rate_limits()
        assert "services" in rl

    def test_invalid_rate_limits_raise(self, tmp_config_dir) -> None:
        """DV.2: rate limit schema validates required numeric ranges."""
        path = tmp_config_dir / "rate_limits.json"
        cfg = json.loads(path.read_text())
        cfg["rate_limits"]["services"]["default"]["requests_per_minute"] = 0
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="requests_per_minute"):
            get_rate_limits()

    def test_rate_limits_missing_services(self, tmp_config_dir) -> None:
        """Missing services in rate limits raises ValueError."""
        path = tmp_config_dir / "rate_limits.json"
        cfg = json.loads(path.read_text())
        cfg["rate_limits"]["services"] = {}
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="must be a non-empty mapping"):
            get_rate_limits()

    def test_rate_limits_negative_retry(self, tmp_config_dir) -> None:
        """Negative retry in rate limits raises ValueError."""
        path = tmp_config_dir / "rate_limits.json"
        cfg = json.loads(path.read_text())
        cfg["rate_limits"]["services"]["default"]["max_retries"] = -1
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="retry/max_retries must be >= 0"):
            get_rate_limits()
