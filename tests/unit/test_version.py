"""Tests for version.py — 6H.1..6H.4.

Validates version parsing and config compatibility checking.
"""

from __future__ import annotations

import pytest

from freq_extractor.shared.version import (
    CODE_VERSION,
    MAX_CONFIG_VERSION,
    MIN_CONFIG_VERSION,
    parse_version,
    validate_config_version,
)


class TestParseVersion:
    """Version string parsing tests."""

    def test_parse_1_00(self) -> None:
        """6H.1: parse_version('1.00') → (1, 0)."""
        assert parse_version("1.00") == (1, 0)

    def test_parse_1_50(self) -> None:
        """parse_version('1.50') → (1, 50)."""
        assert parse_version("1.50") == (1, 50)

    def test_invalid_string_raises(self) -> None:
        """6H.2: Non-numeric string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid version"):
            parse_version("abc")

    def test_too_many_parts_raises(self) -> None:
        """Three-part version raises ValueError."""
        with pytest.raises(ValueError, match="Expected format"):
            parse_version("1.2.3")

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError):
            parse_version("")


class TestValidateConfigVersion:
    """Config version compatibility tests."""

    def test_compatible_version_passes(self) -> None:
        """Current CODE_VERSION is compatible with itself."""
        validate_config_version(CODE_VERSION, "test")

    def test_major_mismatch_raises(self) -> None:
        """6H.3: Major version mismatch raises ValueError."""
        with pytest.raises(ValueError, match="[Mm]ajor version"):
            validate_config_version("2.00", "test")

    def test_above_max_raises(self) -> None:
        """6H.4: Version above MAX_CONFIG_VERSION raises ValueError."""
        high = f"{parse_version(MAX_CONFIG_VERSION)[0]}.{parse_version(MAX_CONFIG_VERSION)[1] + 1}"
        with pytest.raises(ValueError, match="outside supported range"):
            validate_config_version(high, "test")

    def test_min_version_accepted(self) -> None:
        """MIN_CONFIG_VERSION is accepted."""
        validate_config_version(MIN_CONFIG_VERSION, "test")

    def test_max_version_accepted(self) -> None:
        """MAX_CONFIG_VERSION is accepted."""
        validate_config_version(MAX_CONFIG_VERSION, "test")
