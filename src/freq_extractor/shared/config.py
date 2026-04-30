"""Configuration loader and validator for the freq_extractor package.

All configuration is read from JSON files in the ``config/`` directory.
Environment variables may override selected values at runtime (see .env-example).
The loaded config is validated against the current code version on first access.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from freq_extractor.shared.version import validate_config_version

logger = logging.getLogger("freq_extractor.config")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config"
"""Resolved path to the project-level config/ directory."""

_config_cache: dict[str, Any] = {}
"""Module-level cache so config files are only read once per process."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_config_dir() -> Path:
    """Return the config directory, respecting environment overrides.

    Returns:
        Resolved absolute ``Path`` to the config directory.
    """
    override = os.environ.get("FREQ_EXTRACTOR_CONFIG_DIR")
    return Path(override) if override else _DEFAULT_CONFIG_DIR


def load_json(filename: str) -> dict[str, Any]:
    """Load a JSON config file by name from the config directory.

    Results are cached so the file is only read once per process.

    Args:
        filename: Config filename, e.g. ``"setup.json"``.

    Returns:
        Parsed JSON as a dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    if filename in _config_cache:
        return _config_cache[filename]

    path = get_config_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    _config_cache[filename] = data
    logger.debug("Loaded config: %s", path)
    return data


def get_setup() -> dict[str, Any]:
    """Load and return the main setup configuration.

    Validates the config version on first call.

    Returns:
        Parsed ``config/setup.json`` as a dict.
    """
    data = load_json("setup.json")
    validate_config_version(data["version"], "setup.json")
    return data


def get_rate_limits() -> dict[str, Any]:
    """Load and return the rate-limits configuration.

    Returns:
        Parsed ``config/rate_limits.json`` as a dict.
    """
    data = load_json("rate_limits.json")
    validate_config_version(data["rate_limits"]["version"], "rate_limits.json")
    return data["rate_limits"]


def get_logging_config() -> dict[str, Any]:
    """Load and return the logging configuration.

    Returns:
        The ``"logging"`` sub-dict from ``config/logging_config.json``.
    """
    data = load_json("logging_config.json")
    validate_config_version(data["version"], "logging_config.json")
    return data["logging"]


def clear_cache() -> None:
    """Clear the internal config cache (useful for testing).

    After calling this, the next access will re-read all config files from disk.
    """
    _config_cache.clear()
    logger.debug("Config cache cleared.")


def setup_logging() -> None:
    """Configure Python's logging system from logging_config.json.

    Should be called once at application startup, before any other logging.
    Falls back to basicConfig on error so the application can still run.
    """
    import logging.config  # noqa: PLC0415 — lazy import to keep startup fast

    try:
        log_cfg = get_logging_config()
        # Ensure the log file's parent directory exists
        for handler in log_cfg.get("handlers", {}).values():
            if "filename" in handler:
                Path(handler["filename"]).parent.mkdir(parents=True, exist_ok=True)
        logging.config.dictConfig(log_cfg)
        logger.debug("Logging configured from logging_config.json")
    except Exception as exc:  # pragma: no cover
        logging.basicConfig(level=logging.INFO)
        logging.warning("Failed to configure logging from file: %s", exc)
