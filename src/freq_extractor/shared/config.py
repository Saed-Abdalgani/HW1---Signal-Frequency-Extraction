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

from freq_extractor.shared.gatekeeper import read_text
from freq_extractor.shared.version import validate_config_version

logger = logging.getLogger("freq_extractor.config")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config"
"""Resolved path to the project-level config/ directory."""

_config_cache: dict[str, Any] = {}
"""Module-level cache so config files are only read once per process."""


def _require_keys(data: dict[str, Any], keys: set[str], name: str) -> None:
    """Validate that a mapping contains required keys."""
    missing = keys - data.keys()
    if missing:
        raise ValueError(f"{name} missing required keys: {sorted(missing)}")


def _require_positive(data: dict[str, Any], keys: tuple[str, ...], name: str) -> None:
    """Validate that numeric config values are positive."""
    for key in keys:
        if data[key] <= 0:
            raise ValueError(f"{name}.{key} must be > 0, got {data[key]}")


def _validate_setup(data: dict[str, Any]) -> None:
    """Validate setup.json schema and numeric ranges."""
    _require_keys(data, {"version", "signal", "dataset", "model", "training", "evaluation"},
                  "setup.json")
    sig, ds, train = data["signal"], data["dataset"], data["training"]
    _require_keys(sig, {"frequencies_hz", "amplitude", "sampling_rate_hz", "duration_seconds",
                        "noise_std_ratio", "window_size"}, "signal")
    _require_keys(ds, {"train_ratio", "val_ratio", "test_ratio", "seed", "data_dir"}, "dataset")
    _require_keys(train, {"batch_size", "max_epochs", "learning_rate", "checkpoint_dir"},
                  "training")
    freqs = sig["frequencies_hz"]
    if not isinstance(freqs, list) or not freqs or any(f <= 0 for f in freqs):
        raise ValueError("signal.frequencies_hz must be a non-empty list of positive values")
    if sig["sampling_rate_hz"] < 2 * max(freqs):
        raise ValueError("signal.sampling_rate_hz violates Nyquist for frequencies_hz")
    _require_positive(sig, ("sampling_rate_hz", "duration_seconds", "window_size"), "signal")
    if sig["amplitude"] < 0 or sig["noise_std_ratio"] < 0:
        raise ValueError("signal amplitude and noise_std_ratio must be non-negative")
    total_samples = sig["sampling_rate_hz"] * sig["duration_seconds"]
    if sig["window_size"] >= total_samples:
        raise ValueError("signal.window_size must be < total signal samples")
    if abs(ds["train_ratio"] + ds["val_ratio"] + ds["test_ratio"] - 1.0) > 1e-6:
        raise ValueError("dataset split ratios must sum to 1.0")
    _require_positive(train, ("batch_size", "max_epochs", "learning_rate"), "training")


def _validate_rate_limits(data: dict[str, Any]) -> None:
    """Validate rate_limits.json schema and numeric ranges."""
    _require_keys(data, {"version", "services"}, "rate_limits")
    required = {"requests_per_minute", "requests_per_hour", "concurrent_max",
                "retry_after_seconds", "max_retries"}
    if not isinstance(data["services"], dict) or not data["services"]:
        raise ValueError("rate_limits.services must be a non-empty mapping")
    for name, service in data["services"].items():
        _require_keys(service, required, f"rate_limits.services.{name}")
        _require_positive(service, ("requests_per_minute", "requests_per_hour",
                                    "concurrent_max"), f"rate_limits.services.{name}")
        if service["retry_after_seconds"] < 0 or service["max_retries"] < 0:
            raise ValueError(f"rate_limits.services.{name} retry/max_retries must be >= 0")


def _validate_logging(data: dict[str, Any]) -> None:
    """Validate logging_config.json schema and levels."""
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"}
    _require_keys(data, {"version", "disable_existing_loggers", "handlers", "root"}, "logging")
    if not isinstance(data["handlers"], dict) or not data["handlers"]:
        raise ValueError("logging.handlers must be a non-empty mapping")
    for name, handler in data["handlers"].items():
        _require_keys(handler, {"class", "level"}, f"logging.handlers.{name}")
        if handler["level"] not in valid_levels:
            raise ValueError(f"logging.handlers.{name}.level is invalid: {handler['level']}")
    if data["root"].get("level") not in valid_levels:
        raise ValueError(f"logging.root.level is invalid: {data['root'].get('level')}")


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

    data: dict[str, Any] = json.loads(read_text(path))

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
    _validate_setup(data)
    return data


def get_rate_limits() -> dict[str, Any]:
    """Load and return the rate-limits configuration.

    Returns:
        Parsed ``config/rate_limits.json`` as a dict.
    """
    data = load_json("rate_limits.json")
    validate_config_version(data["rate_limits"]["version"], "rate_limits.json")
    _validate_rate_limits(data["rate_limits"])
    return data["rate_limits"]


def get_logging_config() -> dict[str, Any]:
    """Load and return the logging configuration.

    Returns:
        The ``"logging"`` sub-dict from ``config/logging_config.json``.
    """
    data = load_json("logging_config.json")
    validate_config_version(data["version"], "logging_config.json")
    _validate_logging(data["logging"])
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
