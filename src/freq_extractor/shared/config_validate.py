"""JSON schema validation helpers for ``config/*.json``."""

from __future__ import annotations

from typing import Any


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


def validate_setup(data: dict[str, Any]) -> None:
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


def validate_rate_limits(data: dict[str, Any]) -> None:
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


def validate_logging(data: dict[str, Any]) -> None:
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
