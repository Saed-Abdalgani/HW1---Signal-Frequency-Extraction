"""Version management and configuration compatibility validation.

This module defines the application version and provides utilities to verify
that loaded configuration files are compatible with the current codebase.
Increment CODE_VERSION when breaking config changes are introduced.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

CODE_VERSION: str = "1.00"
"""Current application code version.  Increment on breaking API changes."""

MIN_CONFIG_VERSION: str = "1.00"
"""Minimum supported configuration file version."""

MAX_CONFIG_VERSION: str = "1.99"
"""Maximum supported configuration file version (exclusive of next major)."""

__version__: str = CODE_VERSION
__all__ = [
    "CODE_VERSION",
    "MIN_CONFIG_VERSION",
    "MAX_CONFIG_VERSION",
    "validate_config_version",
    "parse_version",
]


# ---------------------------------------------------------------------------
# Version utilities
# ---------------------------------------------------------------------------


def parse_version(version_str: str) -> tuple[int, int]:
    """Parse a version string of the form 'MAJOR.MINOR' into a tuple.

    Args:
        version_str: Version string, e.g. ``"1.00"``.

    Returns:
        Tuple ``(major, minor)`` as integers.

    Raises:
        ValueError: If the version string cannot be parsed.
    """
    try:
        parts = version_str.split(".")
        if len(parts) != 2:  # noqa: PLR2004
            raise ValueError("Expected format MAJOR.MINOR")
        return int(parts[0]), int(parts[1])
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Invalid version string '{version_str}'. Expected format 'MAJOR.MINOR'."
        ) from exc


def validate_config_version(config_version: str, config_name: str = "config") -> None:
    """Validate that a configuration file version is compatible with this code.

    Compatibility rule: the config major version must match the code major version,
    and the config minor version must be within the supported range.

    Args:
        config_version: Version string from the config file (e.g. ``"1.00"``).
        config_name: Human-readable name of the config file for error messages.

    Raises:
        ValueError: If the config version is incompatible with the current code.
    """
    code_major, _ = parse_version(CODE_VERSION)
    min_major, min_minor = parse_version(MIN_CONFIG_VERSION)
    max_major, max_minor = parse_version(MAX_CONFIG_VERSION)
    cfg_major, cfg_minor = parse_version(config_version)

    if cfg_major != code_major:
        raise ValueError(
            f"[{config_name}] Major version mismatch: config={config_version}, "
            f"code={CODE_VERSION}. A major version change indicates a breaking "
            "change — please update your configuration file."
        )

    cfg_tuple = (cfg_major, cfg_minor)
    min_tuple = (min_major, min_minor)
    max_tuple = (max_major, max_minor)

    if not (min_tuple <= cfg_tuple <= max_tuple):
        raise ValueError(
            f"[{config_name}] Version '{config_version}' is outside supported range "
            f"[{MIN_CONFIG_VERSION}, {MAX_CONFIG_VERSION}]."
        )
