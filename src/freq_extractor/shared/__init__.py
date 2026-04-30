"""Shared infrastructure for the freq_extractor package.

Provides the public API for:

* Configuration loading — :mod:`.config`
* Version validation — :mod:`.version`
* Centralised I/O gatekeeper — :mod:`.gatekeeper`

All symbols exported here are importable directly from
``freq_extractor.shared``.

Example
-------
>>> from freq_extractor.shared import get_setup, get_gatekeeper
>>> cfg = get_setup()
>>> gk = get_gatekeeper("file_io")
"""

from __future__ import annotations

from freq_extractor.shared.config import (
    clear_cache,
    get_config_dir,
    get_logging_config,
    get_rate_limits,
    get_setup,
    load_json,
    setup_logging,
)
from freq_extractor.shared.gatekeeper import (
    ApiGatekeeper,
    GatekeeperError,
    get_gatekeeper,
)
from freq_extractor.shared.version import (
    CODE_VERSION,
    MAX_CONFIG_VERSION,
    MIN_CONFIG_VERSION,
    parse_version,
    validate_config_version,
)

__all__ = [
    # config
    "clear_cache",
    "get_config_dir",
    "get_logging_config",
    "get_rate_limits",
    "get_setup",
    "load_json",
    "setup_logging",
    # gatekeeper
    "ApiGatekeeper",
    "GatekeeperError",
    "get_gatekeeper",
    # version
    "CODE_VERSION",
    "MAX_CONFIG_VERSION",
    "MIN_CONFIG_VERSION",
    "parse_version",
    "validate_config_version",
]
