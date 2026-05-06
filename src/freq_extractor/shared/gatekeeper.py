"""Centralised API / I/O Gatekeeper (ADR-4).

All file and checkpoint I/O passes through :class:`ApiGatekeeper` for
rate limiting, retries, and unified logging.
"""

from __future__ import annotations

from pathlib import Path

from freq_extractor.shared.gatekeeper_api import ApiGatekeeper, GatekeeperError

__all__ = ["ApiGatekeeper", "GatekeeperError", "get_gatekeeper", "read_text"]


def read_text(path: Path) -> str:
    """Read UTF-8 text from *path* (bootstrap path outside rate limits)."""
    with path.open(encoding="utf-8") as fh:
        return fh.read()


_gatekeepers: dict[str, ApiGatekeeper] = {}


def get_gatekeeper(service_name: str = "file_io") -> ApiGatekeeper:
    """Return the singleton gatekeeper for *service_name*."""
    if service_name not in _gatekeepers:
        _gatekeepers[service_name] = ApiGatekeeper(service_name)
    return _gatekeepers[service_name]
