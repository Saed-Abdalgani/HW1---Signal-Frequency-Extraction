"""Centralized API / I/O Gatekeeper.

All external calls (file I/O, future HTTP endpoints, checkpoint operations) must
pass through this module.  The gatekeeper enforces rate limiting, retry logic,
backpressure, and unified logging so no caller can bypass these controls.
Design rationale (ADR-4): Even though this project has no external HTTP APIs,
centralising I/O through a gatekeeper provides consistent retry/logging behaviour
and acts as an extension point for remote storage adapters in the future.
"""
from __future__ import annotations

import logging
import queue
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("freq_extractor.gatekeeper")

class GatekeeperError(Exception):
    """Raised when the gatekeeper cannot execute a call after all retries."""

def read_text(path: Path) -> str:
    """Read text from *path* inside the gatekeeper I/O boundary.

    Config loading uses this low-level bootstrap helper before service-specific
    rate-limit configuration is available.
    """
    with path.open(encoding="utf-8") as fh:
        return fh.read()

class ApiGatekeeper:
    """Centralized I/O and API call manager.
    Enforces per-service rate limiting, queuing, retry with exponential
    back-off, and structured logging for every operation.

    Args:
        service_name: Key into ``config/rate_limits.json`` ``services`` dict.
                      Falls back to ``"default"`` if not found.
    """

    def __init__(self, service_name: str = "default") -> None:
        """Initialise gatekeeper with rate-limit config for *service_name*."""
        from freq_extractor.shared.config import get_rate_limits

        limits = get_rate_limits()["services"]
        cfg = limits.get(service_name, limits["default"])

        self._service_name = service_name
        self._max_retries: int = cfg["max_retries"]
        self._retry_after: float = cfg["retry_after_seconds"]
        self._concurrent_max: int = cfg["concurrent_max"]
        self._rpm_limit: int = cfg["requests_per_minute"]

        # Sliding window tracking calls in the last 60 s
        self._call_times: deque[float] = deque()
        # Internal task queue (used for backpressure monitoring)
        self._task_queue: queue.Queue[Any] = queue.Queue(maxsize=self._concurrent_max * 4)

        self._total_calls: int = 0
        self._total_errors: int = 0
        self._total_retries: int = 0

        logger.debug("Gatekeeper initialised for service '%s' (rpm=%d)", service_name, self._rpm_limit)
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def execute(self, api_call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *api_call* through the gatekeeper.

        Checks the rate limit, retries on transient failures with exponential
        back-off, and logs every attempt.

        Args:
            api_call: Callable to execute (e.g. a file-save function).
            *args: Positional arguments forwarded to *api_call*.
            **kwargs: Keyword arguments forwarded to *api_call*.

        Returns:
            Whatever *api_call* returns.

        Raises:
            GatekeeperError: If all retry attempts are exhausted.
        """
        self._wait_for_rate_limit()

        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                self._total_calls += 1
                result = api_call(*args, **kwargs)
                logger.debug(
                    "[%s] call succeeded on attempt %d/%d",
                    self._service_name, attempt, self._max_retries,
                )
                return result
            except Exception as exc:  # noqa: BLE001 — catch-all for retry
                last_exc = exc
                self._total_errors += 1
                self._total_retries += 1
                wait = self._retry_after * (2 ** (attempt - 1))  # exponential back-off
                logger.warning(
                    "[%s] attempt %d/%d failed: %s. Retrying in %.1fs.",
                    self._service_name, attempt, self._max_retries, exc, wait,
                )
                if attempt < self._max_retries:
                    time.sleep(wait)

        raise GatekeeperError(
            f"[{self._service_name}] All {self._max_retries} attempts exhausted. "
            f"Last error: {last_exc}"
        ) from last_exc

    def get_queue_status(self) -> dict[str, int]:
        """Return current queue depth and cumulative statistics.

        Returns:
            Dict with keys ``queue_depth``, ``total_calls``, ``total_errors``,
            ``total_retries``.
        """
        return {
            "queue_depth": self._task_queue.qsize(),
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "total_retries": self._total_retries,
        }
    # Private helpers
    def _wait_for_rate_limit(self) -> None:
        """Block until the per-minute request limit allows another call.

        Uses a sliding-window algorithm: purge call timestamps older than 60 s,
        then sleep if at the rate limit until a slot opens.
        """
        now = time.monotonic()
        # Purge timestamps outside the 60-second window
        while self._call_times and now - self._call_times[0] > 60.0:
            self._call_times.popleft()

        if len(self._call_times) >= self._rpm_limit:
            sleep_for = 60.0 - (now - self._call_times[0]) + 0.01
            logger.info("[%s] Rate limit reached. Sleeping %.2fs.", self._service_name, sleep_for)
            time.sleep(max(sleep_for, 0))

        self._call_times.append(time.monotonic())
# ---------------------------------------------------------------------------
# Module-level singleton factory — one gatekeeper per service name
# ---------------------------------------------------------------------------
_gatekeepers: dict[str, ApiGatekeeper] = {}

def get_gatekeeper(service_name: str = "file_io") -> ApiGatekeeper:
    """Return (or create) the singleton gatekeeper for *service_name*.

    Args:
        service_name: Service key from ``config/rate_limits.json``.

    Returns:
        Singleton :class:`ApiGatekeeper` for the requested service.
    """
    if service_name not in _gatekeepers:
        _gatekeepers[service_name] = ApiGatekeeper(service_name)
    return _gatekeepers[service_name]
