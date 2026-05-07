"""Rate-limited I/O gatekeeper implementation."""

from __future__ import annotations

import logging
import queue
import time
from collections import deque
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("freq_extractor.gatekeeper")


class GatekeeperError(Exception):
    """Raised when the gatekeeper cannot execute a call after all retries."""


class ApiGatekeeper:
    """Per-service rate limiting, retries, and logging for I/O operations."""

    def __init__(self, service_name: str = "default") -> None:
        """Load limits from ``config/rate_limits.json`` for *service_name*."""
        from freq_extractor.shared.config import get_rate_limits

        limits = get_rate_limits()["services"]
        cfg = limits.get(service_name, limits["default"])

        self._service_name = service_name
        self._max_retries: int = cfg["max_retries"]
        self._retry_after: float = cfg["retry_after_seconds"]
        self._concurrent_max: int = cfg["concurrent_max"]
        self._rpm_limit: int = cfg["requests_per_minute"]

        self._call_times: deque[float] = deque()
        self._task_queue: queue.Queue[Any] = queue.Queue(maxsize=self._concurrent_max * 4)

        self._total_calls: int = 0
        self._total_errors: int = 0
        self._total_retries: int = 0

        logger.debug("Gatekeeper initialised for service '%s' (rpm=%d)", service_name, self._rpm_limit)

    def execute(self, api_call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *api_call* with rate limiting and exponential back-off retries.

        Parameters
        ----------
        api_call
            Zero-argument or general callable to run.
        *args, **kwargs
            Forwarded to *api_call*.

        Returns
        -------
        Any
            Return value of *api_call*.

        Raises
        ------
        GatekeeperError
            If every attempt fails.
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
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self._total_errors += 1
                self._total_retries += 1
                wait = self._retry_after * (2 ** (attempt - 1))
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
        """Return queue depth and cumulative counters.

        Returns
        -------
        dict[str, int]
            Status fields ``queue_depth``, ``total_calls``, ``total_errors``,
            and ``total_retries``.
        """
        return {
            "queue_depth": self._task_queue.qsize(),
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "total_retries": self._total_retries,
        }

    def _wait_for_rate_limit(self) -> None:
        """Block until the per-minute request limit allows another call."""
        now = time.monotonic()
        while self._call_times and now - self._call_times[0] > 60.0:
            self._call_times.popleft()

        if len(self._call_times) >= self._rpm_limit:
            sleep_for = 60.0 - (now - self._call_times[0]) + 0.01
            logger.info("[%s] Rate limit reached. Sleeping %.2fs.", self._service_name, sleep_for)
            time.sleep(max(sleep_for, 0))

        self._call_times.append(time.monotonic())
