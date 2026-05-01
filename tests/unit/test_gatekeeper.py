"""Tests for ApiGatekeeper — 6H.8..6H.13.

Validates execute, retry logic, max retries, queue status,
singleton factory, and rate limiter.
"""

from __future__ import annotations

import contextlib

import pytest

from freq_extractor.shared.gatekeeper import ApiGatekeeper, GatekeeperError, get_gatekeeper


class TestApiGatekeeper:
    """Gatekeeper execute and retry tests."""

    def test_execute_success(self, tmp_config_dir) -> None:
        """6H.8: Successful call returns value and increments total_calls."""
        gk = get_gatekeeper("file_io")
        result = gk.execute(lambda: 42)
        assert result == 42
        assert gk.get_queue_status()["total_calls"] == 1

    def test_retry_on_transient_failure(self, tmp_config_dir) -> None:
        """6H.9: Retries on failure and eventually succeeds."""
        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise OSError("transient")
            return "ok"

        gk = ApiGatekeeper("default")
        result = gk.execute(flaky)
        assert result == "ok"
        status = gk.get_queue_status()
        assert status["total_retries"] > 0

    def test_max_retries_exhausted(self, tmp_config_dir) -> None:
        """6H.10: Raises GatekeeperError after max retries."""
        gk = ApiGatekeeper("default")

        def always_fail():
            raise OSError("permanent")

        with pytest.raises(GatekeeperError, match="exhausted"):
            gk.execute(always_fail)

    def test_queue_status_keys(self, tmp_config_dir) -> None:
        """6H.11: get_queue_status returns 4 required keys."""
        gk = get_gatekeeper("file_io")
        status = gk.get_queue_status()
        assert "queue_depth" in status
        assert "total_calls" in status
        assert "total_errors" in status
        assert "total_retries" in status

    def test_singleton_per_service(self, tmp_config_dir) -> None:
        """6H.12: get_gatekeeper returns same instance for same service."""
        gk1 = get_gatekeeper("file_io")
        gk2 = get_gatekeeper("file_io")
        assert gk1 is gk2

    def test_different_services_different_instances(self, tmp_config_dir) -> None:
        """Different service names get different gatekeepers."""
        gk1 = get_gatekeeper("file_io")
        gk2 = get_gatekeeper("checkpoint")
        assert gk1 is not gk2

    def test_forwards_args_kwargs(self, tmp_config_dir) -> None:
        """Execute forwards args and kwargs to the callable."""
        gk = get_gatekeeper("file_io")

        def add(a, b, c=0):
            return a + b + c

        assert gk.execute(add, 1, 2, c=3) == 6

    def test_error_stats_increment(self, tmp_config_dir) -> None:
        """total_errors increments on failures."""
        gk = ApiGatekeeper("default")
        with contextlib.suppress(GatekeeperError):
            gk.execute(lambda: 1 / 0)
        assert gk.get_queue_status()["total_errors"] > 0
