"""Base class for FreqExtractorSDK handling core state and configuration.

Separated from :mod:`sdk.sdk` to comply with the ≤ 145 code-line constraint
after splitting the SDK into multiple files.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from freq_extractor.constants import DEFAULT_SEED
from freq_extractor.shared import get_setup, setup_logging

logger = logging.getLogger("freq_extractor.sdk.base")


class FreqExtractorSDKBase:
    """Base class managing config, random seed, and compute device.

    Parameters
    ----------
    config : dict | None
        Pre-loaded config dict.  If *None*, loads ``config/setup.json``
        automatically on first use.
    seed : int | None
        Random seed override.  Falls back to config/env/DEFAULT_SEED.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialise SDK base, configure logging and device."""
        setup_logging()
        self._config = config
        self._seed = seed
        self._device = self._resolve_device()
        logger.info("SDK base initialised (device=%s)", self._device)

    @property
    def config(self) -> dict[str, Any]:
        """Lazily load and return the setup configuration."""
        if self._config is None:
            self._config = get_setup()
        return self._config

    @property
    def seed(self) -> int:
        """Resolve the seed from init → env → config → default."""
        if self._seed is not None:
            return self._seed
        env = os.environ.get("FREQ_EXTRACTOR_SEED")
        if env is not None:
            return int(env)
        return self.config.get("dataset", {}).get("seed", DEFAULT_SEED)

    @property
    def device(self) -> torch.device:
        """Return the resolved compute device."""
        return self._device

    @staticmethod
    def _resolve_device() -> torch.device:
        """Determine compute device, honouring FREQ_EXTRACTOR_FORCE_CPU."""
        force_cpu = os.environ.get("FREQ_EXTRACTOR_FORCE_CPU", "0") == "1"
        if force_cpu or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")
