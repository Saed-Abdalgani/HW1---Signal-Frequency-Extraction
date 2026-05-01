"""Shared test fixtures for the freq_extractor test suite.

Provides deterministic seeding, module-cache cleanup, sample config,
and small dataset generators used across all unit and integration tests.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from freq_extractor.constants import FREQUENCY_LABELS
from freq_extractor.services.data_service import DatasetBuilder, SignalGenerator


@pytest.fixture(autouse=True)
def _set_seed():
    """Set all random seeds for full determinism."""
    torch.manual_seed(42)
    np.random.seed(42)  # noqa: NPY002
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(autouse=True)
def _reset_caches():
    """Clear module-level caches between tests to avoid cross-contamination."""
    from freq_extractor.shared import gatekeeper as gk_mod
    from freq_extractor.shared.config import clear_cache

    clear_cache()
    gk_mod._gatekeepers.clear()
    yield
    clear_cache()
    gk_mod._gatekeepers.clear()


@pytest.fixture
def rng():
    """A seeded NumPy random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def signal_gen():
    """Default SignalGenerator: fs=200, dur=1.0, amp=1.0."""
    return SignalGenerator(sampling_rate=200, duration=1.0, amplitude=1.0)


@pytest.fixture
def sample_config(tmp_path):
    """Minimal but complete config dict for fast tests."""
    return {
        "version": "1.00",
        "signal": {
            "frequencies_hz": [5, 15, 30, 50],
            "amplitude": 1.0,
            "sampling_rate_hz": 200,
            "duration_seconds": 1.0,
            "noise_std_ratio": 0.10,
            "window_size": 10,
        },
        "dataset": {
            "train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15,
            "seed": 42, "data_dir": str(tmp_path / "data"),
        },
        "model": {
            "hidden_size": 16, "num_layers": 1, "dropout": 0.0,
            "mlp_hidden_sizes": [16, 32, 16],
        },
        "training": {
            "batch_size": 32, "max_epochs": 5, "learning_rate": 0.001,
            "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_eps": 1e-8,
            "early_stop_patience": 3, "early_stop_min_delta": 1e-6,
            "lr_reduce_factor": 0.5, "lr_reduce_patience": 2, "lr_min": 1e-6,
            "grad_clip_max_norm": 1.0,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
        "evaluation": {
            "noise_levels_for_robustness": [0.05, 0.10],
            "results_dir": str(tmp_path / "results"),
            "plot_dpi": 72, "n_prediction_examples": 10,
        },
    }


@pytest.fixture
def small_entries(signal_gen, rng):
    """Small dataset of entries for all 4 frequencies."""
    entries: list = []
    for f in FREQUENCY_LABELS:
        clean = signal_gen.generate_clean(float(f))
        noisy = signal_gen.generate_noisy(clean, 0.1, rng)
        entries.extend(DatasetBuilder.build_windows(clean, noisy, f, 10))
    return entries


@pytest.fixture
def tmp_config_dir(tmp_path, sample_config):
    """Temp config dir with all 3 JSON files. Sets env override."""
    import os

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "setup.json").write_text(json.dumps(sample_config))
    rl = {"rate_limits": {"version": "1.00", "services": {
        "default": {"requests_per_minute": 9999, "requests_per_hour": 99999,
                     "concurrent_max": 4, "retry_after_seconds": 0.001, "max_retries": 3},
        "file_io": {"requests_per_minute": 9999, "requests_per_hour": 99999,
                    "concurrent_max": 8, "retry_after_seconds": 0.001, "max_retries": 3},
        "checkpoint": {"requests_per_minute": 9999, "requests_per_hour": 99999,
                       "concurrent_max": 2, "retry_after_seconds": 0.001, "max_retries": 3},
    }}}
    (config_dir / "rate_limits.json").write_text(json.dumps(rl))
    lc = {"version": "1.00", "logging": {
        "version": 1, "disable_existing_loggers": False,
        "handlers": {"console": {"class": "logging.StreamHandler", "level": "WARNING"}},
        "root": {"level": "WARNING", "handlers": ["console"]},
    }}
    (config_dir / "logging_config.json").write_text(json.dumps(lc))
    old = os.environ.get("FREQ_EXTRACTOR_CONFIG_DIR")
    os.environ["FREQ_EXTRACTOR_CONFIG_DIR"] = str(config_dir)
    yield config_dir
    if old is None:
        os.environ.pop("FREQ_EXTRACTOR_CONFIG_DIR", None)
    else:
        os.environ["FREQ_EXTRACTOR_CONFIG_DIR"] = old
