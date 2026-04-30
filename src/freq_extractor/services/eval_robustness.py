"""Noise robustness evaluation.

Sweeps across multiple noise levels to measure model degradation,
generating fresh noisy signals at each σ and computing test MSE.

References
----------
- PRD FR-7, PRD_tr §11
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torch import nn

from freq_extractor.services.data_service import (
    DatasetBuilder,
    SignalGenerator,
)
from freq_extractor.services.data_transforms import DataNormalizer
from freq_extractor.services.training_service import evaluate


def run_noise_robustness(
    model: nn.Module,
    config: dict[str, Any],
    model_type: str,
    noise_levels: list[float] | None = None,
    seed: int = 42,
) -> dict[float, float]:
    """Evaluate model at multiple noise levels.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    config : dict
        Parsed ``config/setup.json``.
    model_type : str
        ``"mlp"`` or ``"rnn"``/``"lstm"``.
    noise_levels : list[float], optional
        σ values to sweep.  Defaults to config value.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[float, float]
        Mapping ``{sigma: test_mse}``.
    """
    from freq_extractor.services.datasets import (
        MLPDataset,
        SequentialDataset,
        create_dataloader,
    )

    if noise_levels is None:
        noise_levels = config["evaluation"]["noise_levels_for_robustness"]

    sig = config["signal"]
    results: dict[float, float] = {}

    for sigma in noise_levels:
        rng = np.random.default_rng(seed)
        gen = SignalGenerator(sig["sampling_rate_hz"], sig["duration_seconds"], sig["amplitude"])
        all_entries: list = []
        for f in sig["frequencies_hz"]:
            phase = rng.uniform(0, 2 * np.pi)
            clean = gen.generate_clean(float(f), phase)
            noisy = gen.generate_noisy(clean, sigma, rng)
            all_entries.extend(
                DatasetBuilder.build_windows(clean, noisy, f, sig["window_size"])
            )

        norm = DataNormalizer()
        norm.fit(all_entries)
        all_entries = norm.transform(all_entries)

        ds_cls = MLPDataset if model_type == "mlp" else SequentialDataset
        ds = ds_cls(all_entries)
        loader = create_dataloader(
            ds, batch_size=config["training"]["batch_size"], shuffle=False,
        )
        criterion = nn.MSELoss()
        results[sigma] = evaluate(model, loader, criterion)

    return results
