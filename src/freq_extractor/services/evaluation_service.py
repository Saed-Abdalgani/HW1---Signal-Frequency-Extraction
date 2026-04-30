"""Evaluation service — metrics computation and comparison tables.

Provides split-level MSE computation, per-frequency analysis,
and Markdown comparison table generation.

References
----------
- PRD FR-7, PRD_tr §§10–11
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from freq_extractor.constants import FREQUENCY_LABELS
from freq_extractor.services.training_service import evaluate

logger = logging.getLogger("freq_extractor.evaluation")


def compute_split_mse(
    model: nn.Module,
    loaders: dict[str, DataLoader],
) -> dict[str, float]:
    """Compute MSE for each data split.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    loaders : dict[str, DataLoader]
        Mapping ``{"train": ..., "val": ..., "test": ...}``.

    Returns
    -------
    dict[str, float]
        MSE per split.
    """
    criterion = nn.MSELoss()
    return {name: evaluate(model, loader, criterion) for name, loader in loaders.items()}


def compute_per_frequency_mse(
    model: nn.Module,
    entries: list[dict[str, np.ndarray]],
    model_type: str,
    config: dict[str, Any],
) -> dict[int, float]:
    """Compute test MSE separately for each frequency class.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    entries : list
        Test-set entries (normalised).
    model_type : str
        ``"mlp"`` or ``"rnn"``/``"lstm"``.
    config : dict
        Parsed setup config.

    Returns
    -------
    dict[int, float]
        Mapping ``{freq_hz: mse}``.
    """
    from freq_extractor.services.datasets import (
        MLPDataset,
        SequentialDataset,
        create_dataloader,
    )

    results: dict[int, float] = {}
    for freq_idx, freq_hz in enumerate(FREQUENCY_LABELS):
        subset = [e for e in entries if int(np.argmax(e["frequency_label"])) == freq_idx]
        if not subset:
            continue
        ds_cls = MLPDataset if model_type == "mlp" else SequentialDataset
        ds = ds_cls(subset)
        loader = create_dataloader(ds, batch_size=config["training"]["batch_size"], shuffle=False)
        criterion = nn.MSELoss()
        results[freq_hz] = evaluate(model, loader, criterion)
    return results


def build_comparison_table(results: dict[str, dict[str, float]]) -> str:
    """Build a Markdown comparison table.

    Parameters
    ----------
    results : dict
        ``{model_type: {"train": mse, "val": mse, "test": mse}}``.

    Returns
    -------
    str
        Markdown-formatted table.
    """
    lines = ["| Model | Train MSE | Val MSE | Test MSE |",
             "|-------|-----------|---------|----------|"]
    for name, mse in results.items():
        lines.append(
            f"| {name.upper()} | {mse['train']:.6f} | "
            f"{mse['val']:.6f} | {mse['test']:.6f} |"
        )
    return "\n".join(lines)
