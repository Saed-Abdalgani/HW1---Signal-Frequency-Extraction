"""Training and evaluation helpers for :class:`~freq_extractor.sdk.sdk.FreqExtractorSDK`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from freq_extractor.constants import SPLIT_NAMES

if TYPE_CHECKING:
    from freq_extractor.sdk.sdk_base import FreqExtractorSDKBase

logger = logging.getLogger("freq_extractor.sdk")


def train_model(
    sdk: FreqExtractorSDKBase,
    model_type: str,
    train_entries: list,
    val_entries: list,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train one model and retry on CPU if a CUDA OOM occurs.

    Parameters
    ----------
    sdk
        SDK providing ``seed``, ``config``, and ``device``.
    model_type
        ``"mlp"``, ``"rnn"``, or ``"lstm"``.
    train_entries, val_entries
        Normalised dataset entry lists.

    Returns
    -------
    tuple[nn.Module, dict[str, Any]]
        Trained module and its history dict.
    """
    from freq_extractor.services.datasets import (
        MLPDataset,
        SequentialDataset,
        create_dataloader,
    )
    from freq_extractor.services.model_factory import ModelFactory
    from freq_extractor.services.training_helpers import set_all_seeds
    from freq_extractor.services.training_service import train as _train

    ds_cls = MLPDataset if model_type == "mlp" else SequentialDataset
    bs = sdk.config["training"]["batch_size"]

    def run_on(device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
        set_all_seeds(sdk.seed)
        model = ModelFactory.create_model(model_type, sdk.config).to(device)
        train_loader = create_dataloader(ds_cls(train_entries), bs, shuffle=True, seed=sdk.seed)
        val_loader = create_dataloader(ds_cls(val_entries), bs, shuffle=False, seed=sdk.seed)
        return model, _train(model, train_loader, val_loader, sdk.config, model_type)

    try:
        model, history = run_on(sdk.device)
    except RuntimeError as exc:
        if sdk.device.type != "cuda" or "out of memory" not in str(exc).lower():
            raise
        logger.warning("CUDA OOM during %s training; retrying on CPU.", model_type.upper())
        torch.cuda.empty_cache()
        sdk._device = torch.device("cpu")
        model, history = run_on(sdk.device)
    logger.info("[%s] Training complete - best_val_mse=%.6f", model_type.upper(), history["best_val_mse"])
    return model, history


def evaluate_model(
    sdk: FreqExtractorSDKBase,
    model: nn.Module,
    model_type: str,
    train_entries: list,
    val_entries: list,
    test_entries: list,
) -> dict[str, float]:
    """Compute MSE on train, validation, and test splits.

    Parameters
    ----------
    sdk
        SDK providing ``config``, ``seed``, and loaders configuration.
    model, model_type
        Trained module and its architecture tag.
    train_entries, val_entries, test_entries
        Normalised splits.

    Returns
    -------
    dict[str, float]
        Per-split mean squared errors.
    """
    from freq_extractor.services.datasets import (
        MLPDataset,
        SequentialDataset,
        create_dataloader,
    )
    from freq_extractor.services.evaluation_service import compute_split_mse

    ds_cls = MLPDataset if model_type == "mlp" else SequentialDataset
    bs = sdk.config["training"]["batch_size"]
    loaders = {}
    for name, entries in zip(SPLIT_NAMES, [train_entries, val_entries, test_entries], strict=False):
        loaders[name] = create_dataloader(ds_cls(entries), bs, shuffle=False, seed=sdk.seed)
    return compute_split_mse(model, loaders)
