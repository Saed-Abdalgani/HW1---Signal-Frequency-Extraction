"""Checkpoint save/load utilities for model persistence.

All file I/O is routed through the ``file_io`` gatekeeper to enforce
rate limiting, retry logic, and unified logging (ADR-4).

References
----------
- PLAN §5, PRD_tr §9
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from freq_extractor.constants import MODEL_CHECKPOINT_VERSION, PT_EXT
from freq_extractor.shared import get_gatekeeper
from freq_extractor.shared.version import validate_config_version

logger = logging.getLogger("freq_extractor.checkpoint")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_mse: float,
    model_type: str,
    config: dict[str, Any],
) -> Path:
    """Save a training checkpoint via the gatekeeper.

    Parameters
    ----------
    model : nn.Module
        Trained model whose state to persist.
    optimizer : torch.optim.Optimizer
        Optimizer whose state to persist (for resumption).
    epoch : int
        Best epoch number.
    val_mse : float
        Validation MSE at the best epoch.
    model_type : str
        Model identifier (``"mlp"``, ``"rnn"``, ``"lstm"``).
    config : dict
        Full setup config (for the checkpoint dir path).

    Returns
    -------
    Path
        Absolute path to the saved checkpoint file.
    """
    ckpt_dir = Path(config["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{model_type}_best{PT_EXT}"

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_mse": val_mse,
        "config_version": MODEL_CHECKPOINT_VERSION,
        "model_type": model_type,
    }

    def _do_save() -> None:
        torch.save(payload, str(path))

    get_gatekeeper("checkpoint").execute(_do_save)
    logger.info("Checkpoint saved -> %s (epoch %d, val_mse=%.6f)", path, epoch, val_mse)
    return path


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and restore model (+ optional optimizer) state.

    Parameters
    ----------
    path : Path | str
        Path to the ``.pt`` checkpoint file.
    model : nn.Module
        Model to load weights into.
    optimizer : torch.optim.Optimizer, optional
        Optimizer to restore state into.

    Returns
    -------
    dict
        The full checkpoint dict (epoch, val_mse, etc.).

    Raises
    ------
    ValueError
        If the checkpoint's ``config_version`` is incompatible.
    """
    path = Path(path)

    def _do_load() -> dict:
        return torch.load(str(path), map_location="cpu", weights_only=False)

    ckpt: dict[str, Any] = get_gatekeeper("checkpoint").execute(_do_load)

    # Validate version compatibility
    validate_config_version(ckpt["config_version"], f"checkpoint:{path.name}")

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    logger.info(
        "Checkpoint loaded <- %s (epoch %d, val_mse=%.6f)",
        path, ckpt["epoch"], ckpt["val_mse"],
    )
    return ckpt
