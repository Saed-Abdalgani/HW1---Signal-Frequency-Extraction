"""Training service — full training loop with LR schedule and early stop."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from freq_extractor.services.training_batches import evaluate, train_one_epoch
from freq_extractor.services.training_helpers import EarlyStopping, set_all_seeds

logger = logging.getLogger("freq_extractor.training")

__all__ = ["EarlyStopping", "set_all_seeds", "train_one_epoch", "evaluate", "train"]


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    model_type: str = "model",
) -> dict[str, Any]:
    """Train with early stopping, LR reduction, and checkpoint save."""
    from freq_extractor.services.checkpoint import save_checkpoint

    tc = config["training"]
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tc["learning_rate"],
        betas=(tc["adam_beta1"], tc["adam_beta2"]),
        eps=tc["adam_eps"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=tc["lr_reduce_factor"],
        patience=tc["lr_reduce_patience"], min_lr=tc["lr_min"],
    )
    stopper = EarlyStopping(tc["early_stop_patience"], tc["early_stop_min_delta"])

    history: dict[str, Any] = {"train_losses": [], "val_losses": []}
    for epoch in range(1, tc["max_epochs"] + 1):
        t_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, tc["grad_clip_max_norm"],
        )
        v_loss = evaluate(model, val_loader, criterion)
        history["train_losses"].append(t_loss)
        history["val_losses"].append(v_loss)
        scheduler.step(v_loss)
        stopper.step(v_loss, model)

        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "[%s] Epoch %3d | train=%.6f | val=%.6f | lr=%.2e",
            model_type.upper(), epoch, t_loss, v_loss, cur_lr,
        )

        if stopper.should_stop:
            logger.info("[%s] Early stopping at epoch %d", model_type.upper(), epoch)
            break

    stopper.restore_best(model)
    best_epoch = history["val_losses"].index(min(history["val_losses"])) + 1
    history["best_epoch"] = best_epoch
    history["best_val_mse"] = min(history["val_losses"])

    save_checkpoint(model, optimizer, best_epoch, history["best_val_mse"], model_type, config)
    return history
