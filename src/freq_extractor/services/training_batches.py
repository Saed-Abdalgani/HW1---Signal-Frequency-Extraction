"""Per-epoch training and evaluation batch loops."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def _module_device(model: nn.Module) -> torch.device:
    """Return the device used by the model parameters."""
    return next(model.parameters(), torch.empty(0)).device


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    grad_clip: float = 1.0,
) -> float:
    """Run one training epoch and return mean batch loss.

    Parameters
    ----------
    model, loader, optimizer, criterion
        Standard PyTorch training components.
    grad_clip
        Max norm for gradient clipping.

    Returns
    -------
    float
        Mean training loss.
    """
    model.train()
    device = _module_device(model)
    total_loss = 0.0
    n_batches = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    if n_batches == 0:
        raise ValueError("DataLoader is empty - cannot train on zero batches.")
    return total_loss / n_batches


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    """Evaluate model on a loader and return mean loss.

    Parameters
    ----------
    model, loader, criterion
        Model, data, and loss for evaluation.

    Returns
    -------
    float
        Mean validation loss.
    """
    model.eval()
    device = _module_device(model)
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)
