"""Training helpers — early stopping and seed management.

Separated from ``training_service.py`` to comply with the ≤ 145 code-line
constraint (NFR-3 v1.10).

References
----------
- PRD_tr §§6, 8
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
from torch import nn


class EarlyStopping:
    """Stop training when validation loss fails to improve.

    Parameters
    ----------
    patience : int
        Epochs to wait for improvement before stopping.
    min_delta : float
        Minimum decrease in val loss to qualify as improvement.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-6) -> None:
        """Initialise early-stopping tracker."""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.best_state: dict[str, Any] | None = None
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        """Record validation loss and update stopping decision."""
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module) -> None:
        """Load the best model weights back into *model*."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Seeds: ``torch``, ``numpy``, ``random``, ``cudnn``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
