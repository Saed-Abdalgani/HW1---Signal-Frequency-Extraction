"""Fully Connected Multi-Layer Perceptron for frequency extraction.

Architecture: 14 → 64 → 128 → 64 → 1 with Tanh activations.
The MLP treats the 10-sample sliding window as a flat feature vector
concatenated with a 4-dim one-hot frequency label (total input = 14).

References
----------
- PRD FR-3, PRD_mod §3
"""

from __future__ import annotations

import torch
from torch import nn

from freq_extractor.constants import NUM_CLASSES


class MLPModel(nn.Module):
    """Multi-Layer Perceptron for next-sample prediction.

    Parameters
    ----------
    window_size : int
        Number of noisy samples in the input window (default 10).
    hidden_sizes : list[int]
        Sizes of hidden layers (default [64, 128, 64]).

    Attributes
    ----------
    network : nn.Sequential
        The full feed-forward network including activations.
    """

    def __init__(
        self,
        window_size: int = 10,
        hidden_sizes: list[int] | None = None,
    ) -> None:
        """Initialise MLP with configurable layer widths."""
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 128, 64]

        input_dim = window_size + NUM_CLASSES  # 10 + 4 = 14

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, window_size + NUM_CLASSES)``.

        Returns
        -------
        torch.Tensor
            Predicted next clean sample, shape ``(B, 1)``.
        """
        return self.network(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns
        -------
        int
            Sum of ``numel()`` for all parameters with ``requires_grad``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
