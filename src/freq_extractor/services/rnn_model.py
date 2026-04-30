"""Vanilla Recurrent Neural Network for frequency extraction.

Architecture: 2-layer RNN (input_size=5, hidden=64, dropout=0.1)
followed by a linear head that maps the final hidden state to a
scalar prediction.  Recurrent weights are orthogonally initialised.

References
----------
- PRD FR-4, PRD_mod §4, §7
"""

from __future__ import annotations

import torch
from torch import nn

from freq_extractor.constants import NUM_CLASSES


class RNNModel(nn.Module):
    """Vanilla RNN for next-sample prediction.

    Parameters
    ----------
    input_size : int
        Features per timestep (default ``1 + NUM_CLASSES = 5``).
    hidden_size : int
        RNN hidden dimension (default 64).
    num_layers : int
        Number of stacked RNN layers (default 2).
    dropout : float
        Dropout between RNN layers (default 0.1).
    """

    def __init__(
        self,
        input_size: int = 1 + NUM_CLASSES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialise RNN with orthogonal recurrent weights."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialisation to recurrent weight matrices."""
        for name, param in self.rnn.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the RNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, input_size)``.

        Returns
        -------
        torch.Tensor
            Predicted next clean sample, shape ``(B, 1)``.
        """
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size,
            device=x.device, dtype=x.dtype,
        )
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])

    def count_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns
        -------
        int
            Sum of ``numel()`` for all parameters with ``requires_grad``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
