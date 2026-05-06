"""Model factory for creating neural network architectures by name.

Provides a single entry point ``create_model(model_type, config)``
that returns the appropriate ``nn.Module`` subclass, fully configured
from ``config/setup.json``.

References
----------
- PLAN §6, PRD_mod §9 (MD-T6)
"""

from __future__ import annotations

import logging
from typing import Any

from torch import nn

from freq_extractor.constants import MODEL_TYPES, SEQ_INPUT_SIZE
from freq_extractor.services.lstm_model import LSTMModel
from freq_extractor.services.mlp_model import MLPModel
from freq_extractor.services.rnn_model import RNNModel

logger = logging.getLogger("freq_extractor.model_factory")


class ModelFactory:
    """Factory for instantiating frequency-extraction models.

    All models are created with parameters drawn from ``config/setup.json``
    so that no caller needs to hard-code layer sizes or dropout values.
    """

    @staticmethod
    def create_model(
        model_type: str,
        config: dict[str, Any] | None = None,
    ) -> nn.Module:
        """Create and return a model instance.

        Parameters
        ----------
        model_type : str
            One of ``"mlp"``, ``"rnn"``, or ``"lstm"`` (case-insensitive).
        config : dict, optional
            Parsed ``config/setup.json``.  If *None*, defaults are used.

        Returns
        -------
        nn.Module
            Configured model instance.

        Raises
        ------
        ValueError
            If *model_type* is not in ``MODEL_TYPES``.
        """
        model_type = model_type.lower().strip()
        if model_type not in MODEL_TYPES:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Must be one of {MODEL_TYPES}."
            )

        sig_cfg = config.get("signal", {}) if config else {}
        mod_cfg = config.get("model", {}) if config else {}

        window_size = sig_cfg.get("window_size", 10)
        hidden_size = mod_cfg.get("hidden_size", 64)
        num_layers = mod_cfg.get("num_layers", 2)
        dropout = mod_cfg.get("dropout", 0.1)
        mlp_hidden = mod_cfg.get("mlp_hidden_sizes", [64, 128, 64])

        if model_type == "mlp":
            model = MLPModel(
                window_size=window_size,
                hidden_sizes=mlp_hidden,
            )
        elif model_type == "rnn":
            model = RNNModel(
                input_size=SEQ_INPUT_SIZE,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:  # lstm
            model = LSTMModel(
                input_size=SEQ_INPUT_SIZE,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )

        logger.info(
            "Created %s model (%d parameters)",
            model_type.upper(),
            model.count_parameters(),
        )
        return model
