"""FreqExtractorSDK — single entry point for all business logic.

External callers (including ``src/main.py``) interact **only** through
this class.  Direct service imports are prohibited (PLAN §1, ADR-4).

The SDK lazily loads the configuration, seeds the RNG, respects the
``FREQ_EXTRACTOR_FORCE_CPU`` environment variable, and delegates to
service modules for data generation, training, evaluation, and the
interactive dashboard.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from freq_extractor.constants import SPLIT_NAMES
from freq_extractor.sdk.sdk_base import FreqExtractorSDKBase

logger = logging.getLogger("freq_extractor.sdk")

class FreqExtractorSDK(FreqExtractorSDKBase):
    """Façade over all freq_extractor services.

    Examples
    --------
    >>> sdk = FreqExtractorSDK()
    >>> sdk.run_all()
    """

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def generate_data(self) -> tuple[list, list, list]:
        """Generate and persist the canonical train/val/test dataset.

        Returns
        -------
        tuple[list, list, list]
            ``(train, val, test)`` lists of normalised dataset entries.
        """
        from freq_extractor.services.data_pipeline import build_full_dataset
        from freq_extractor.services.training_helpers import set_all_seeds

        set_all_seeds(self.seed)
        train, val, test = build_full_dataset(self.config, self.seed)
        logger.info(
            "Dataset generated: %d/%d/%d (train/val/test)",
            len(train), len(val), len(test),
        )
        return train, val, test

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self, model_type: str, train_entries: list, val_entries: list,
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Train a single model and return it with its history.

        Parameters
        ----------
        model_type : str
            ``"mlp"``, ``"rnn"``, or ``"lstm"``.
        train_entries, val_entries : list
            Normalised dataset entries for the respective splits.

        Returns
        -------
        tuple[nn.Module, dict]
            ``(trained_model, history_dict)``.
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
        bs = self.config["training"]["batch_size"]

        def run_on(device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
            set_all_seeds(self.seed)
            model = ModelFactory.create_model(model_type, self.config).to(device)
            train_loader = create_dataloader(ds_cls(train_entries), bs, shuffle=True,
                                             seed=self.seed)
            val_loader = create_dataloader(ds_cls(val_entries), bs, shuffle=False,
                                           seed=self.seed)
            return model, _train(model, train_loader, val_loader, self.config, model_type)

        try:
            model, history = run_on(self.device)
        except RuntimeError as exc:
            if self.device.type != "cuda" or "out of memory" not in str(exc).lower():
                raise
            logger.warning("CUDA OOM during %s training; retrying on CPU.",
                           model_type.upper())
            torch.cuda.empty_cache()
            self._device = torch.device("cpu")
            model, history = run_on(self.device)
        logger.info("[%s] Training complete — best_val_mse=%.6f", model_type.upper(),
                    history["best_val_mse"])
        return model, history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, model: nn.Module, model_type: str,
        train_entries: list, val_entries: list, test_entries: list,
    ) -> dict[str, float]:
        """Evaluate a trained model on all splits.

        Returns
        -------
        dict[str, float]
            ``{"train": mse, "val": mse, "test": mse}``.
        """
        from freq_extractor.services.datasets import (
            MLPDataset,
            SequentialDataset,
            create_dataloader,
        )
        from freq_extractor.services.evaluation_service import compute_split_mse

        ds_cls = MLPDataset if model_type == "mlp" else SequentialDataset
        bs = self.config["training"]["batch_size"]
        loaders = {}
        for name, entries in zip(SPLIT_NAMES, [train_entries, val_entries, test_entries],
                                 strict=False):
            loaders[name] = create_dataloader(ds_cls(entries), bs, shuffle=False, seed=self.seed)
        return compute_split_mse(model, loaders)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_all(self) -> dict[str, Any]:
        """Execute the full pipeline: generate → train → evaluate → plot.

        Returns
        -------
        dict
            Summary with keys ``models``, ``histories``, ``metrics``,
            ``comparison_table``.
        """
        from freq_extractor.sdk.sdk_runner import run_full_pipeline

        return run_full_pipeline(self)

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def launch_ui(self, port: int = 8050) -> None:
        """Launch the Sinusoid Explorer interactive dashboard."""
        from freq_extractor.services.ui_service import UIService

        svc = UIService(config=self.config)
        svc.launch_ui(port=port)
