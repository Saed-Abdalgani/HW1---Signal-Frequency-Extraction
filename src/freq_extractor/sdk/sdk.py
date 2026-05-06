"""FreqExtractorSDK — single entry point for all business logic.

External callers interact only through this class (PLAN §1, ADR-4).
"""

from __future__ import annotations

import logging
from typing import Any

from torch import nn

from freq_extractor.sdk.sdk_base import FreqExtractorSDKBase
from freq_extractor.sdk.sdk_train_eval import evaluate_model, train_model

logger = logging.getLogger("freq_extractor.sdk")


class FreqExtractorSDK(FreqExtractorSDKBase):
    """Façade over all freq_extractor services."""

    def generate_data(self) -> tuple[list, list, list]:
        """Generate and persist train/val/test dataset entries."""
        from freq_extractor.services.data_pipeline import build_full_dataset
        from freq_extractor.services.training_helpers import set_all_seeds

        set_all_seeds(self.seed)
        train, val, test = build_full_dataset(self.config, self.seed)
        logger.info("Dataset generated: %d/%d/%d (train/val/test)", len(train), len(val), len(test))
        return train, val, test

    def train(
        self, model_type: str, train_entries: list, val_entries: list,
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Train a single model and return it with its history."""
        return train_model(self, model_type, train_entries, val_entries)

    def evaluate(
        self, model: nn.Module, model_type: str,
        train_entries: list, val_entries: list, test_entries: list,
    ) -> dict[str, float]:
        """Evaluate a trained model on all splits."""
        return evaluate_model(self, model, model_type, train_entries, val_entries, test_entries)

    def run_all(self) -> dict[str, Any]:
        """Execute generate, train, evaluate, and plot pipeline."""
        from freq_extractor.sdk.sdk_runner import run_full_pipeline

        return run_full_pipeline(self)

    def launch_ui(self, port: int = 8050, *, debug: bool = True) -> None:
        """Launch the Sinusoid Explorer interactive dashboard."""
        from freq_extractor.services.ui_service import UIService

        UIService(config=self.config).launch_ui(port=port, debug=debug)
