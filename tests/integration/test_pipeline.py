"""Integration tests — end-to-end pipeline with tiny config.

Validates that the full data→train→evaluate pipeline works
for all model types using a minimal dataset.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from freq_extractor.sdk.sdk import FreqExtractorSDK
from freq_extractor.services.evaluation_service import build_comparison_table
from freq_extractor.services.model_factory import ModelFactory
from freq_extractor.services.training_service import train


class TestEndToEnd:
    """Full pipeline integration tests."""

    def test_mlp_pipeline(self, sample_config, tmp_path, tmp_config_dir) -> None:
        """6I.1: MLP end-to-end: build → train → evaluate."""
        sample_config["training"]["max_epochs"] = 3
        sample_config["training"]["early_stop_patience"] = 2
        sample_config["training"]["checkpoint_dir"] = str(tmp_path / "ckpts")

        sdk = FreqExtractorSDK()
        train_e, val_e, test_e = sdk.generate_data()

        model, history = sdk.train("mlp", train_e, val_e)
        assert len(history["train_losses"]) > 0

        metrics = sdk.evaluate(model, "mlp", train_e, val_e, test_e)
        assert metrics["test"] >= 0

    def test_rnn_pipeline(self, sample_config, tmp_path, tmp_config_dir) -> None:
        """6I.1: RNN end-to-end pipeline."""
        sample_config["training"]["max_epochs"] = 3
        sample_config["training"]["early_stop_patience"] = 2
        sample_config["training"]["checkpoint_dir"] = str(tmp_path / "ckpts")

        sdk = FreqExtractorSDK()
        train_e, val_e, _ = sdk.generate_data()

        model, history = sdk.train("rnn", train_e, val_e)
        assert history["best_val_mse"] >= 0

    def test_comparison_table(self) -> None:
        """6I.3: Comparison table generates valid markdown."""
        results = {
            "mlp": {"train": 0.01, "val": 0.02, "test": 0.03},
            "rnn": {"train": 0.005, "val": 0.01, "test": 0.02},
            "lstm": {"train": 0.003, "val": 0.008, "test": 0.015},
        }
        table = build_comparison_table(results)
        assert "MLP" in table
        assert "RNN" in table
        assert "LSTM" in table

    def test_checkpoint_exists_after_train(
        self, sample_config, tmp_path, tmp_config_dir,
    ) -> None:
        """6I.2: Checkpoint file exists after training."""
        ckpt_dir = tmp_path / "ckpts"
        sample_config["training"]["max_epochs"] = 2
        sample_config["training"]["early_stop_patience"] = 1
        sample_config["training"]["checkpoint_dir"] = str(ckpt_dir)
        train_loader = DataLoader(
            TensorDataset(torch.randn(64, 15), torch.randn(64, 1)), batch_size=32,
        )
        val_loader = DataLoader(
            TensorDataset(torch.randn(32, 15), torch.randn(32, 1)), batch_size=32,
        )
        model = ModelFactory.create_model("mlp", sample_config)
        train(model, train_loader, val_loader, sample_config, model_type="mlp")
        ckpt_files = list(ckpt_dir.glob("*.pt"))
        assert len(ckpt_files) >= 1

    def test_sdk_run_all(self, sample_config, tmp_path, tmp_config_dir) -> None:
        """Test full pipeline using sdk.run_all()."""
        sample_config["training"]["max_epochs"] = 1
        sample_config["training"]["early_stop_patience"] = 1
        sample_config["training"]["checkpoint_dir"] = str(tmp_path / "ckpts")
        sample_config["signal"]["duration_s"] = 0.5

        sdk = FreqExtractorSDK()
        results = sdk.run_all()
        assert "models" in results
        assert "metrics" in results
