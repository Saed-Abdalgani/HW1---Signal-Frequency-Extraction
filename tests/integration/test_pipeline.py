"""Integration tests — end-to-end pipeline with tiny config.

Validates that the full data→train→evaluate pipeline works
for all model types using a minimal dataset.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from freq_extractor.services.data_pipeline import build_full_dataset
from freq_extractor.services.datasets import MLPDataset, SequentialDataset, create_dataloader
from freq_extractor.services.evaluation_service import build_comparison_table, compute_split_mse
from freq_extractor.services.model_factory import ModelFactory
from freq_extractor.services.training_service import evaluate, train


class TestEndToEnd:
    """Full pipeline integration tests."""

    def test_mlp_pipeline(self, sample_config, tmp_path, tmp_config_dir) -> None:
        """6I.1: MLP end-to-end: build → train → evaluate."""
        sample_config["training"]["max_epochs"] = 3
        sample_config["training"]["early_stop_patience"] = 2
        sample_config["training"]["checkpoint_dir"] = str(tmp_path / "ckpts")
        train_e, val_e, test_e = build_full_dataset(sample_config, seed=42)
        model = ModelFactory.create_model("mlp", sample_config)
        train_ds = MLPDataset(train_e)
        val_ds = MLPDataset(val_e)
        test_ds = MLPDataset(test_e)
        bs = sample_config["training"]["batch_size"]
        t_loader = create_dataloader(train_ds, batch_size=bs, shuffle=True)
        v_loader = create_dataloader(val_ds, batch_size=bs, shuffle=False)
        te_loader = create_dataloader(test_ds, batch_size=bs, shuffle=False)
        history = train(model, t_loader, v_loader, sample_config, model_type="mlp")
        assert len(history["train_losses"]) > 0
        test_mse = evaluate(model, te_loader, nn.MSELoss())
        assert test_mse >= 0

    def test_rnn_pipeline(self, sample_config, tmp_path, tmp_config_dir) -> None:
        """6I.1: RNN end-to-end pipeline."""
        sample_config["training"]["max_epochs"] = 3
        sample_config["training"]["early_stop_patience"] = 2
        sample_config["training"]["checkpoint_dir"] = str(tmp_path / "ckpts")
        train_e, val_e, _ = build_full_dataset(sample_config, seed=42)
        model = ModelFactory.create_model("rnn", sample_config)
        bs = sample_config["training"]["batch_size"]
        t_loader = create_dataloader(SequentialDataset(train_e), batch_size=bs)
        v_loader = create_dataloader(SequentialDataset(val_e), batch_size=bs, shuffle=False)
        history = train(model, t_loader, v_loader, sample_config, model_type="rnn")
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
            TensorDataset(torch.randn(64, 14), torch.randn(64, 1)), batch_size=32,
        )
        val_loader = DataLoader(
            TensorDataset(torch.randn(32, 14), torch.randn(32, 1)), batch_size=32,
        )
        model = ModelFactory.create_model("mlp", sample_config)
        train(model, train_loader, val_loader, sample_config, model_type="mlp")
        ckpt_files = list(ckpt_dir.glob("*.pt"))
        assert len(ckpt_files) >= 1
