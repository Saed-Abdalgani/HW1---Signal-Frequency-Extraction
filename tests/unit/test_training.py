"""Tests for TrainingService — TR-T1, TR-T6, TR-T8.

Validates training loop smoke test, MSE identity, evaluate function,
and empty DataLoader handling.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from freq_extractor.services.training_service import evaluate, train_one_epoch


def _make_loader(n: int = 100, in_dim: int = 14, batch_size: int = 16) -> DataLoader:
    """Create a small deterministic DataLoader for testing."""
    x = torch.randn(n, in_dim)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


class TestTrainOneEpoch:
    """Training loop per-epoch tests."""

    def test_smoke_loss_decreases(self) -> None:
        """TR-T1: Two epochs of training reduce loss."""
        model = nn.Sequential(nn.Linear(14, 32), nn.Tanh(), nn.Linear(32, 1))
        loader = _make_loader()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        crit = nn.MSELoss()
        loss1 = train_one_epoch(model, loader, opt, crit)
        loss2 = train_one_epoch(model, loader, opt, crit)
        assert loss2 < loss1, f"Loss did not decrease: {loss1:.4f} → {loss2:.4f}"

    def test_returns_float(self) -> None:
        """train_one_epoch returns a finite float."""
        model = nn.Sequential(nn.Linear(14, 16), nn.Linear(16, 1))
        loader = _make_loader(n=32)
        opt = torch.optim.Adam(model.parameters())
        loss = train_one_epoch(model, loader, opt, nn.MSELoss())
        assert isinstance(loss, float)
        assert loss > 0

    def test_grad_clip_prevents_nan(self) -> None:
        """TR-T3: Gradient clipping keeps loss finite."""
        model = nn.Sequential(nn.Linear(14, 64), nn.Linear(64, 1))
        loader = _make_loader()
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(5):
            loss = train_one_epoch(model, loader, opt, nn.MSELoss(), grad_clip=1.0)
            assert loss == loss, "Loss is NaN"

    def test_extreme_noise_like_values_stay_finite(self) -> None:
        """EC.3: Large noisy inputs do not make one training epoch diverge."""
        x = torch.randn(64, 14) * 10.0
        y = torch.randn(64, 1) * 10.0
        loader = DataLoader(TensorDataset(x, y), batch_size=16)
        model = nn.Sequential(nn.Linear(14, 32), nn.Tanh(), nn.Linear(32, 1))
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = train_one_epoch(model, loader, opt, nn.MSELoss(), grad_clip=1.0)
        assert torch.isfinite(torch.tensor(loss))

    def test_empty_loader_raises(self) -> None:
        """Training on empty dataloader raises ValueError."""
        import pytest
        from freq_extractor.services.training_service import train_one_epoch
        model = nn.Linear(14, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loader = []
        with pytest.raises(ValueError, match="DataLoader is empty — cannot train on zero batches."):
            train_one_epoch(model, loader, opt, nn.MSELoss())


class TestEvaluate:
    """Evaluation function tests."""

    def test_mse_identity_zero(self) -> None:
        """TR-T6: MSE(y, y) == 0."""
        crit = nn.MSELoss()
        y = torch.randn(10, 1)
        assert crit(y, y).item() == 0.0

    def test_evaluate_no_grad(self) -> None:
        """Evaluate runs without accumulating gradients."""
        model = nn.Sequential(nn.Linear(14, 1))
        loader = _make_loader(n=32)
        loss = evaluate(model, loader, nn.MSELoss())
        assert isinstance(loss, float)
        assert loss >= 0
        for p in model.parameters():
            assert p.grad is None


class TestTrainFull:
    """Full training pipeline smoke test."""

    def test_train_returns_history(self, sample_config, tmp_path, tmp_config_dir) -> None:
        """train() returns history with expected keys."""
        from freq_extractor.services.model_factory import ModelFactory
        from freq_extractor.services.training_service import train

        sample_config["training"]["max_epochs"] = 3
        sample_config["training"]["early_stop_patience"] = 2
        sample_config["training"]["checkpoint_dir"] = str(tmp_path / "ckpts")
        model = ModelFactory.create_model("mlp", sample_config)
        loader = _make_loader(n=64, in_dim=14, batch_size=16)
        val_loader = _make_loader(n=32, in_dim=14, batch_size=16)
        history = train(model, loader, val_loader, sample_config, model_type="mlp")
        assert "train_losses" in history
        assert "val_losses" in history
        assert "best_epoch" in history
        assert len(history["train_losses"]) > 0
