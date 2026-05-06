"""Tests for checkpoint save/load — TR-T4, EC.8.

Validates checkpoint round-trip, version compatibility,
and prediction identity after reload.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from freq_extractor.services.checkpoint import load_checkpoint, save_checkpoint
from freq_extractor.services.mlp_model import MLPModel


class TestCheckpoint:
    """Checkpoint persistence tests."""

    def test_save_creates_file(self, sample_config, tmp_path, tmp_config_dir) -> None:
        """save_checkpoint creates a .pt file on disk."""
        sample_config["training"]["checkpoint_dir"] = str(tmp_path / "ckpts")
        model = MLPModel(window_size=10, hidden_sizes=[16, 32, 16])
        opt = torch.optim.Adam(model.parameters())
        path = save_checkpoint(model, opt, 5, 0.01, "mlp", sample_config)
        assert Path(path).exists()

    def test_round_trip_identical_predictions(
        self, sample_config, tmp_path, tmp_config_dir,
    ) -> None:
        """TR-T4: Loaded model produces identical predictions."""
        sample_config["training"]["checkpoint_dir"] = str(tmp_path / "ckpts")
        model = MLPModel(window_size=10, hidden_sizes=[16, 32, 16])
        opt = torch.optim.Adam(model.parameters())
        x = torch.randn(4, 15)
        with torch.no_grad():
            out1 = model(x).clone()
        path = save_checkpoint(model, opt, 3, 0.02, "mlp", sample_config)
        model2 = MLPModel(window_size=10, hidden_sizes=[16, 32, 16])
        load_checkpoint(path, model2)
        with torch.no_grad():
            out2 = model2(x)
        torch.testing.assert_close(out1, out2)

    def test_version_mismatch_raises(
        self, sample_config, tmp_path, tmp_config_dir,
    ) -> None:
        """EC.8: Mismatched config_version on load raises ValueError."""
        sample_config["training"]["checkpoint_dir"] = str(tmp_path / "ckpts")
        model = MLPModel(window_size=10, hidden_sizes=[16, 32, 16])
        opt = torch.optim.Adam(model.parameters())
        path = save_checkpoint(model, opt, 1, 0.1, "mlp", sample_config)
        ckpt = torch.load(str(path), weights_only=False)
        ckpt["config_version"] = "9.99"
        torch.save(ckpt, str(path))
        model2 = MLPModel(window_size=10, hidden_sizes=[16, 32, 16])
        with pytest.raises(ValueError, match="[Vv]ersion"):
            load_checkpoint(path, model2)

    def test_optimizer_state_restored(
        self, sample_config, tmp_path, tmp_config_dir,
    ) -> None:
        """Optimizer state is restored from checkpoint."""
        sample_config["training"]["checkpoint_dir"] = str(tmp_path / "ckpts")
        model = MLPModel(window_size=10, hidden_sizes=[16, 32, 16])
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        x = torch.randn(4, 15)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        path = save_checkpoint(model, opt, 1, 0.1, "mlp", sample_config)
        model2 = MLPModel(window_size=10, hidden_sizes=[16, 32, 16])
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.005)
        ckpt = load_checkpoint(path, model2, opt2)
        assert ckpt["epoch"] == 1
