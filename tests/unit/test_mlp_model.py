"""Tests for MLPModel — MLP-T1..MLP-T7.

Validates forward pass shapes, parameter count, gradient flow,
Tanh activation presence, input validation, and serialisation.
"""

from __future__ import annotations

import torch
import pytest
from torch import nn

from freq_extractor.services.mlp_model import MLPModel


class TestMLPModel:
    """MLP architecture tests."""

    def test_forward_shape_batch32(self) -> None:
        """MLP-T1: Forward pass (B=32) → output shape (32, 1), no NaN."""
        model = MLPModel(window_size=10, hidden_sizes=[64, 128, 64])
        x = torch.randn(32, 14)
        out = model(x)
        assert out.shape == (32, 1)
        assert torch.isfinite(out).all()

    def test_forward_batch1(self) -> None:
        """MLP-T2: Batch size 1 works without crash."""
        model = MLPModel(window_size=10)
        x = torch.randn(1, 14)
        out = model(x)
        assert out.shape == (1, 1)

    def test_parameter_count(self) -> None:
        """MLP-T3: Parameter count ≈ 17,601 within 5%."""
        model = MLPModel(window_size=10, hidden_sizes=[64, 128, 64])
        count = model.count_parameters()
        expected = 17_601
        assert abs(count - expected) / expected < 0.05, f"Got {count}, expected ~{expected}"

    def test_gradients_flow(self) -> None:
        """MLP-T4: Gradients flow through all linear layers."""
        model = MLPModel(window_size=10, hidden_sizes=[64, 128, 64])
        x = torch.randn(8, 14)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No grad for {name}"
            assert param.grad.abs().sum() > 0, f"Zero grad for {name}"

    def test_tanh_activation_present(self) -> None:
        """MLP-T5: Tanh activation is used in the network."""
        model = MLPModel(window_size=10, hidden_sizes=[64, 128, 64])
        tanh_count = sum(1 for m in model.modules() if isinstance(m, nn.Tanh))
        assert tanh_count >= 3, f"Expected ≥3 Tanh layers, found {tanh_count}"

    def test_wrong_input_dim_raises(self) -> None:
        """MLP-T6: Wrong input dimension raises RuntimeError."""
        model = MLPModel(window_size=10, hidden_sizes=[64, 128, 64])
        x = torch.randn(8, 13)
        with pytest.raises(RuntimeError):
            model(x)

    def test_serialise_deserialise(self, tmp_path) -> None:
        """MLP-T7: Save + load produces identical output."""
        model = MLPModel(window_size=10, hidden_sizes=[64, 128, 64])
        x = torch.randn(4, 14)
        out1 = model(x)
        path = tmp_path / "mlp.pt"
        torch.save(model.state_dict(), str(path))
        model2 = MLPModel(window_size=10, hidden_sizes=[64, 128, 64])
        model2.load_state_dict(torch.load(str(path), weights_only=True))
        out2 = model2(x)
        torch.testing.assert_close(out1, out2)

    def test_custom_hidden_sizes(self) -> None:
        """Custom hidden layer widths produce valid output."""
        model = MLPModel(window_size=10, hidden_sizes=[32, 16])
        x = torch.randn(4, 14)
        out = model(x)
        assert out.shape == (4, 1)
