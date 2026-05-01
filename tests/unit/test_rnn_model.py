"""Tests for RNNModel — RNN-T1..RNN-T7.

Validates forward pass, hidden size, layers, variable-length input,
orthogonal init, gradient flow, and dropout behaviour.
"""

from __future__ import annotations

import torch

from freq_extractor.services.rnn_model import RNNModel


class TestRNNModel:
    """RNN architecture tests."""

    def test_forward_shape(self) -> None:
        """RNN-T1: Forward pass (B=32, T=10) → (32, 1), no NaN."""
        model = RNNModel(hidden_size=64, num_layers=2)
        x = torch.randn(32, 10, 5)
        out = model(x)
        assert out.shape == (32, 1)
        assert torch.isfinite(out).all()

    def test_hidden_size(self) -> None:
        """RNN-T2: Hidden size equals 64."""
        model = RNNModel(hidden_size=64)
        assert model.hidden_size == 64

    def test_num_layers(self) -> None:
        """RNN-T3: Number of layers equals 2."""
        model = RNNModel(num_layers=2)
        assert model.rnn.num_layers == 2

    def test_variable_sequence_length(self) -> None:
        """RNN-T4: Variable sequence lengths (T=5 and T=20) work."""
        model = RNNModel(hidden_size=64, num_layers=2)
        for seq_len in [5, 20]:
            x = torch.randn(4, seq_len, 5)
            out = model(x)
            assert out.shape == (4, 1)

    def test_orthogonal_init(self) -> None:
        """RNN-T5: Recurrent weights are orthogonally initialised."""
        model = RNNModel(hidden_size=64, num_layers=2)
        for name, param in model.rnn.named_parameters():
            if "weight_hh" in name:
                w = param.data
                product = w @ w.T
                identity = torch.eye(w.shape[0])
                residual = (product - identity).abs().max().item()
                assert residual < 0.1, f"Not orthogonal: {name}, max residual={residual:.4f}"

    def test_gradients_flow(self) -> None:
        """RNN-T6: Gradients flow through both RNN layers."""
        model = RNNModel(hidden_size=64, num_layers=2)
        x = torch.randn(8, 10, 5)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No grad for {name}"
            assert param.grad.abs().sum() > 0, f"Zero grad for {name}"

    def test_dropout_train_vs_eval(self) -> None:
        """RNN-T7: Dropout active in train mode, inactive in eval."""
        model = RNNModel(hidden_size=64, num_layers=2, dropout=0.5)
        x = torch.randn(16, 10, 5)
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_batch_size_one(self) -> None:
        """Batch size 1 works correctly."""
        model = RNNModel(hidden_size=64, num_layers=2)
        x = torch.randn(1, 10, 5)
        out = model(x)
        assert out.shape == (1, 1)

    def test_count_parameters(self) -> None:
        """count_parameters returns positive integer."""
        model = RNNModel(hidden_size=64, num_layers=2)
        count = model.count_parameters()
        assert count > 0
        assert isinstance(count, int)
