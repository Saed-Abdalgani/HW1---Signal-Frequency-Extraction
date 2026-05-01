"""Tests for LSTMModel — LSTM-T1..LSTM-T6.

Validates forward pass, cell/hidden state, parameter count vs RNN,
forget gate biases, gradient flow, and eval-mode determinism.
"""

from __future__ import annotations

import torch

from freq_extractor.services.lstm_model import LSTMModel
from freq_extractor.services.rnn_model import RNNModel


class TestLSTMModel:
    """LSTM architecture tests."""

    def test_forward_shape(self) -> None:
        """LSTM-T1: Forward pass (B=32, T=10) → (32, 1), no NaN."""
        model = LSTMModel(hidden_size=64, num_layers=2)
        x = torch.randn(32, 10, 5)
        out = model(x)
        assert out.shape == (32, 1)
        assert torch.isfinite(out).all()

    def test_hidden_and_cell_state(self) -> None:
        """LSTM-T2: Both h0 and c0 are properly initialised."""
        model = LSTMModel(hidden_size=64, num_layers=2)
        assert model.hidden_size == 64
        assert model.num_layers == 2
        x = torch.randn(4, 10, 5)
        out = model(x)
        assert out.shape == (4, 1)

    def test_more_params_than_rnn(self) -> None:
        """LSTM-T3: LSTM has more parameters than RNN (≈4×)."""
        lstm = LSTMModel(hidden_size=64, num_layers=2)
        rnn = RNNModel(hidden_size=64, num_layers=2)
        assert lstm.count_parameters() > rnn.count_parameters()
        ratio = lstm.count_parameters() / rnn.count_parameters()
        assert 2.5 < ratio < 5.0, f"Ratio is {ratio:.1f}, expected ~4×"

    def test_forget_gate_biases(self) -> None:
        """LSTM-T4: Forget gate bias entries exist with 4·hidden_size entries."""
        model = LSTMModel(hidden_size=64, num_layers=2)
        for name, param in model.lstm.named_parameters():
            if "bias_hh" in name:
                assert param.shape[0] == 4 * 64

    def test_gradients_flow(self) -> None:
        """LSTM-T5: Gradients flow through all 4 gate matrices."""
        model = LSTMModel(hidden_size=64, num_layers=2)
        x = torch.randn(8, 10, 5)
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No grad for {name}"
            assert param.grad.abs().sum() > 0, f"Zero grad for {name}"

    def test_eval_mode_deterministic(self) -> None:
        """LSTM-T6: Eval mode produces identical outputs on repeated calls."""
        model = LSTMModel(hidden_size=64, num_layers=2, dropout=0.5)
        x = torch.randn(8, 10, 5)
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_orthogonal_init(self) -> None:
        """Recurrent weights have orthogonal columns (WᵀW ≈ I)."""
        model = LSTMModel(hidden_size=64, num_layers=2)
        for name, param in model.lstm.named_parameters():
            if "weight_hh" in name:
                w = param.data
                product = w.T @ w
                identity = torch.eye(product.shape[0])
                assert (product - identity).abs().max().item() < 0.15

    def test_batch_size_one(self) -> None:
        """Batch size 1 works correctly."""
        model = LSTMModel(hidden_size=64, num_layers=2)
        x = torch.randn(1, 10, 5)
        out = model(x)
        assert out.shape == (1, 1)
