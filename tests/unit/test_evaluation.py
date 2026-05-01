"""Tests for EvaluationService — compute_split_mse, per_frequency, comparison table.

Validates metrics computation and markdown table generation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from freq_extractor.services.evaluation_service import (
    build_comparison_table,
    compute_per_frequency_mse,
    compute_split_mse,
)


def _trivial_loader(n: int = 32) -> DataLoader:
    """Create a trivial DataLoader for metric tests."""
    x = torch.randn(n, 14)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(x, y), batch_size=16)


class TestComputeSplitMSE:
    """Split-level MSE computation tests."""

    def test_returns_all_splits(self) -> None:
        """compute_split_mse returns dict with train, val, test keys."""
        model = nn.Sequential(nn.Linear(14, 1))
        loaders = {
            "train": _trivial_loader(),
            "val": _trivial_loader(),
            "test": _trivial_loader(),
        }
        result = compute_split_mse(model, loaders)
        assert "train" in result
        assert "val" in result
        assert "test" in result

    def test_mse_non_negative(self) -> None:
        """All MSE values are non-negative."""
        model = nn.Sequential(nn.Linear(14, 1))
        loaders = {"train": _trivial_loader(), "val": _trivial_loader()}
        result = compute_split_mse(model, loaders)
        for v in result.values():
            assert v >= 0


class TestPerFrequencyMSE:
    """Per-frequency MSE computation tests."""

    def test_returns_dict(self, small_entries, sample_config) -> None:
        """compute_per_frequency_mse returns dict mapping freq → MSE."""
        from freq_extractor.services.mlp_model import MLPModel

        model = MLPModel(window_size=10, hidden_sizes=[16, 32, 16])
        result = compute_per_frequency_mse(model, small_entries, "mlp", sample_config)
        assert len(result) == 4
        for freq, mse in result.items():
            assert mse >= 0


class TestComparisonTable:
    """Markdown comparison table generation tests."""

    def test_table_format(self) -> None:
        """Table contains header row and model rows."""
        results = {
            "mlp": {"train": 0.01, "val": 0.02, "test": 0.03},
            "rnn": {"train": 0.005, "val": 0.01, "test": 0.015},
        }
        table = build_comparison_table(results)
        assert "| Model |" in table
        assert "MLP" in table
        assert "RNN" in table
        lines = table.strip().split("\n")
        assert len(lines) == 4

    def test_values_present(self) -> None:
        """Numeric MSE values appear in the table."""
        results = {"lstm": {"train": 0.001, "val": 0.002, "test": 0.003}}
        table = build_comparison_table(results)
        assert "0.001000" in table
        assert "LSTM" in table
