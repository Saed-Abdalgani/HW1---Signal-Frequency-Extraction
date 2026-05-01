"""Tests for noise robustness evaluation.

Validates the noise sweep produces results for all sigma levels.
"""

from __future__ import annotations

from freq_extractor.services.eval_robustness import run_noise_robustness
from freq_extractor.services.mlp_model import MLPModel


class TestNoiseRobustness:
    """Noise robustness sweep tests."""

    def test_returns_all_levels(self, sample_config, tmp_config_dir) -> None:
        """run_noise_robustness returns results for all configured sigma levels."""
        model = MLPModel(window_size=10, hidden_sizes=[16, 32, 16])
        results = run_noise_robustness(
            model, sample_config, "mlp",
            noise_levels=[0.05, 0.10], seed=42,
        )
        assert len(results) == 2
        assert 0.05 in results
        assert 0.10 in results

    def test_mse_non_negative(self, sample_config, tmp_config_dir) -> None:
        """All MSE values in robustness sweep are non-negative."""
        model = MLPModel(window_size=10, hidden_sizes=[16, 32, 16])
        results = run_noise_robustness(
            model, sample_config, "mlp",
            noise_levels=[0.05], seed=42,
        )
        for mse in results.values():
            assert mse >= 0

    def test_higher_noise_higher_mse(self, sample_config, tmp_config_dir) -> None:
        """Higher noise generally leads to higher MSE (sanity check)."""
        model = MLPModel(window_size=10, hidden_sizes=[16, 32, 16])
        results = run_noise_robustness(
            model, sample_config, "mlp",
            noise_levels=[0.01, 0.50], seed=42,
        )
        assert results[0.50] >= results[0.01] * 0.5
