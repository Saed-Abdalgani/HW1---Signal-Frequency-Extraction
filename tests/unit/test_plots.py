"""Tests for plotting modules — plot_helpers and plot_analysis.

Validates that plots save to disk without errors.
"""

from __future__ import annotations

from freq_extractor.services.plot_analysis import (
    plot_noise_robustness,
    plot_per_frequency_mse,
    plot_signal_examples,
)
from freq_extractor.services.plot_helpers import plot_predictions, plot_training_curves


class TestPlotHelpers:
    """Plot saving tests."""

    def test_training_curves(self, tmp_path, tmp_config_dir) -> None:
        """plot_training_curves saves a PNG file."""
        histories = {
            "mlp": {"train_losses": [1.0, 0.5, 0.3], "val_losses": [1.1, 0.6, 0.4]},
            "rnn": {"train_losses": [0.9, 0.4, 0.2], "val_losses": [1.0, 0.5, 0.3]},
        }
        path = tmp_path / "curves.png"
        plot_training_curves(histories, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_predictions_mlp(self, tmp_path, small_entries, tmp_config_dir) -> None:
        """plot_predictions saves a PNG for MLP model."""
        from freq_extractor.services.mlp_model import MLPModel

        model = MLPModel(window_size=10, hidden_sizes=[16, 32, 16])
        path = tmp_path / "pred_mlp.png"
        plot_predictions(model, small_entries, "mlp", "mlp", path, n_examples=10)
        assert path.exists()

    def test_predictions_rnn(self, tmp_path, small_entries, tmp_config_dir) -> None:
        """plot_predictions saves a PNG for RNN model."""
        from freq_extractor.services.rnn_model import RNNModel

        model = RNNModel(hidden_size=16, num_layers=1, dropout=0.0)
        path = tmp_path / "pred_rnn.png"
        plot_predictions(model, small_entries, "rnn", "rnn", path, n_examples=10)
        assert path.exists()


class TestPlotAnalysis:
    """Analysis plot tests."""

    def test_noise_robustness(self, tmp_path, tmp_config_dir) -> None:
        """plot_noise_robustness saves a PNG."""
        results = {
            "mlp": {0.05: 0.01, 0.10: 0.02, 0.20: 0.05},
            "rnn": {0.05: 0.005, 0.10: 0.01, 0.20: 0.03},
        }
        path = tmp_path / "robustness.png"
        plot_noise_robustness(results, path)
        assert path.exists()

    def test_per_frequency_mse(self, tmp_path, tmp_config_dir) -> None:
        """plot_per_frequency_mse saves a PNG."""
        results = {
            "mlp": {5: 0.01, 15: 0.02, 30: 0.03, 50: 0.04},
            "rnn": {5: 0.005, 15: 0.01, 30: 0.02, 50: 0.03},
        }
        path = tmp_path / "per_freq.png"
        plot_per_frequency_mse(results, path)
        assert path.exists()

    def test_signal_examples(self, tmp_path, sample_config, tmp_config_dir) -> None:
        """plot_signal_examples saves a PNG."""
        path = tmp_path / "signals.png"
        plot_signal_examples(sample_config, path)
        assert path.exists()
