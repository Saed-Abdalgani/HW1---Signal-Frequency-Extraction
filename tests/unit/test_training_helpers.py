"""Tests for EarlyStopping and set_all_seeds — TR-T2, TR-T7.

Validates early stopping patience, best-weight restoration,
and seed reproducibility.
"""

from __future__ import annotations

import torch
from torch import nn

from freq_extractor.services.training_helpers import EarlyStopping, set_all_seeds


class TestEarlyStopping:
    """EarlyStopping behaviour tests."""

    def test_stops_after_patience(self) -> None:
        """TR-T2: Stops after patience epochs of no improvement."""
        stopper = EarlyStopping(patience=3, min_delta=1e-6)
        model = nn.Linear(10, 1)
        stopper.step(1.0, model)
        stopper.step(0.5, model)
        stopper.step(0.6, model)
        stopper.step(0.7, model)
        stopper.step(0.8, model)
        assert stopper.should_stop

    def test_does_not_stop_on_improvement(self) -> None:
        """Continuous improvement prevents early stopping."""
        stopper = EarlyStopping(patience=3)
        model = nn.Linear(10, 1)
        for v in [1.0, 0.9, 0.8, 0.7, 0.6]:
            stopper.step(v, model)
        assert not stopper.should_stop

    def test_restores_best_weights(self) -> None:
        """restore_best loads the best epoch weights."""
        stopper = EarlyStopping(patience=2)
        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.fill_(1.0)
        stopper.step(0.5, model)
        with torch.no_grad():
            model.weight.fill_(99.0)
        stopper.step(0.9, model)
        stopper.step(0.9, model)
        stopper.restore_best(model)
        assert torch.allclose(model.weight, torch.ones_like(model.weight))

    def test_best_loss_tracked(self) -> None:
        """best_loss is updated correctly."""
        stopper = EarlyStopping(patience=5)
        model = nn.Linear(2, 1)
        stopper.step(1.0, model)
        assert stopper.best_loss == 1.0
        stopper.step(0.5, model)
        assert stopper.best_loss == 0.5

    def test_counter_resets_on_improvement(self) -> None:
        """Counter resets to 0 when loss improves."""
        stopper = EarlyStopping(patience=5)
        model = nn.Linear(2, 1)
        stopper.step(1.0, model)
        stopper.step(1.1, model)
        stopper.step(1.2, model)
        assert stopper.counter == 2
        stopper.step(0.5, model)
        assert stopper.counter == 0


class TestSetAllSeeds:
    """Seed management tests."""

    def test_reproducible_torch(self) -> None:
        """TR-T7: Same seed → identical torch random numbers."""
        set_all_seeds(42)
        a = torch.randn(10)
        set_all_seeds(42)
        b = torch.randn(10)
        torch.testing.assert_close(a, b)

    def test_reproducible_numpy(self) -> None:
        """Same seed → identical numpy random numbers."""
        import numpy as np

        set_all_seeds(42)
        a = np.random.rand(10)
        set_all_seeds(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)
