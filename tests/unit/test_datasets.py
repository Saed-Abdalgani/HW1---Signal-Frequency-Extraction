"""Tests for PyTorch Dataset wrappers and DataLoader factory.

Validates MLPDataset, SequentialDataset tensor shapes, dtypes,
and deterministic DataLoader construction.
"""

from __future__ import annotations

import torch

from freq_extractor.constants import NUM_CLASSES, TENSOR_DTYPE
from freq_extractor.services.datasets import (
    MLPDataset,
    SequentialDataset,
    create_dataloader,
)


class TestMLPDataset:
    """MLPDataset shape and dtype tests."""

    def test_length(self, small_entries) -> None:
        """Dataset length matches entry count."""
        ds = MLPDataset(small_entries)
        assert len(ds) == len(small_entries)

    def test_item_shapes(self, small_entries) -> None:
        """X has shape (14,), y has shape (1,)."""
        ds = MLPDataset(small_entries)
        x, y = ds[0]
        assert x.shape == (10 + NUM_CLASSES,)
        assert y.shape == (1,)

    def test_dtype(self, small_entries) -> None:
        """Tensors are float32."""
        ds = MLPDataset(small_entries)
        x, y = ds[0]
        assert x.dtype == TENSOR_DTYPE
        assert y.dtype == TENSOR_DTYPE

    def test_no_nan(self, small_entries) -> None:
        """No NaN values in dataset tensors."""
        ds = MLPDataset(small_entries)
        x, y = ds[0]
        assert torch.isfinite(x).all()
        assert torch.isfinite(y).all()


class TestSequentialDataset:
    """SequentialDataset shape and dtype tests."""

    def test_length(self, small_entries) -> None:
        """Dataset length matches entry count."""
        ds = SequentialDataset(small_entries)
        assert len(ds) == len(small_entries)

    def test_item_shapes(self, small_entries) -> None:
        """X has shape (10, 5), y has shape (1,)."""
        ds = SequentialDataset(small_entries)
        x, y = ds[0]
        assert x.shape == (10, 1 + NUM_CLASSES)
        assert y.shape == (1,)

    def test_dtype(self, small_entries) -> None:
        """Tensors are float32."""
        ds = SequentialDataset(small_entries)
        x, y = ds[0]
        assert x.dtype == TENSOR_DTYPE
        assert y.dtype == TENSOR_DTYPE


class TestDataLoader:
    """DataLoader factory tests."""

    def test_create_dataloader(self, small_entries) -> None:
        """DataLoader yields batches of correct shape."""
        ds = MLPDataset(small_entries)
        loader = create_dataloader(ds, batch_size=16, shuffle=True, seed=42)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[0] <= 16
        assert batch_x.shape[1] == 14

    def test_deterministic_order(self, small_entries) -> None:
        """Same seed → same batch order."""
        ds = MLPDataset(small_entries)
        l1 = create_dataloader(ds, batch_size=8, shuffle=True, seed=42)
        l2 = create_dataloader(ds, batch_size=8, shuffle=True, seed=42)
        x1, _ = next(iter(l1))
        x2, _ = next(iter(l2))
        torch.testing.assert_close(x1, x2)

    def test_partial_batch_for_small_dataset(self, small_entries) -> None:
        """EC.11: Dataset smaller than batch_size yields a partial batch."""
        ds = MLPDataset(small_entries[:3])
        loader = create_dataloader(ds, batch_size=64, shuffle=False, seed=42)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[0] == 3
        assert batch_y.shape[0] == 3
