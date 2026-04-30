"""PyTorch Dataset wrappers and DataLoader factory.

Provides two dataset classes — one for the MLP (flat input) and one
for sequential models (RNN/LSTM) — plus a deterministic DataLoader
factory that guarantees reproducibility across runs.

References
----------
- PLAN §5, PRD FR-2, PRD_tr §8
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from freq_extractor.constants import TENSOR_DTYPE


class MLPDataset(Dataset):
    """PyTorch Dataset for the MLP architecture.

    Each sample concatenates ``[noisy_samples ‖ frequency_label]`` into a
    14-dimensional input vector paired with a scalar target.

    Parameters
    ----------
    entries : list[dict[str, np.ndarray]]
        Dataset entries from :func:`data_service.build_full_dataset`.
    """

    def __init__(self, entries: list[dict[str, np.ndarray]]) -> None:
        """Store entries and pre-compute tensors for fast access."""
        self.entries = entries

    def __len__(self) -> int:
        """Return number of entries."""
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(X[14], y[1])`` for the given index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Input features ``(window_size + NUM_CLASSES,)`` and target ``(1,)``.
        """
        e = self.entries[idx]
        x = np.concatenate([e["noisy_samples"], e["frequency_label"]])
        return (
            torch.tensor(x, dtype=TENSOR_DTYPE),
            torch.tensor(e["target_output"], dtype=TENSOR_DTYPE),
        )


class SequentialDataset(Dataset):
    """PyTorch Dataset for RNN / LSTM architectures.

    Each sample reshapes the noisy window into a ``(T, 1+NUM_CLASSES)``
    tensor where every timestep carries the sample value and the
    frequency one-hot label.

    Parameters
    ----------
    entries : list[dict[str, np.ndarray]]
        Dataset entries from :func:`data_service.build_full_dataset`.
    """

    def __init__(self, entries: list[dict[str, np.ndarray]]) -> None:
        """Store entries."""
        self.entries = entries

    def __len__(self) -> int:
        """Return number of entries."""
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(X[T, 1+NUM_CLASSES], y[1])`` for the given index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Sequence input ``(window_size, 5)`` and target ``(1,)``.
        """
        e = self.entries[idx]
        noisy = e["noisy_samples"]  # shape (window_size,)
        label = e["frequency_label"]  # shape (NUM_CLASSES,)
        # Stack: each timestep → [sample_t, one_hot_0, ..., one_hot_3]
        seq = np.column_stack([
            noisy.reshape(-1, 1),
            np.tile(label, (len(noisy), 1)),
        ])
        return (
            torch.tensor(seq, dtype=TENSOR_DTYPE),
            torch.tensor(e["target_output"], dtype=TENSOR_DTYPE),
        )


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker for deterministic data loading."""
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42,
    **kwargs: Any,
) -> DataLoader:
    """Create a deterministic ``DataLoader``.

    Parameters
    ----------
    dataset : Dataset
        A :class:`MLPDataset` or :class:`SequentialDataset`.
    batch_size : int
        Mini-batch size.
    shuffle : bool
        Whether to shuffle at each epoch.
    seed : int
        Seed for the shuffle generator.

    Returns
    -------
    DataLoader
        Configured loader with reproducible shuffling.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
        **kwargs,
    )
