"""Data transforms — splitting and normalisation.

Contains ``DatasetSplitter`` for stratified train/val/test splitting
and ``DataNormalizer`` for zero-mean unit-variance normalisation
(fitted on training data only to prevent leakage).

References
----------
- PRD FR-2, PRD_sig §§6–7, PLAN §4
"""

from __future__ import annotations

import numpy as np


class DatasetSplitter:
    """Stratified train / validation / test splitting."""

    @staticmethod
    def split(
        entries: list[dict[str, np.ndarray]],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        rng: np.random.Generator,
    ) -> tuple[list, list, list]:
        """Stratified split ensuring all frequencies appear in each subset.

        Parameters
        ----------
        entries : list
            Full list of dataset entries (each with ``frequency_label``).
        train_ratio, val_ratio, test_ratio : float
            Must sum to 1.0.
        rng : np.random.Generator
            Seeded generator for reproducibility.

        Returns
        -------
        tuple[list, list, list]
            ``(train, val, test)`` subsets.
        """
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        by_freq: dict[int, list] = {}
        for e in entries:
            idx = int(np.argmax(e["frequency_label"]))
            by_freq.setdefault(idx, []).append(e)

        train, val, test = [], [], []
        for freq_entries in by_freq.values():
            perm = rng.permutation(len(freq_entries)).tolist()
            n = len(freq_entries)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            for i, p in enumerate(perm):
                if i < n_train:
                    train.append(freq_entries[p])
                elif i < n_train + n_val:
                    val.append(freq_entries[p])
                else:
                    test.append(freq_entries[p])

        return train, val, test


class DataNormalizer:
    """Fit normalisation statistics on training data, transform all splits.

    Prevents data leakage by computing mean/std exclusively from the
    training subset.
    """

    def __init__(self) -> None:
        """Initialise with identity statistics."""
        self.mean: float = 0.0
        self.std: float = 1.0

    def fit(self, entries: list[dict[str, np.ndarray]]) -> None:
        """Compute mean and std from training noisy samples."""
        all_vals = np.concatenate([e["noisy_samples"] for e in entries])
        self.mean = float(np.mean(all_vals))
        self.std = float(np.std(all_vals))
        if self.std < 1e-8:
            self.std = 1.0

    def transform(self, entries: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
        """Normalise noisy_samples, clean_samples, and target_output in-place."""
        for e in entries:
            e["noisy_samples"] = ((e["noisy_samples"] - self.mean) / self.std).astype(np.float32)
            e["clean_samples"] = ((e["clean_samples"] - self.mean) / self.std).astype(np.float32)
            e["target_output"] = ((e["target_output"] - self.mean) / self.std).astype(np.float32)
        return entries
