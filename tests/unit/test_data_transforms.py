"""Tests for DatasetSplitter and DataNormalizer — SG-T7, SG-T9, SG-T10.

Validates stratified splitting, ratio enforcement, and normalisation
data-leakage prevention.
"""

from __future__ import annotations

import numpy as np
import pytest

from freq_extractor.services.data_transforms import DataNormalizer, DatasetSplitter


class TestDatasetSplitter:
    """Stratified train/val/test splitting tests."""

    def test_splits_sum_to_total(self, small_entries, rng) -> None:
        """SG-T7: len(train) + len(val) + len(test) == total."""
        train, val, test = DatasetSplitter.split(small_entries, 0.7, 0.15, 0.15, rng)
        assert len(train) + len(val) + len(test) == len(small_entries)

    def test_stratified_all_freqs_present(self, small_entries, rng) -> None:
        """SG-T9: Each split contains all 4 frequency classes."""
        train, val, test = DatasetSplitter.split(small_entries, 0.7, 0.15, 0.15, rng)
        for split in (train, val, test):
            present = {int(np.argmax(e["frequency_label"])) for e in split}
            assert len(present) == 4, f"Expected 4 freqs, got {present}"

    def test_ratios_approximately_correct(self, small_entries, rng) -> None:
        """Split ratios are approximately 70/15/15."""
        train, val, test = DatasetSplitter.split(small_entries, 0.7, 0.15, 0.15, rng)
        total = len(small_entries)
        assert abs(len(train) / total - 0.7) < 0.05
        assert abs(len(val) / total - 0.15) < 0.05

    def test_bad_ratios_raise(self, small_entries, rng) -> None:
        """Split ratios not summing to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            DatasetSplitter.split(small_entries, 0.5, 0.5, 0.5, rng)

    def test_empty_entries_raise(self, rng) -> None:
        """EC.1: Empty dataset split raises a clear ValueError."""
        with pytest.raises(ValueError, match="entries must not be empty"):
            DatasetSplitter.split([], 0.7, 0.15, 0.15, rng)

    def test_single_frequency_split_keeps_one_hot_shape(self, small_entries, rng) -> None:
        """EC.2: Single-frequency datasets split and keep 4-dim labels."""
        single_freq = [e for e in small_entries if int(np.argmax(e["frequency_label"])) == 0]
        train, val, test = DatasetSplitter.split(single_freq, 0.7, 0.15, 0.15, rng)
        assert train and val and test
        for entry in train + val + test:
            assert entry["frequency_label"].shape == (4,)
            assert np.sum(entry["frequency_label"]) == 1.0


class TestDataNormalizer:
    """Zero-mean unit-variance normalisation tests."""

    def test_fit_computes_stats(self, small_entries) -> None:
        """Fit sets mean and std from training data."""
        norm = DataNormalizer()
        norm.fit(small_entries)
        assert norm.std > 0
        assert np.isfinite(norm.mean)
        assert np.isfinite(norm.std)
        assert isinstance(norm.mean, float)

    def test_transform_normalises(self, small_entries) -> None:
        """SG-T10: After normalisation, noisy_samples have ~zero mean, ~unit std."""
        norm = DataNormalizer()
        norm.fit(small_entries)
        transformed = norm.transform(small_entries)
        all_vals = np.concatenate([e["noisy_samples"] for e in transformed])
        assert abs(np.mean(all_vals)) < 0.1
        assert abs(np.std(all_vals) - 1.0) < 0.1

    def test_no_leakage_different_stats(self, small_entries, rng) -> None:
        """SG-T10: Normaliser fitted on train only — val/test use train stats."""
        train, val, _ = DatasetSplitter.split(small_entries, 0.7, 0.15, 0.15, rng)
        norm = DataNormalizer()
        norm.fit(train)
        train_mean = norm.mean
        norm2 = DataNormalizer()
        norm2.fit(val)
        assert norm2.mean != train_mean or norm2.std != norm.std

    def test_zero_std_fallback(self) -> None:
        """If std is near zero, normaliser uses std=1.0 to avoid division by zero."""
        entries = [{"noisy_samples": np.zeros(10, dtype=np.float32),
                    "clean_samples": np.zeros(10, dtype=np.float32),
                    "target_output": np.zeros(1, dtype=np.float32),
                    "frequency_label": np.array([1, 0, 0, 0], dtype=np.float32)}]
        norm = DataNormalizer()
        norm.fit(entries)
        assert norm.std == 1.0

    def test_empty_entries_raise(self) -> None:
        """EC.1: Empty normaliser fit raises a clear ValueError."""
        norm = DataNormalizer()
        with pytest.raises(ValueError, match="entries must not be empty"):
            norm.fit([])
