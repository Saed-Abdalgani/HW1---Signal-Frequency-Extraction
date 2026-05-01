"""Tests for the data pipeline — SG-T4, save/load round-trip.

Validates end-to-end dataset construction, reproducibility,
and persistence via gatekeeper.
"""

from __future__ import annotations

import numpy as np

from freq_extractor.services.data_pipeline import (
    build_full_dataset,
    load_dataset,
    save_dataset,
)


class TestDataPipeline:
    """End-to-end data pipeline tests."""

    def test_build_full_dataset(self, sample_config, tmp_config_dir) -> None:
        """Build produces non-empty train/val/test splits."""
        train, val, test = build_full_dataset(sample_config, seed=42)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_generated_dataset_integrity(self, sample_config, tmp_config_dir) -> None:
        """DV.9-DV.12: generated splits are finite, disjoint, and valid."""
        train, val, test = build_full_dataset(sample_config, seed=42)
        assert {id(e) for e in train}.isdisjoint({id(e) for e in val})
        assert {id(e) for e in train}.isdisjoint({id(e) for e in test})
        assert {id(e) for e in val}.isdisjoint({id(e) for e in test})
        for entry in train + val + test:
            for key in ("noisy_samples", "clean_samples", "target_output",
                        "frequency_label"):
                assert entry[key].dtype == np.float32
                assert np.isfinite(entry[key]).all()
            assert np.sum(entry["frequency_label"]) == 1.0

    def test_reproducibility(self, sample_config, tmp_config_dir) -> None:
        """SG-T4: Same seed produces identical datasets."""
        t1, v1, te1 = build_full_dataset(sample_config, seed=42)
        t2, v2, te2 = build_full_dataset(sample_config, seed=42)
        for a, b in zip(t1[:5], t2[:5], strict=True):
            np.testing.assert_array_equal(a["noisy_samples"], b["noisy_samples"])

    def test_different_seed_different_data(self, sample_config, tmp_config_dir) -> None:
        """Different seeds produce different datasets."""
        t1, _, _ = build_full_dataset(sample_config, seed=42)
        t2, _, _ = build_full_dataset(sample_config, seed=99)
        differ = any(
            not np.array_equal(a["noisy_samples"], b["noisy_samples"])
            for a, b in zip(t1[:5], t2[:5], strict=True)
        )
        assert differ

    def test_save_load_round_trip(self, tmp_path, small_entries, rng, tmp_config_dir) -> None:
        """Saved and loaded dataset entries match originals."""
        from freq_extractor.services.data_transforms import DatasetSplitter

        train, val, test = DatasetSplitter.split(small_entries, 0.7, 0.15, 0.15, rng)
        path = tmp_path / "test_ds.npz"
        save_dataset(path, train, val, test)
        t2, v2, te2 = load_dataset(path)
        assert len(t2) == len(train)
        assert len(v2) == len(val)
        assert len(te2) == len(test)
        for original, loaded in zip(train[:5], t2[:5], strict=True):
            for key in original:
                np.testing.assert_array_equal(original[key], loaded[key])
