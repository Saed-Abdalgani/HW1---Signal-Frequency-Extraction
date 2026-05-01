"""Tests for DatasetBuilder — SG-T3, SG-T6, SG-T8.

Validates sliding-window count, one-hot encoding, and window-size edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from freq_extractor.constants import FREQUENCY_INDEX, FREQUENCY_LABELS, NUM_CLASSES
from freq_extractor.services.data_service import DatasetBuilder, SignalGenerator


class TestDatasetBuilder:
    """Sliding-window dataset construction tests."""

    def test_window_count(self, signal_gen: SignalGenerator) -> None:
        """SG-T3: Sliding window of 10 on 200-sample signal → 190 entries."""
        clean = signal_gen.generate_clean(15.0)
        noisy = clean.copy()
        entries = DatasetBuilder.build_windows(clean, noisy, 15, 10)
        expected = len(clean) - 10
        assert len(entries) == expected

    def test_one_hot_correctness(self, signal_gen: SignalGenerator) -> None:
        """SG-T8: One-hot label sums to 1 and argmax matches frequency index."""
        for freq in FREQUENCY_LABELS:
            clean = signal_gen.generate_clean(float(freq))
            entries = DatasetBuilder.build_windows(clean, clean, freq, 10)
            for e in entries[:5]:
                label = e["frequency_label"]
                assert label.shape == (NUM_CLASSES,)
                assert abs(np.sum(label) - 1.0) < 1e-6
                assert np.argmax(label) == FREQUENCY_INDEX[freq]

    def test_entry_shapes(self, signal_gen: SignalGenerator) -> None:
        """Each entry has correct shapes for all 4 keys."""
        clean = signal_gen.generate_clean(5.0)
        entries = DatasetBuilder.build_windows(clean, clean, 5, 10)
        e = entries[0]
        assert e["noisy_samples"].shape == (10,)
        assert e["clean_samples"].shape == (10,)
        assert e["target_output"].shape == (1,)
        assert e["frequency_label"].shape == (NUM_CLASSES,)

    def test_target_is_next_clean_sample(self, signal_gen: SignalGenerator) -> None:
        """Target output equals the next clean sample after the window."""
        clean = signal_gen.generate_clean(15.0)
        entries = DatasetBuilder.build_windows(clean, clean, 15, 10)
        for i, e in enumerate(entries[:10]):
            assert abs(e["target_output"][0] - clean[i + 10]) < 1e-6

    def test_window_too_large_raises(self, signal_gen: SignalGenerator) -> None:
        """SG-T6: window_size >= signal length raises ValueError."""
        clean = signal_gen.generate_clean(5.0)
        with pytest.raises(ValueError, match="window_size"):
            DatasetBuilder.build_windows(clean, clean, 5, len(clean))

    def test_float32_dtype(self, signal_gen: SignalGenerator) -> None:
        """All array values are float32."""
        clean = signal_gen.generate_clean(5.0)
        entries = DatasetBuilder.build_windows(clean, clean, 5, 10)
        e = entries[0]
        assert e["noisy_samples"].dtype == np.float32
        assert e["clean_samples"].dtype == np.float32
        assert e["target_output"].dtype == np.float32
        assert e["frequency_label"].dtype == np.float32
