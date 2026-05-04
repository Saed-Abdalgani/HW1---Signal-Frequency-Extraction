"""Setup validation tests for config.py — DV.1, DV.4-DV.6.

Validates setup schema, training ranges, signal frequencies, and Nyquist constraint.
"""

from __future__ import annotations

import json

import pytest

from freq_extractor.shared.config import clear_cache, get_setup


class TestSetupValidation:
    """Setup configuration validation tests."""

    def test_setup_validates_version(self, tmp_config_dir) -> None:
        """get_setup validates config version."""
        cfg = get_setup()
        assert "signal" in cfg

    def test_setup_schema_and_ranges(self, tmp_config_dir) -> None:
        """DV.1, DV.4-DV.6: setup schema and critical ranges validate."""
        cfg = get_setup()
        assert cfg["training"]["learning_rate"] > 0
        assert cfg["training"]["batch_size"] > 0
        assert cfg["training"]["max_epochs"] > 0
        assert min(cfg["signal"]["frequencies_hz"]) > 0
        assert cfg["signal"]["sampling_rate_hz"] >= 2 * max(cfg["signal"]["frequencies_hz"])

    def test_invalid_training_range_raises(self, tmp_config_dir) -> None:
        """DV.4: Bad hyperparameter ranges fail fast."""
        path = tmp_config_dir / "setup.json"
        cfg = json.loads(path.read_text())
        cfg["training"]["batch_size"] = 0
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="batch_size"):
            get_setup()

    def test_invalid_frequency_list_raises(self, tmp_config_dir) -> None:
        """DV.5: Empty or non-positive frequencies fail fast."""
        path = tmp_config_dir / "setup.json"
        cfg = json.loads(path.read_text())
        cfg["signal"]["frequencies_hz"] = [5, -1]
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="frequencies_hz"):
            get_setup()

    def test_invalid_nyquist_raises(self, tmp_config_dir) -> None:
        """DV.6: sampling_rate must satisfy Nyquist at startup."""
        path = tmp_config_dir / "setup.json"
        cfg = json.loads(path.read_text())
        cfg["signal"]["sampling_rate_hz"] = 50
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="Nyquist"):
            get_setup()

    def test_missing_keys_raises(self, tmp_config_dir) -> None:
        """DV.1: Missing keys raise ValueError."""
        path = tmp_config_dir / "setup.json"
        cfg = json.loads(path.read_text())
        del cfg["dataset"]
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="missing required keys: \\['dataset'\\]"):
            get_setup()

    def test_negative_amplitude_or_noise_raises(self, tmp_config_dir) -> None:
        """Negative amplitude or noise raises ValueError."""
        path = tmp_config_dir / "setup.json"
        cfg = json.loads(path.read_text())
        cfg["signal"]["amplitude"] = -1.0
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="amplitude and noise_std_ratio must be non-negative"):
            get_setup()

    def test_window_size_too_large_raises(self, tmp_config_dir) -> None:
        """Window size >= total samples raises ValueError."""
        path = tmp_config_dir / "setup.json"
        cfg = json.loads(path.read_text())
        cfg["signal"]["window_size"] = 1000000
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="window_size must be < total signal samples"):
            get_setup()

    def test_split_ratios_do_not_sum_to_one(self, tmp_config_dir) -> None:
        """Split ratios not summing to 1.0 raises ValueError."""
        path = tmp_config_dir / "setup.json"
        cfg = json.loads(path.read_text())
        cfg["dataset"]["train_ratio"] = 0.5
        cfg["dataset"]["val_ratio"] = 0.5
        cfg["dataset"]["test_ratio"] = 0.5
        path.write_text(json.dumps(cfg))
        clear_cache()
        with pytest.raises(ValueError, match="ratios must sum to 1.0"):
            get_setup()
