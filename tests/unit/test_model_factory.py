"""Tests for ModelFactory — 6F.1..6F.4.

Validates factory creates correct model types and rejects unknowns.
"""

from __future__ import annotations

import pytest

from freq_extractor.services.lstm_model import LSTMModel
from freq_extractor.services.mlp_model import MLPModel
from freq_extractor.services.model_factory import ModelFactory
from freq_extractor.services.rnn_model import RNNModel


class TestModelFactory:
    """Model factory tests."""

    def test_creates_mlp(self, sample_config) -> None:
        """6F.1: Factory returns MLPModel for 'mlp'."""
        model = ModelFactory.create_model("mlp", sample_config)
        assert isinstance(model, MLPModel)

    def test_creates_rnn(self, sample_config) -> None:
        """6F.2: Factory returns RNNModel for 'rnn'."""
        model = ModelFactory.create_model("rnn", sample_config)
        assert isinstance(model, RNNModel)

    def test_creates_lstm(self, sample_config) -> None:
        """6F.3: Factory returns LSTMModel for 'lstm'."""
        model = ModelFactory.create_model("lstm", sample_config)
        assert isinstance(model, LSTMModel)

    def test_unknown_type_raises(self, sample_config) -> None:
        """6F.4: Unknown model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create_model("transformer", sample_config)

    def test_case_insensitive(self, sample_config) -> None:
        """Factory accepts uppercase model names."""
        model = ModelFactory.create_model("MLP", sample_config)
        assert isinstance(model, MLPModel)

    def test_uses_config_params(self, sample_config) -> None:
        """Factory applies config hidden_size and layers."""
        model = ModelFactory.create_model("rnn", sample_config)
        assert model.hidden_size == sample_config["model"]["hidden_size"]

    def test_none_config_uses_defaults(self) -> None:
        """Factory works with None config using defaults."""
        model = ModelFactory.create_model("mlp", None)
        assert isinstance(model, MLPModel)
