"""Tests for FreqExtractorSDK defensive training behaviour."""

from __future__ import annotations

import torch
from torch import nn

from freq_extractor.sdk.sdk import FreqExtractorSDK


class _FakeModel(nn.Module):
    """Tiny model whose ``to`` method avoids requiring a real CUDA device."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(14, 1)
        self.requested_devices: list[torch.device] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    def to(self, device):  # type: ignore[override]
        self.requested_devices.append(torch.device(device))
        return self

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def test_cuda_oom_falls_back_to_cpu(monkeypatch, sample_config, small_entries) -> None:
    """EC.10: CUDA OOM at train start retries on CPU and returns history."""
    from freq_extractor.services import model_factory, training_service

    calls = {"train": 0}

    def fake_create_model(_model_type, _config):
        return _FakeModel()

    def fake_train(_model, _train_loader, _val_loader, _config, _model_type):
        calls["train"] += 1
        if calls["train"] == 1:
            raise RuntimeError("CUDA out of memory")
        return {"train_losses": [1.0], "val_losses": [1.0],
                "best_epoch": 1, "best_val_mse": 1.0}

    monkeypatch.setattr(model_factory.ModelFactory, "create_model",
                        staticmethod(fake_create_model))
    monkeypatch.setattr(training_service, "train", fake_train)

    sdk = FreqExtractorSDK(config=sample_config, seed=42)
    sdk._device = torch.device("cuda")
    model, history = sdk.train("mlp", small_entries[:8], small_entries[8:16])

    assert isinstance(model, _FakeModel)
    assert sdk.device.type == "cpu"
    assert calls["train"] == 2
    assert history["best_val_mse"] == 1.0
