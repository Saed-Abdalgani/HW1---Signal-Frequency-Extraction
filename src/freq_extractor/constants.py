"""Package-wide constants for freq_extractor.

Centralises all enumeration-like tuples, type tags, and magic values so
that no service needs to hard-code them.  Import from here instead of
redefining local literals in individual modules.

References
----------
- PRD FR-1, FR-2, FR-9 §9.1 (PLAN §6, ADR-5)
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Model taxonomy
# ---------------------------------------------------------------------------

MODEL_TYPES: tuple[str, ...] = ("mlp", "rnn", "lstm")
"""All supported model architecture identifiers (lowercase)."""

# ---------------------------------------------------------------------------
# Signal / dataset
# ---------------------------------------------------------------------------

FREQUENCY_LABELS: tuple[int, ...] = (2, 3, 4, 6)
"""Default frequencies in Hz (each < 7); must match config/setup.json."""

FREQUENCY_INDEX: dict[int, int] = {f: i for i, f in enumerate(FREQUENCY_LABELS)}
"""Maps frequency (Hz) → one-hot class index."""

NUM_CLASSES: int = len(FREQUENCY_LABELS)
"""Number of frequency classes (== one-hot vector length)."""

SEQ_INPUT_SIZE: int = 1 + NUM_CLASSES + 1
"""RNN/LSTM features per timestep: sample, one-hot label, sigma."""

MLP_EXTRA_FEATURES: int = NUM_CLASSES + 1
"""One-hot plus sigma concatenated after noisy window for MLP."""

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
"""Dataset split identifiers in their standard processing order."""

DEFAULT_SEED: int = 42
"""Default global random seed used across all random operations."""

# ---------------------------------------------------------------------------
# Tensor / data types
# ---------------------------------------------------------------------------

TENSOR_DTYPE = torch.float32
"""Default PyTorch dtype for all signal and label tensors."""

# ---------------------------------------------------------------------------
# File / persistence
# ---------------------------------------------------------------------------

NPZ_EXT: str = ".npz"
"""Extension for NumPy compressed dataset archives."""

PT_EXT: str = ".pt"
"""Extension for PyTorch checkpoint files."""

# ---------------------------------------------------------------------------
# Evaluation / plotting
# ---------------------------------------------------------------------------

MIN_PLOT_DPI: int = 150
"""Minimum DPI for all saved result figures (PRD §15)."""

MODEL_CHECKPOINT_VERSION: str = "1.01"
"""Schema version tag stored in every checkpoint file (PLAN §5)."""

# ---------------------------------------------------------------------------
# UI constants — Sinusoid Explorer (FR-9 §9.1)
# ---------------------------------------------------------------------------

DISPLAY_MODES: tuple[str, ...] = ("LINE", "DOTS")
"""Sinusoid Explorer plot display mode options."""

NOISE_TYPES: tuple[str, ...] = ("None", "Gaussian", "Uniform")
"""Noise model choices exposed in the UI Noise dropdown."""

FILTER_TYPES: tuple[str, ...] = ("None", "Lowpass", "Highpass", "Bandpass")
"""Filter type choices exposed in the UI Filter dropdown."""

NUM_SINUSOIDS: int = 4
"""Number of independently configurable sinusoid channels in the UI."""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_SEED",
    "DISPLAY_MODES",
    "FILTER_TYPES",
    "FREQUENCY_INDEX",
    "FREQUENCY_LABELS",
    "MIN_PLOT_DPI",
    "MLP_EXTRA_FEATURES",
    "MODEL_CHECKPOINT_VERSION",
    "MODEL_TYPES",
    "NOISE_TYPES",
    "NPZ_EXT",
    "NUM_CLASSES",
    "NUM_SINUSOIDS",
    "PT_EXT",
    "SEQ_INPUT_SIZE",
    "SPLIT_NAMES",
    "TENSOR_DTYPE",
]
