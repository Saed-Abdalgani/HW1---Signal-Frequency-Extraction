"""Analysis plots — robustness, per-frequency, and signal examples.

References
----------
- PRD FR-7, PRD_tr §11
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from freq_extractor.constants import FREQUENCY_LABELS  # noqa: E402
from freq_extractor.services.plot_helpers import (  # noqa: E402
    COLOURS,
    MODEL_COLOURS,
    STYLE_KW,
    _save_fig,
)


def plot_noise_robustness(
    results: dict[str, dict[float, float]], save_path: Path,
) -> None:
    """Plot test MSE vs noise sigma for all models."""
    with plt.rc_context(STYLE_KW):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, sigmas in results.items():
            c = MODEL_COLOURS.get(name, "#ffffff")
            xs = sorted(sigmas.keys())
            ax.plot(xs, [sigmas[s] for s in xs], "o-", color=c, label=name.upper(), lw=2, ms=6)
        ax.set(xlabel="Noise sigma", ylabel="Test MSE", title="Noise Robustness")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    _save_fig(fig, save_path)


def plot_per_frequency_mse(
    results: dict[str, dict[int, float]], save_path: Path,
) -> None:
    """Bar chart: 3 models x 4 frequencies."""
    with plt.rc_context(STYLE_KW):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(FREQUENCY_LABELS))
        width = 0.25
        for i, (name, freq_mse) in enumerate(results.items()):
            c = MODEL_COLOURS.get(name, "#ffffff")
            vals = [freq_mse.get(f, 0.0) for f in FREQUENCY_LABELS]
            ax.bar(x + i * width, vals, width, label=name.upper(), color=c, alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{f} Hz" for f in FREQUENCY_LABELS])
        ax.set(xlabel="Frequency", ylabel="Test MSE", title="Per-Frequency MSE")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
    _save_fig(fig, save_path)


def plot_signal_examples(config: dict[str, Any], save_path: Path) -> None:
    """Plot clean/noisy for each frequency."""
    from freq_extractor.services.data_service import SignalGenerator

    sig = config["signal"]
    gen = SignalGenerator(sig["sampling_rate_hz"], sig["duration_seconds"], sig["amplitude"])
    rng = np.random.default_rng(42)
    with plt.rc_context(STYLE_KW):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        for ax, freq, colour in zip(axes.flat, sig["frequencies_hz"], COLOURS, strict=False):
            phase = rng.uniform(0, 2 * np.pi)
            clean = gen.generate_clean(float(freq), phase)
            noisy = gen.generate_noisy(clean, sig["noise_std_ratio"], rng)
            t = gen.t[:200]
            ax.plot(t, clean[:200], color=colour, label="Clean", lw=1.2)
            ax.plot(t, noisy[:200], color="#8b949e", alpha=0.5, label="Noisy", lw=0.8)
            ax.set(title=f"{freq} Hz", xlabel="Time (s)", ylabel="Amplitude")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Signal Examples", fontweight="bold", fontsize=14)
        fig.tight_layout()
    _save_fig(fig, save_path)
