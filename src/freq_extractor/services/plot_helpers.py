"""Plot helpers — training curves and prediction plots.

Provides the core plotting infrastructure (palette, style, save helper)
plus ``plot_training_curves`` and ``plot_predictions``.

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
import torch  # noqa: E402
from torch import nn  # noqa: E402

from freq_extractor.constants import MIN_PLOT_DPI  # noqa: E402
from freq_extractor.shared import get_gatekeeper  # noqa: E402

# -- Consistent palette --
COLOURS = ["#00d2ff", "#ff6b6b", "#51cf66", "#ffd43b"]
MODEL_COLOURS = {"mlp": COLOURS[0], "rnn": COLOURS[1], "lstm": COLOURS[2]}
STYLE_KW: dict[str, Any] = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22", "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3", "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "axes.edgecolor": "#30363d", "grid.color": "#21262d"}


def _save_fig(fig: plt.Figure, path: Path) -> None:
    """Save figure via gatekeeper with minimum DPI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    get_gatekeeper("file_io").execute(
        lambda: fig.savefig(str(path), dpi=MIN_PLOT_DPI, bbox_inches="tight",
                            facecolor=fig.get_facecolor()))
    plt.close(fig)


def plot_training_curves(
    histories: dict[str, dict[str, list[float]]], save_path: Path,
) -> None:
    """Plot train/val MSE vs epoch for all models."""
    with plt.rc_context(STYLE_KW):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, h in histories.items():
            c = MODEL_COLOURS.get(name, "#ffffff")
            axes[0].plot(h["train_losses"], label=f"{name.upper()} train", color=c, alpha=0.8)
            axes[1].plot(h["val_losses"], label=f"{name.upper()} val", color=c, alpha=0.8)
        for ax, title in zip(axes, ["Training Loss", "Validation Loss"], strict=False):
            ax.set(xlabel="Epoch", ylabel="MSE", title=title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.suptitle("Training Curves — All Models", fontweight="bold", fontsize=14)
        fig.tight_layout()
    _save_fig(fig, save_path)


def plot_predictions(
    model: nn.Module, entries: list, model_name: str,
    model_type: str, save_path: Path, n_examples: int = 100,
) -> None:
    """Plot noisy input, model prediction, and clean ground truth."""
    model.eval()
    preds, targets, noisy_vals = [], [], []
    with torch.no_grad():
        for e in entries[:n_examples]:
            ns = e["noisy_samples"]
            fl = e["frequency_label"]
            sig = np.array([np.float32(e["sigma"])], dtype=np.float32)
            if model_type == "mlp":
                x = torch.tensor(
                    np.concatenate([ns, fl, sig]), dtype=torch.float32,
                ).unsqueeze(0)
            else:
                sig_col = np.full((len(ns), 1), np.float32(e["sigma"]), dtype=np.float32)
                seq = np.column_stack([
                    ns.reshape(-1, 1), np.tile(fl, (len(ns), 1)), sig_col,
                ])
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            preds.append(model(x).item())
            targets.append(e["target_output"][0])
            noisy_vals.append(ns[-1])
    with plt.rc_context(STYLE_KW):
        fig, ax = plt.subplots(figsize=(12, 4))
        idx = range(len(preds))
        ax.plot(idx, noisy_vals, ".", color="#8b949e", alpha=0.4, label="Noisy", ms=3)
        ax.plot(idx, targets, "-", color="#51cf66", alpha=0.8, label="Clean target", lw=1.2)
        ax.plot(idx, preds, "-", color="#ff6b6b", alpha=0.8, label=f"{model_name} pred", lw=1.2)
        ax.set(xlabel="Sample index", ylabel="Amplitude",
               title=f"Predictions — {model_name.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
    _save_fig(fig, save_path)
