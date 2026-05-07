"""Window stacking and subsampling for PCA / T-SNE analysis callbacks."""

from __future__ import annotations

from typing import Any

import numpy as np

from freq_extractor.services.ui_layout import SIN_COLOURS
from freq_extractor.services.ui_signal_utils import (
    clamp_n_cycles,
    compute_shared_grid,
    f_min_from_sin_bundle,
    signal_on_grid,
)

MAX_TSNE_ROWS = 800
TSNE_ITER = 400


def stack_window_features(
    fs: float, n_cyc: float, sin_vals: tuple[Any, ...], window: int,
) -> tuple[list, list, list]:
    """Windows of length *window* aligned with the SIGNALS-tab time grid."""
    fs = fs or 200
    n_cyc = clamp_n_cycles(n_cyc)
    t = compute_shared_grid(fs, n_cyc, f_min_from_sin_bundle(sin_vals))
    feats, labels, colours = [], [], []
    for i in range(4):
        mix = sin_vals[i * 6]
        freq = sin_vals[i * 6 + 2]
        if not (mix and "mix" in mix and freq and freq > 0):
            continue
        phase = sin_vals[i * 6 + 3] or 0
        amp = sin_vals[i * 6 + 4] or 1
        sig = signal_on_grid(t, float(freq), float(phase), float(amp))
        for j in range(len(sig) - window):
            feats.append(sig[j:j + window])
            labels.append(f"Sin {i + 1}")
            colours.append(SIN_COLOURS[i])
    return feats, labels, colours


def subsample_feature_rows(
    feats: np.ndarray, labels: list, colours: list, max_rows: int, seed: int = 42,
) -> tuple[np.ndarray, list, list]:
    """Uniform row subsample for expensive manifold fits."""
    n = feats.shape[0]
    if n <= max_rows:
        return feats, labels, colours
    rng = np.random.default_rng(seed)
    ix = rng.choice(n, max_rows, replace=False)
    ix.sort()
    return feats[ix], [labels[int(i)] for i in ix], [colours[int(i)] for i in ix]


def safe_tsne_perplexity(perp_choice: Any, n_samples: int) -> float:
    """TSNE perplexity bounded below *n_samples*."""
    p = float(perp_choice if perp_choice is not None else 30.0)
    ub = float(max(n_samples - 1.08, 2.09))
    return float(min(max(p, 2.1), ub))
