"""Data pipeline — end-to-end dataset construction and persistence.

Orchestrates signal generation, dataset building, splitting, normalisation,
and persistence.  Separated from ``data_service.py`` to comply with the
≤ 145 code-line constraint (NFR-3 v1.10).

References
----------
- PRD FR-1, FR-2, PRD_sig §§2–7, PLAN §4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from freq_extractor.constants import FREQUENCY_INDEX, NPZ_EXT
from freq_extractor.services.data_service import DatasetBuilder, SignalGenerator
from freq_extractor.services.data_transforms import DataNormalizer, DatasetSplitter
from freq_extractor.shared import get_gatekeeper

logger = logging.getLogger("freq_extractor.data_pipeline")


def save_dataset(path: Path, train: list, val: list, test: list) -> None:
    """Persist splits to ``.npz`` via the file-I/O gatekeeper."""

    def _do_save() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            train=np.array(train, dtype=object),
            val=np.array(val, dtype=object),
            test=np.array(test, dtype=object),
        )

    get_gatekeeper("file_io").execute(_do_save)
    logger.info("Dataset saved to %s", path)


def load_dataset(path: Path) -> tuple[list, list, list]:
    """Load splits from ``.npz`` via the file-I/O gatekeeper."""

    def _do_load() -> dict:
        return dict(np.load(str(path), allow_pickle=True))

    data = get_gatekeeper("file_io").execute(_do_load)
    return data["train"].tolist(), data["val"].tolist(), data["test"].tolist()


def build_full_dataset(
    config: dict[str, Any], seed: int = 42,
) -> tuple[list, list, list]:
    """End-to-end dataset construction from config.

    Parameters
    ----------
    config : dict
        Parsed ``config/setup.json``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[list, list, list]
        ``(train, val, test)`` lists of normalised dataset entries.
    """
    sig = config["signal"]
    ds = config["dataset"]
    rng = np.random.default_rng(seed)
    gen = SignalGenerator(sig["sampling_rate_hz"], sig["duration_seconds"], sig["amplitude"])
    freqs = sig["frequencies_hz"]
    gen.validate_nyquist(freqs)

    all_entries: list[dict[str, np.ndarray]] = []
    for f in freqs:
        phase = rng.uniform(0, 2 * np.pi)
        clean = gen.generate_clean(float(f), phase)
        noisy = gen.generate_noisy(clean, sig["noise_std_ratio"], rng)
        entries = DatasetBuilder.build_windows(
            clean,
            noisy,
            f,
            sig["window_size"],
            sigma=float(sig["noise_std_ratio"]),
            class_index=int(FREQUENCY_INDEX[f]),
        )
        all_entries.extend(entries)

    train, val, test = DatasetSplitter.split(
        all_entries, ds["train_ratio"], ds["val_ratio"], ds["test_ratio"], rng,
    )

    normalizer = DataNormalizer()
    normalizer.fit(train)
    train = normalizer.transform(train)
    val = normalizer.transform(val)
    test = normalizer.transform(test)

    data_dir = Path(ds["data_dir"])
    save_dataset(data_dir / f"dataset{NPZ_EXT}", train, val, test)

    logger.info("Built dataset: %d train / %d val / %d test", len(train), len(val), len(test))
    return train, val, test
