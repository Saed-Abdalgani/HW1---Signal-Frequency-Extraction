"""Plotting phase for :func:`~freq_extractor.sdk.sdk_runner.run_full_pipeline`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from freq_extractor.sdk.sdk import FreqExtractorSDK

logger = logging.getLogger("freq_extractor.sdk.runner")


def run_plotting_phase(
    sdk: FreqExtractorSDK,
    models: dict[str, Any],
    histories: dict[str, dict[str, Any]],
    test: list,
) -> None:
    """Generate all result plots and save them under ``evaluation.results_dir``.

    Parameters
    ----------
    sdk
        SDK with loaded ``config``.
    models, histories
        Trained models and their training histories keyed by model type.
    test
        Test split entries for prediction and robustness plots.
    """
    from freq_extractor.services.eval_robustness import run_noise_robustness
    from freq_extractor.services.evaluation_service import compute_per_frequency_mse
    from freq_extractor.services.plot_analysis import (
        plot_noise_robustness,
        plot_per_frequency_mse,
        plot_signal_examples,
    )
    from freq_extractor.services.plot_helpers import plot_predictions, plot_training_curves

    cfg = sdk.config
    results_dir = Path(cfg["evaluation"]["results_dir"])
    n_ex = cfg["evaluation"]["n_prediction_examples"]

    plot_training_curves(histories, results_dir / "training_curves.png")

    for mt, model in models.items():
        plot_predictions(model, test, mt, mt, results_dir / f"predictions_{mt}.png", n_ex)

    noise_results: dict[str, dict[float, float]] = {}
    for mt, model in models.items():
        noise_results[mt] = run_noise_robustness(model, cfg, mt)
    plot_noise_robustness(noise_results, results_dir / "noise_robustness.png")

    freq_results: dict[str, dict[int, float]] = {}
    for mt, model in models.items():
        freq_results[mt] = compute_per_frequency_mse(model, test, mt, cfg)
    plot_per_frequency_mse(freq_results, results_dir / "per_frequency_mse.png")

    plot_signal_examples(cfg, results_dir / "signal_examples.png")
    logger.info("All plots saved to %s/", results_dir)
