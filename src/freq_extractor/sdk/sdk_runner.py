"""SDK pipeline runner — ``run_all`` orchestration logic.
Separated from :mod:`sdk.sdk` to comply with the ≤ 145 code-line constraint.
Executes the full generate → train → evaluate → plot pipeline and produces
the Markdown comparison table.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from freq_extractor.constants import MODEL_TYPES

if TYPE_CHECKING:
    from freq_extractor.sdk.sdk import FreqExtractorSDK
logger = logging.getLogger("freq_extractor.sdk.runner")


def _run_training_phase(
    sdk: FreqExtractorSDK,
    train: list,
    val: list,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Train all three architectures and collect results.

    Returns
    -------
    tuple[dict, dict]
        ``(models, histories)`` — both keyed by model type.
    """
    models: dict[str, Any] = {}
    histories: dict[str, dict[str, Any]] = {}
    for mt in MODEL_TYPES:
        model, hist = sdk.train(mt, train, val)
        models[mt] = model
        histories[mt] = hist
    return models, histories


def _run_evaluation_phase(
    sdk: FreqExtractorSDK,
    models: dict[str, Any],
    train: list,
    val: list,
    test: list,
) -> dict[str, dict[str, float]]:
    """Evaluate all trained models on every split.

    Returns
    -------
    dict
        ``{model_type: {"train": mse, "val": mse, "test": mse}}``.
    """
    metrics: dict[str, dict[str, float]] = {}
    for mt, model in models.items():
        metrics[mt] = sdk.evaluate(model, mt, train, val, test)
        logger.info(
            "[%s] test_mse=%.6f", mt.upper(), metrics[mt]["test"],
        )
    return metrics


def _run_plotting_phase(
    sdk: FreqExtractorSDK,
    models: dict[str, Any],
    histories: dict[str, dict[str, Any]],
    test: list,
) -> None:
    """Generate all result plots and save to disk."""
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

    # Training curves
    plot_training_curves(histories, results_dir / "training_curves.png")

    # Per-model predictions
    for mt, model in models.items():
        plot_predictions(model, test, mt, mt, results_dir / f"predictions_{mt}.png", n_ex)

    # Noise robustness sweep
    noise_results: dict[str, dict[float, float]] = {}
    for mt, model in models.items():
        noise_results[mt] = run_noise_robustness(model, cfg, mt)
    plot_noise_robustness(noise_results, results_dir / "noise_robustness.png")

    # Per-frequency MSE
    freq_results: dict[str, dict[int, float]] = {}
    for mt, model in models.items():
        freq_results[mt] = compute_per_frequency_mse(model, test, mt, cfg)
    plot_per_frequency_mse(freq_results, results_dir / "per_frequency_mse.png")

    # Signal examples
    plot_signal_examples(cfg, results_dir / "signal_examples.png")
    logger.info("All plots saved to %s/", results_dir)


def run_full_pipeline(sdk: FreqExtractorSDK) -> dict[str, Any]:
    """Execute the complete generate → train → evaluate → plot pipeline.

    Parameters
    ----------
    sdk : FreqExtractorSDK
        Initialised SDK instance.

    Returns
    -------
    dict
        Summary with keys ``models``, ``histories``, ``metrics``,
        ``comparison_table``.
    """
    from freq_extractor.services.evaluation_service import build_comparison_table

    logger.info("=" * 60)
    logger.info("FULL PIPELINE — seed=%d, device=%s", sdk.seed, sdk.device)
    logger.info("=" * 60)

    # Phase A — Data
    train, val, test = sdk.generate_data()

    # Phase B — Training
    models, histories = _run_training_phase(sdk, train, val)

    # Phase C — Evaluation
    metrics = _run_evaluation_phase(sdk, models, train, val, test)

    # Phase D — Comparison table
    table = build_comparison_table(metrics)
    logger.info("Comparison table:\n%s", table)

    # Phase E — Plots
    _run_plotting_phase(sdk, models, histories, test)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    return {
        "models": models,
        "histories": histories,
        "metrics": metrics,
        "comparison_table": table,
    }
