"""SDK pipeline runner — ``run_all`` orchestration logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from freq_extractor.constants import MODEL_TYPES
from freq_extractor.sdk.sdk_runner_plots import run_plotting_phase

if TYPE_CHECKING:
    from freq_extractor.sdk.sdk import FreqExtractorSDK

logger = logging.getLogger("freq_extractor.sdk.runner")


def _run_training_phase(
    sdk: FreqExtractorSDK, train: list, val: list,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Train all three architectures."""
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
    """Evaluate all models on every split."""
    metrics: dict[str, dict[str, float]] = {}
    for mt, model in models.items():
        metrics[mt] = sdk.evaluate(model, mt, train, val, test)
        logger.info("[%s] test_mse=%.6f", mt.upper(), metrics[mt]["test"])
    return metrics


def run_full_pipeline(sdk: FreqExtractorSDK) -> dict[str, Any]:
    """Execute generate, train, evaluate, and plot pipeline."""
    from freq_extractor.services.evaluation_service import build_comparison_table

    logger.info("=" * 60)
    logger.info("FULL PIPELINE - seed=%d, device=%s", sdk.seed, sdk.device)
    logger.info("=" * 60)

    train, val, test = sdk.generate_data()
    models, histories = _run_training_phase(sdk, train, val)
    metrics = _run_evaluation_phase(sdk, models, train, val, test)
    table = build_comparison_table(metrics)
    logger.info("Comparison table:\n%s", table)

    run_plotting_phase(sdk, models, histories, test)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    return {
        "models": models,
        "histories": histories,
        "metrics": metrics,
        "comparison_table": table,
    }
