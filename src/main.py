"""CLI entry point for freq_extractor.

Provides ``--mode`` dispatch for the full pipeline, individual model
training, evaluation, or the interactive Sinusoid Explorer dashboard.

All business logic is accessed exclusively through :class:`FreqExtractorSDK`
— no direct service imports are permitted (PLAN §1, ADR-4).

Usage Examples
--------------
.. code-block:: bash

    # Full pipeline (generate → train all → evaluate → plot)
    uv run python src/main.py --mode all

    # Train a single model
    uv run python src/main.py --mode train --model lstm

    # Launch the interactive dashboard
    uv run python src/main.py --mode ui --port 8050

References
----------
- PRD FR-8, FR-13, PLAN §1
"""

from __future__ import annotations

import argparse
import logging
import sys

from freq_extractor.constants import MODEL_TYPES

logger = logging.getLogger("freq_extractor.cli")

# Valid CLI modes
_MODES = ("all", "generate", "train", "evaluate", "ui")


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with full validation.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with mode, model, seed, and port arguments.
    """
    parser = argparse.ArgumentParser(
        prog="freq_extractor",
        description="Signal Frequency Extraction using MLP, RNN, and LSTM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/main.py --mode all\n"
            "  python src/main.py --mode train --model lstm --seed 123\n"
            "  python src/main.py --mode ui --port 8050\n"
        ),
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=_MODES,
        help="Execution mode: all | generate | train | evaluate | ui",
    )
    parser.add_argument(
        "--model", type=str, default=None, choices=list(MODEL_TYPES),
        help="Model type for train/evaluate modes (default: all three).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed override (default: from config/setup.json).",
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Port for the UI dashboard (default: 8050).",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable Dash debug reloader/hot reload (use with --mode ui).",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Raise ``SystemExit`` on invalid argument combinations."""
    if args.mode in ("train", "evaluate") and args.model:
        model_lower = args.model.lower().strip()
        if model_lower not in MODEL_TYPES:
            sys.exit(f"Error: --model must be one of {MODEL_TYPES}, got '{args.model}'")
    if args.mode == "ui" and args.port <= 0:
        sys.exit(f"Error: --port must be > 0, got {args.port}")


def _dispatch(args: argparse.Namespace) -> None:
    """Route to the correct SDK method based on --mode."""
    from freq_extractor.sdk import FreqExtractorSDK

    sdk = FreqExtractorSDK(seed=args.seed)

    if args.mode == "all":
        result = sdk.run_all()
        print(f"\n{result['comparison_table']}")

    elif args.mode == "generate":
        sdk.generate_data()
        print("Dataset generation complete.")

    elif args.mode == "train":
        train, val, _test = sdk.generate_data()
        models = [args.model] if args.model else list(MODEL_TYPES)
        for mt in models:
            _model, hist = sdk.train(mt, train, val)
            print(f"[{mt.upper()}] best_val_mse={hist['best_val_mse']:.6f}")

    elif args.mode == "evaluate":
        train, val, test = sdk.generate_data()
        models_to_eval = [args.model] if args.model else list(MODEL_TYPES)
        for mt in models_to_eval:
            model, _hist = sdk.train(mt, train, val)
            metrics = sdk.evaluate(model, mt, train, val, test)
            print(f"[{mt.upper()}] test_mse={metrics['test']:.6f}")

    elif args.mode == "ui":
        sdk.launch_ui(port=args.port, debug=not args.no_reload)


def main() -> None:
    """Parse arguments and execute."""
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)
    logger.info("CLI started - mode=%s, model=%s, seed=%s", args.mode, args.model, args.seed)
    _dispatch(args)
    logger.info("CLI finished.")


if __name__ == "__main__":
    main()
