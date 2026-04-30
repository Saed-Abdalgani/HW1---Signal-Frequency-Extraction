"""Service modules for freq_extractor.

Each service (DataService, model classes, TrainingService, EvaluationService,
UIService) lives in its own module within this package.  They are imported
**only** by :mod:`freq_extractor.sdk.sdk`, never directly by external code.

This ``__init__.py`` intentionally exposes **no** public symbols so that the
SDK remains the sole entry point for callers (PLAN §1).
"""

from __future__ import annotations

__all__: list[str] = []
