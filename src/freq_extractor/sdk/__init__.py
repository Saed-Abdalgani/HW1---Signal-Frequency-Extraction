"""SDK layer for freq_extractor.

:class:`FreqExtractorSDK` (implemented in Phase 5) is the **single entry
point** for all business logic.  External callers — including ``src/main.py``
— must only call methods on this class; direct imports of service modules are
prohibited (PLAN §1, ADR-4).

This ``__init__.py`` is intentionally kept thin.  After Phase 5 the SDK
class will be re-exported here so callers can do::

    from freq_extractor.sdk import FreqExtractorSDK
"""

from __future__ import annotations

# FreqExtractorSDK will be added here once sdk.py is implemented (Phase 5).
__all__: list[str] = []
