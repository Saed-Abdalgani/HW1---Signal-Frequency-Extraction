"""SDK layer for freq_extractor.

:class:`FreqExtractorSDK` is the **single entry point** for all business
logic.  External callers — including ``src/main.py`` — must only call
methods on this class; direct imports of service modules are prohibited
(PLAN §1, ADR-4).

Usage::

    from freq_extractor.sdk import FreqExtractorSDK

    sdk = FreqExtractorSDK()
    sdk.run_all()
"""

from __future__ import annotations

from freq_extractor.sdk.sdk import FreqExtractorSDK

__all__: list[str] = ["FreqExtractorSDK"]
