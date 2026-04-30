"""freq_extractor: Signal frequency extraction using MLP, RNN, and LSTM.

This is the top-level package.  All business logic is accessed through
:mod:`freq_extractor.sdk`; shared infrastructure is in
:mod:`freq_extractor.shared`; signal-independent constants live in
:mod:`freq_extractor.constants`.

Example
-------
>>> from freq_extractor import __version__
>>> print(__version__)
'1.00'
"""

from __future__ import annotations

from freq_extractor.shared.version import CODE_VERSION

__version__: str = CODE_VERSION
__author__: str = "Saed Abdalgani"
__all__ = [
    "__author__",
    "__version__",
    "CODE_VERSION",
]
