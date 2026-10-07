"""Sofia core: contracts, units, time, quality, validation and errors.

This package is the dependency floor of Sofia Engine. It imports only the standard
library and numpy. Nothing here may import telemetry, inference, diagnostics or any
transport/ML framework — that rule is enforced by
``tests/architecture/test_boundaries.py``.
"""

from __future__ import annotations

__all__ = [
    "contracts",
    "errors",
    "hashing",
    "quality",
    "serialization",
    "time",
    "units",
    "validation",
]
