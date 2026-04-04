from __future__ import annotations

import sys
from pathlib import Path

_PKG = str(Path(__file__).resolve().parent)
_ROOT = str(Path(__file__).resolve().parents[1])
for _p in (_PKG, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ["__version__"]

__version__ = "0.1.0"
