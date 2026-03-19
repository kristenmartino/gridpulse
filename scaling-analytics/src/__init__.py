"""
WattCast v2 src package.

Adds the v1 project root to sys.path so v1 modules (data/, models/, config)
can be imported without modification.
"""
import sys
from pathlib import Path

# v1 project root: scaling-analytics/src/__init__.py -> ../../ = energy-forecast-final/
_V1_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _V1_ROOT not in sys.path:
    sys.path.insert(0, _V1_ROOT)
