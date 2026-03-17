"""OpenBB Bridge — Thin adapter imported by Metadron-Capital engine."""
import sys
from pathlib import Path

# Add backend to path
_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backends.openbb.openbb_backend import OpenBBBackend

__all__ = ["OpenBBBackend"]
