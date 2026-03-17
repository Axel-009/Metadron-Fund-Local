"""Air-LLM Bridge — Thin adapter imported by Metadron-Capital engine."""
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backends.airllm.airllm_backend import AirLLMBackend

__all__ = ["AirLLMBackend"]
