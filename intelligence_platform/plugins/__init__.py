"""
Multi-Language Integration Plugins for Metadron Capital Intelligence Platform.

Provides Python bindings and integration layers for non-Python components:
- Rust (AI-Newton symbolic physics engine)
- Go (Kserve ML model serving)
- C++/CUDA (NVIDIA deep learning optimizations)
- TypeScript/JavaScript (Frontend dashboards, React/Vue apps)
"""

from .rust_plugin import RustIntegration
from .go_plugin import GoIntegration
from .cuda_plugin import CUDAIntegration
from .typescript_plugin import TypeScriptIntegration
from .unified_bridge import UnifiedLanguageBridge

__all__ = [
    "RustIntegration",
    "GoIntegration",
    "CUDAIntegration",
    "TypeScriptIntegration",
    "UnifiedLanguageBridge",
]
