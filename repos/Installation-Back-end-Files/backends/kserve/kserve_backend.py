"""KServe Backend — Kubernetes ML Model Serving.

Provides model serving infrastructure for Metadron Capital ML models
via KServe (formerly KFServing). Supports:
    - Model deployment and versioning
    - Auto-scaling based on inference load
    - A/B testing and canary rollouts
    - Multi-model serving (alpha optimizer, regime classifier, etc.)

Dependencies:
    pip install kserve

Usage:
    from backends.kserve.kserve_backend import KServeBackend
    backend = KServeBackend()
    backend.serve_model("alpha_optimizer", model_obj)
"""

import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

try:
    from kserve import Model, ModelServer, InferRequest, InferResponse
    _HAS_KSERVE = True
    logger.info("KServe loaded successfully")
except ImportError:
    _HAS_KSERVE = False
    Model = None
    logger.warning("KServe not available — install with: pip install kserve")

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "kserve"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelMetadata:
    """Metadata for a served model."""
    name: str = ""
    version: str = "v1"
    framework: str = "custom"
    input_shape: list = field(default_factory=list)
    output_shape: list = field(default_factory=list)
    description: str = ""


class MetadronModel:
    """Base KServe model wrapper for Metadron Capital ML models.

    Wraps any sklearn-compatible model (LightGBM, XGBoost, linear)
    into a KServe-compatible inference server.
    """

    def __init__(self, name: str, model: Any = None):
        self.name = name
        self._model = model
        self._ready = model is not None

        if _HAS_KSERVE and Model is not None:
            self._kserve_model = _create_kserve_model(name, model)
        else:
            self._kserve_model = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference on input features."""
        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' not loaded")

        if hasattr(self._model, 'predict'):
            return np.array(self._model.predict(features))
        else:
            raise RuntimeError(f"Model '{self.name}' has no predict method")

    def load(self, path: Optional[str] = None):
        """Load model from disk."""
        model_path = Path(path) if path else MODELS_DIR / self.name
        if not model_path.exists():
            logger.warning(f"Model path not found: {model_path}")
            return False

        try:
            import joblib
            self._model = joblib.load(model_path / "model.pkl")
            self._ready = True
            logger.info(f"Model '{self.name}' loaded from {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model '{self.name}': {e}")
            return False

    def save(self, path: Optional[str] = None):
        """Save model to disk."""
        model_path = Path(path) if path else MODELS_DIR / self.name
        model_path.mkdir(parents=True, exist_ok=True)

        try:
            import joblib
            joblib.dump(self._model, model_path / "model.pkl")
            logger.info(f"Model '{self.name}' saved to {model_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save model '{self.name}': {e}")
            return False

    @property
    def ready(self) -> bool:
        return self._ready


class KServeBackend:
    """KServe backend for deploying Metadron Capital ML models.

    In production: serves models via KServe on Kubernetes.
    In dev/paper mode: serves models locally with the same API.
    Local mode is fully functional — register, predict, save/load all work
    without a K8s cluster. Only serve() requires K8s.
    """

    def __init__(self):
        self._models: dict[str, MetadronModel] = {}
        self._server = None
        self._local_mode = not _HAS_KSERVE
        mode = "local" if self._local_mode else "kserve"
        logger.info(f"KServe backend initialized (mode={mode}, kserve_sdk={_HAS_KSERVE})")

    def register_model(self, name: str, model: Any,
                        metadata: Optional[ModelMetadata] = None) -> MetadronModel:
        """Register a model for serving."""
        wrapped = MetadronModel(name=name, model=model)
        self._models[name] = wrapped
        logger.info(f"Model '{name}' registered for serving")
        return wrapped

    def get_model(self, name: str) -> Optional[MetadronModel]:
        """Get a registered model by name."""
        return self._models.get(name)

    def predict(self, model_name: str, features: np.ndarray) -> np.ndarray:
        """Run inference on a registered model."""
        model = self._models.get(model_name)
        if model is None:
            raise KeyError(f"Model '{model_name}' not registered")
        return model.predict(features)

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())

    def serve(self, port: int = 8080):
        """Start the model server (requires KServe + Kubernetes)."""
        if not _HAS_KSERVE:
            logger.warning("KServe not available — cannot start model server")
            return

        if not self._models:
            logger.warning("No models registered — nothing to serve")
            return

        try:
            server = ModelServer()
            for name, model in self._models.items():
                if model._kserve_model is not None:
                    server.start([model._kserve_model])
            logger.info(f"KServe model server started on port {port}")
        except Exception as e:
            logger.warning(f"KServe server start failed: {e}")


def _create_kserve_model(name: str, model: Any):
    """Create a KServe Model instance wrapping an ML model."""
    if not _HAS_KSERVE or Model is None:
        return None

    class _WrappedModel(Model):
        def __init__(self, name, ml_model):
            super().__init__(name)
            self.ml_model = ml_model
            self.ready = ml_model is not None

        def predict(self, payload, headers=None, **kwargs):
            if self.ml_model is None:
                return {"error": "model not loaded"}
            try:
                features = np.array(payload.get("instances", []))
                predictions = self.ml_model.predict(features)
                return {"predictions": predictions.tolist()}
            except Exception as e:
                return {"error": str(e)}

    try:
        return _WrappedModel(name, model)
    except Exception as e:
        logger.warning(f"Failed to create KServe model wrapper: {e}")
        return None
