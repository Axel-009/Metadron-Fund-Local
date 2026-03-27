"""
ModelStore — Simple model persistence for Metadron Capital ML pipeline.

Saves and loads:
- sklearn models (via joblib)
- numpy weight arrays (via np.save)
- Training metadata (JSON)

Used by: AlphaOptimizer, UniverseClassifier, MLVoteEnsemble, PatternRecognition
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

# joblib for sklearn models
try:
    import joblib
except ImportError:
    joblib = None


class ModelStore:
    """Persistent model storage with versioning."""

    def __init__(self, base_dir: str = "data/models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_sklearn(self, name: str, model: Any, metadata: Optional[dict] = None) -> str:
        """Save sklearn model with metadata."""
        if joblib is None:
            logger.warning("joblib not available — cannot save sklearn model")
            return ""

        model_dir = self.base_dir / name
        model_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / f"{timestamp}_model.joblib"
        meta_path = model_dir / f"{timestamp}_meta.json"

        joblib.dump(model, model_path)

        meta = {
            "model_name": name,
            "timestamp": timestamp,
            "model_type": type(model).__name__,
            **(metadata or {}),
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

        logger.info("Saved model %s to %s", name, model_path)
        return str(model_path)

    def load_sklearn(self, name: str) -> tuple:
        """Load latest sklearn model + metadata."""
        if joblib is None:
            return None, None

        model_dir = self.base_dir / name
        if not model_dir.exists():
            return None, None

        joblib_files = sorted(model_dir.glob("*_model.joblib"))
        if not joblib_files:
            return None, None

        latest = joblib_files[-1]
        meta_path = Path(str(latest).replace("_model.joblib", "_meta.json"))

        model = joblib.load(latest)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return model, meta

    def save_numpy(self, name: str, weights: np.ndarray, metadata: Optional[dict] = None) -> str:
        """Save numpy weight array."""
        model_dir = self.base_dir / name
        model_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weights_path = model_dir / f"{timestamp}_weights.npy"
        meta_path = model_dir / f"{timestamp}_meta.json"

        np.save(weights_path, weights)

        meta = {
            "model_name": name,
            "timestamp": timestamp,
            "shape": list(weights.shape),
            "dtype": str(weights.dtype),
            **(metadata or {}),
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=str))

        logger.info("Saved weights %s to %s", name, weights_path)
        return str(weights_path)

    def load_numpy(self, name: str) -> tuple:
        """Load latest numpy weights + metadata."""
        model_dir = self.base_dir / name
        if not model_dir.exists():
            return None, None

        npy_files = sorted(model_dir.glob("*_weights.npy"))
        if not npy_files:
            return None, None

        latest = npy_files[-1]
        meta_path = Path(str(latest).replace("_weights.npy", "_meta.json"))

        weights = np.load(latest)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return weights, meta

    def list_models(self) -> Dict[str, list]:
        """List all stored models with versions."""
        result = {}
        if not self.base_dir.exists():
            return result

        for model_dir in sorted(self.base_dir.iterdir()):
            if model_dir.is_dir():
                versions = []
                for f in sorted(model_dir.glob("*_meta.json")):
                    meta = json.loads(f.read_text())
                    versions.append(meta)
                result[model_dir.name] = versions

        return result

    def cleanup(self, name: str, keep_last: int = 5):
        """Remove old model versions, keep latest N."""
        model_dir = self.base_dir / name
        if not model_dir.exists():
            return

        # Find all version timestamps
        timestamps = set()
        for f in model_dir.iterdir():
            parts = f.name.split("_")
            if len(parts) >= 2:
                timestamps.add(parts[0])

        # Sort and remove old ones
        sorted_ts = sorted(timestamps)
        if len(sorted_ts) > keep_last:
            for ts in sorted_ts[:-keep_last]:
                for f in model_dir.glob(f"{ts}_*"):
                    f.unlink()
                logger.info("Cleaned up old model version: %s/%s", name, ts)
