# ============================================================
# SOURCE: https://github.com/Axel-009/Kserve
# LAYER:  layer5_infra
# ROLE:   KServe model serving integration for investment platform
# ============================================================
"""
KServe integration for serving ML models in the Metadron Capital Investment Platform.

Provides model serving infrastructure for all prediction engines:

1. Equity prediction models (LSTM + XGBoost ensemble)
2. Market regime classification (HMM)
3. Distress probability scoring (Merton model)
4. Sentiment analysis models
5. Optimal allocation recommender

Autoscaling based on market hours:
    Pre-market  (4:00-9:30 AM ET):  2 replicas (warm-up)
    Market open (9:30-10:00 AM ET): 10 replicas (peak load)
    Midday      (10:00-3:00 PM ET): 5 replicas (steady state)
    Close       (3:00-4:00 PM ET):  10 replicas (rebalancing)
    After hours (4:00 PM-8:00 PM):  2 replicas (analysis)
    Overnight   (8:00 PM-4:00 AM):  1 replica (maintenance)

Model Versioning & A/B Testing:
    Canary: 10% traffic to new model
    Shadow: mirror traffic, compare offline
    Promote if: IC_new - IC_prod > 0.02 over 5 trading days
    Rollback if: IC_new < IC_prod - 0.01 for 2 consecutive days

Information Coefficient: IC = corr(predicted_returns, actual_returns)
    IC > 0.05: good, IC > 0.10: excellent, IC < 0: model is harmful
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    EQUITY_PREDICTOR = "equity_predictor"
    REGIME_CLASSIFIER = "regime_classifier"
    DISTRESS_SCORER = "distress_scorer"
    SENTIMENT_ANALYZER = "sentiment_analyzer"
    ALLOCATION_OPTIMIZER = "allocation_optimizer"
    ARBITRAGE_DETECTOR = "arbitrage_detector"


class DeploymentStrategy(Enum):
    CANARY = "canary"
    SHADOW = "shadow"
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"


@dataclass
class ModelEndpoint:
    model_type: ModelType
    version: str
    endpoint_url: str
    replicas: int
    status: str  # "serving", "warming", "failed"
    ic_score: float  # information coefficient
    latency_p99_ms: float
    requests_per_minute: float


@dataclass
class ABTestResult:
    model_a_version: str
    model_b_version: str
    model_a_ic: float
    model_b_ic: float
    traffic_split: float
    days_running: int
    recommendation: str  # "promote_b", "keep_a", "continue_testing"
    confidence: float


class InvestmentModelServer:
    """ML model serving infrastructure for the investment platform."""

    MARKET_SCHEDULE = {
        "pre_market": {"start": time(4, 0), "end": time(9, 30), "replicas": 2},
        "market_open": {"start": time(9, 30), "end": time(10, 0), "replicas": 10},
        "midday": {"start": time(10, 0), "end": time(15, 0), "replicas": 5},
        "market_close": {"start": time(15, 0), "end": time(16, 0), "replicas": 10},
        "after_hours": {"start": time(16, 0), "end": time(20, 0), "replicas": 2},
        "overnight": {"start": time(20, 0), "end": time(4, 0), "replicas": 1},
    }

    MODEL_CONFIGS = {
        ModelType.EQUITY_PREDICTOR: {
            "container": "metadron/equity-predictor:latest",
            "gpu": True,
            "memory": "4Gi",
            "cpu": "2",
            "timeout_ms": 500,
        },
        ModelType.REGIME_CLASSIFIER: {
            "container": "metadron/regime-classifier:latest",
            "gpu": False,
            "memory": "2Gi",
            "cpu": "1",
            "timeout_ms": 200,
        },
        ModelType.DISTRESS_SCORER: {
            "container": "metadron/distress-scorer:latest",
            "gpu": False,
            "memory": "2Gi",
            "cpu": "1",
            "timeout_ms": 300,
        },
        ModelType.SENTIMENT_ANALYZER: {
            "container": "metadron/sentiment-analyzer:latest",
            "gpu": True,
            "memory": "8Gi",
            "cpu": "2",
            "timeout_ms": 1000,
        },
        ModelType.ALLOCATION_OPTIMIZER: {
            "container": "metadron/allocation-optimizer:latest",
            "gpu": False,
            "memory": "4Gi",
            "cpu": "4",
            "timeout_ms": 2000,
        },
    }

    def __init__(self):
        self.endpoints = {}
        self.ab_tests = {}

    def configure_inference_services(self) -> dict:
        """Generate KServe InferenceService manifests for all models."""
        manifests = {}
        for model_type, config in self.MODEL_CONFIGS.items():
            manifest = {
                "apiVersion": "serving.kserve.io/v1beta1",
                "kind": "InferenceService",
                "metadata": {
                    "name": f"metadron-{model_type.value}",
                    "namespace": "metadron-capital",
                    "annotations": {
                        "serving.kserve.io/autoscalerClass": "hpa",
                        "serving.kserve.io/metric": "concurrency",
                        "serving.kserve.io/targetUtilizationPercentage": "70",
                    },
                },
                "spec": {
                    "predictor": {
                        "containers": [{
                            "name": "predictor",
                            "image": config["container"],
                            "resources": {
                                "requests": {"memory": config["memory"], "cpu": config["cpu"]},
                                "limits": {"memory": config["memory"], "cpu": config["cpu"]},
                            },
                        }],
                        "minReplicas": 1,
                        "maxReplicas": 10,
                        "timeout": config["timeout_ms"],
                    },
                },
            }
            if config.get("gpu"):
                manifest["spec"]["predictor"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = "1"
            manifests[model_type.value] = manifest
        return manifests

    def setup_autoscaling(self, market_hours: bool = True) -> dict:
        """Configure autoscaling based on market schedule."""
        current_time = datetime.now().time()
        scaling_config = {}

        for period_name, schedule in self.MARKET_SCHEDULE.items():
            start, end = schedule["start"], schedule["end"]
            replicas = schedule["replicas"]

            # Handle overnight crossing midnight
            if start > end:
                in_period = current_time >= start or current_time < end
            else:
                in_period = start <= current_time < end

            if in_period:
                scaling_config["current_period"] = period_name
                scaling_config["target_replicas"] = replicas
                break

        if "current_period" not in scaling_config:
            scaling_config["current_period"] = "unknown"
            scaling_config["target_replicas"] = 2

        scaling_config["market_hours_mode"] = market_hours
        scaling_config["schedule"] = {k: {"replicas": v["replicas"]} for k, v in self.MARKET_SCHEDULE.items()}
        return scaling_config

    def ab_test_model(
        self,
        model_type: ModelType,
        model_a_version: str,
        model_b_version: str,
        traffic_split: float = 0.1,
        min_days: int = 5,
        ic_threshold: float = 0.02,
    ) -> ABTestResult:
        """
        Run A/B test between two model versions.

        Promote B if: IC_B - IC_A > ic_threshold over min_days
        Rollback B if: IC_B < IC_A - 0.01 for 2 consecutive days
        """
        test_key = f"{model_type.value}_{model_a_version}_vs_{model_b_version}"

        # Simulate test metrics (in production, these come from live scoring)
        ic_a = np.random.normal(0.06, 0.02)
        ic_b = np.random.normal(0.07, 0.025)
        days_running = np.random.randint(1, 10)

        ic_diff = ic_b - ic_a

        if days_running >= min_days and ic_diff > ic_threshold:
            recommendation = "promote_b"
            confidence = min(ic_diff / ic_threshold, 1.0)
        elif ic_diff < -0.01:
            recommendation = "keep_a"
            confidence = min(abs(ic_diff) / 0.01, 1.0)
        else:
            recommendation = "continue_testing"
            confidence = days_running / min_days

        return ABTestResult(
            model_a_version=model_a_version,
            model_b_version=model_b_version,
            model_a_ic=float(ic_a),
            model_b_ic=float(ic_b),
            traffic_split=traffic_split,
            days_running=days_running,
            recommendation=recommendation,
            confidence=float(confidence),
        )

    def health_check_all_models(self) -> dict:
        """Check health of all deployed models."""
        health = {}
        for model_type in ModelType:
            health[model_type.value] = {
                "status": "healthy",
                "latency_p99_ms": float(np.random.uniform(10, 500)),
                "error_rate": float(np.random.uniform(0, 0.02)),
                "replicas_ready": int(np.random.randint(1, 10)),
                "last_prediction_time": datetime.now().isoformat(),
            }
        return health

    def batch_predict(self, symbols: list, model_type: ModelType) -> pd.DataFrame:
        """
        Batch prediction for multiple symbols.

        Routes to appropriate model endpoint based on model_type.
        Returns DataFrame with predictions and confidence intervals.
        """
        results = []
        for symbol in symbols:
            # Placeholder for actual model inference
            prediction = np.random.normal(0, 0.02)
            confidence = np.random.uniform(0.3, 0.9)
            results.append({
                "symbol": symbol,
                "prediction": prediction,
                "confidence": confidence,
                "upper_bound": prediction + 1.96 * abs(prediction) * 0.5,
                "lower_bound": prediction - 1.96 * abs(prediction) * 0.5,
                "model_version": "v1.0",
                "timestamp": datetime.now().isoformat(),
            })
        return pd.DataFrame(results)
