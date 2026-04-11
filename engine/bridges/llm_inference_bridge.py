"""
Metadron Capital — Unified LLM Inference Bridge

Provides a single interface for all LLM inference across the platform,
routing through Brain Power (Xiaomi Mimo V2 Pro) as the primary backend
and Air-LLM (meta-llama/Llama-3.1-70B) as a local fallback.

PM2 manages this as a persistent FastAPI service on port 8002.
"""

import os
import sys
import json
import time
import signal
import logging
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("llm-bridge")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── Brain Power Backend ─────────────────────────────────────────────

try:
    from engine.bridges.brain_power import BrainPowerClient
    _brain_power_available = True
except ImportError:
    _brain_power_available = False
    logger.warning("BrainPowerClient not available — brain power fallback disabled")

# ─── Air-LLM In-Process Backend ──────────────────────────────────────

try:
    from engine.bridges.airllm_model_server import AirLLMModelManager
    _airllm_available = True
except ImportError:
    _airllm_available = False
    logger.warning("AirLLMModelManager not available — Air-LLM fallback disabled")


@dataclass
class LLMBackend:
    name: str
    backend_type: str  # "brain_power" | "airllm"
    available: bool = False
    model_id: str = ""
    capabilities: list = field(default_factory=list)
    avg_latency_ms: float = 0.0
    request_count: int = 0
    error_count: int = 0

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count


class LLMInferenceBridge:
    """Unified LLM inference — Brain Power primary, Air-LLM local fallback."""

    def __init__(self):
        self.backends: dict[str, LLMBackend] = {}
        self._brain_power_client: Optional[BrainPowerClient] = None
        self._airllm_manager: Optional[AirLLMModelManager] = None
        self._initialize_backends()

    def _initialize_backends(self):
        """Register Brain Power as primary and Air-LLM as local fallback."""
        self.backends["brain_power"] = LLMBackend(
            name="Brain Power (Xiaomi Mimo V2 Pro)",
            backend_type="brain_power",
            model_id="xiaomi-mimo-v2-pro",
            capabilities=["text", "reasoning", "code", "analysis", "long_context",
                          "sentiment", "earnings", "sec_filing", "trade_thesis", "narrative"],
        )
        self.backends["airllm"] = LLMBackend(
            name="Air-LLM (Llama-3.1-70B)",
            backend_type="airllm",
            model_id=os.environ.get("AIRLLM_MODEL_PATH", "meta-llama/Llama-3.1-70B"),
            capabilities=["text", "reasoning", "code", "analysis"],
        )
        self._probe_backends()

    def _probe_backends(self):
        """Check backend availability."""
        if _brain_power_available:
            self._brain_power_client = BrainPowerClient()
            self.backends["brain_power"].available = True
            if self._brain_power_client.is_stub:
                logger.info("Brain Power backend: STUB MODE (key not configured)")
            else:
                logger.info("Brain Power backend: AVAILABLE")
        else:
            logger.warning("Brain Power backend: BrainPowerClient import failed")

        if _airllm_available:
            self._airllm_manager = AirLLMModelManager()
            self.backends["airllm"].available = True
            logger.info("Air-LLM backend: AVAILABLE (lazy-load on first request)")
        else:
            logger.warning("Air-LLM backend: AirLLMModelManager import failed")

    def select_backend(self, task_type: str, preferred: Optional[str] = None) -> Optional[str]:
        """Select backend — prefer the requested one, then Brain Power, then Air-LLM."""
        if preferred and self.backends.get(preferred, LLMBackend(name="", backend_type="")).available:
            return preferred
        if self.backends.get("brain_power", LLMBackend(name="", backend_type="")).available:
            return "brain_power"
        if self.backends.get("airllm", LLMBackend(name="", backend_type="")).available:
            return "airllm"
        return None

    async def infer(
        self,
        prompt: str,
        task_type: str = "text",
        preferred_backend: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        images: Optional[list] = None,
        audio: Optional[str] = None,
    ) -> dict:
        """Run inference through selected backend.

        Returns:
            {
                "text": str,
                "backend": str,
                "latency_ms": float,
                "tokens_used": int,
                "task_type": str,
            }
        """
        backend_name = self.select_backend(task_type, preferred_backend)
        if not backend_name:
            return {
                "text": "",
                "backend": "none",
                "error": "No LLM backend available",
                "latency_ms": 0,
                "tokens_used": 0,
                "task_type": task_type,
            }

        backend = self.backends[backend_name]
        start_time = time.time()

        try:
            if backend_name == "brain_power":
                result = await self._infer_brain_power(prompt, max_tokens, temperature, system_prompt)
            elif backend_name == "airllm":
                result = await self._infer_airllm(prompt, max_tokens, temperature)
            else:
                raise RuntimeError(f"Unknown backend: {backend_name}")

            latency_ms = (time.time() - start_time) * 1000
            backend.request_count += 1
            backend.avg_latency_ms = (
                (backend.avg_latency_ms * (backend.request_count - 1) + latency_ms)
                / backend.request_count
            )

            return {
                "text": result.get("text", ""),
                "backend": backend_name,
                "latency_ms": round(latency_ms, 1),
                "tokens_used": result.get("tokens_used", 0),
                "task_type": task_type,
                "stub": result.get("stub", False),
            }

        except Exception as e:
            backend.request_count += 1
            backend.error_count += 1
            logger.error(f"{backend_name} inference error: {e}")

            # Try fallback if primary failed
            fallback_name = None
            if backend_name == "brain_power" and self.backends.get("airllm", LLMBackend(name="", backend_type="")).available:
                fallback_name = "airllm"
            elif backend_name == "airllm" and self.backends.get("brain_power", LLMBackend(name="", backend_type="")).available:
                fallback_name = "brain_power"

            if fallback_name:
                logger.info(f"Falling back to {fallback_name}")
                try:
                    if fallback_name == "brain_power":
                        result = await self._infer_brain_power(prompt, max_tokens, temperature, system_prompt)
                    else:
                        result = await self._infer_airllm(prompt, max_tokens, temperature)
                    fb = self.backends[fallback_name]
                    fb.request_count += 1
                    latency_ms = (time.time() - start_time) * 1000
                    return {
                        "text": result.get("text", ""),
                        "backend": fallback_name,
                        "latency_ms": round(latency_ms, 1),
                        "tokens_used": result.get("tokens_used", 0),
                        "task_type": task_type,
                        "fallback_from": backend_name,
                    }
                except Exception as e2:
                    logger.error(f"Fallback {fallback_name} also failed: {e2}")

            return {
                "text": "",
                "backend": backend_name,
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000,
                "tokens_used": 0,
                "task_type": task_type,
            }

    async def _infer_brain_power(
        self, prompt: str, max_tokens: int, temperature: float,
        system_prompt: Optional[str],
    ) -> dict:
        """Inference via Brain Power (Xiaomi Mimo V2 Pro)."""
        if not self._brain_power_client:
            raise RuntimeError("Brain Power client not initialized")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        result = await asyncio.to_thread(
            self._brain_power_client.chat,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        usage = result.get("usage", {})
        return {
            "text": result.get("text", ""),
            "tokens_used": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            "stub": result.get("stub", False),
        }

    async def _infer_airllm(
        self, prompt: str, max_tokens: int, temperature: float,
    ) -> dict:
        """Inference via Air-LLM in-process (Llama-3.1-70B layer-by-layer)."""
        if not self._airllm_manager:
            raise RuntimeError("Air-LLM manager not initialized")

        result = await asyncio.to_thread(
            self._airllm_manager.generate,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "text": result.get("text", ""),
            "tokens_used": result.get("tokens_generated", 0),
        }

    def get_status(self) -> dict:
        """Return status of all backends."""
        return {
            name: {
                "available": b.available,
                "model_id": b.model_id,
                "capabilities": b.capabilities,
                "request_count": b.request_count,
                "error_rate": round(b.error_rate, 3),
                "avg_latency_ms": round(b.avg_latency_ms, 1),
            }
            for name, b in self.backends.items()
        }


# ─── FastAPI Service (PM2 managed) ───────────────────────────────────

def create_app():
    """Create FastAPI app for the LLM inference bridge."""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(
        title="Metadron LLM Inference Bridge",
        description="Unified LLM inference via Brain Power (Xiaomi Mimo V2 Pro) + Air-LLM (Llama-3.1-70B)",
        version="2.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    bridge = LLMInferenceBridge()

    class InferenceRequest(BaseModel):
        prompt: str
        task_type: str = "text"
        preferred_backend: Optional[str] = None
        max_tokens: int = 2048
        temperature: float = 0.3
        system_prompt: Optional[str] = None

    @app.get("/health")
    async def health():
        return {"status": "healthy", "backends": bridge.get_status()}

    @app.post("/infer")
    async def infer(request: InferenceRequest):
        return await bridge.infer(
            prompt=request.prompt,
            task_type=request.task_type,
            preferred_backend=request.preferred_backend,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt,
        )

    @app.get("/backends")
    async def backends():
        return bridge.get_status()

    # ─── Investment-Specific Endpoints ─────────────────────────

    @app.post("/analyze/sentiment")
    async def analyze_sentiment(request: InferenceRequest):
        request.task_type = "sentiment"
        request.system_prompt = (
            "You are a financial sentiment analyst. Analyze the following text and return "
            "a JSON object with: sentiment (-1.0 to 1.0), confidence (0-1), key_phrases (list), "
            "market_impact (bearish/neutral/bullish), and reasoning (brief)."
        )
        return await bridge.infer(**request.model_dump())

    @app.post("/analyze/earnings")
    async def analyze_earnings(request: InferenceRequest):
        request.task_type = "earnings"
        request.system_prompt = (
            "You are an expert earnings analyst. Analyze the earnings transcript/report and return "
            "a JSON with: revenue_surprise (%), eps_surprise (%), guidance (raised/maintained/lowered), "
            "key_themes (list), risk_factors (list), and overall_signal (bullish/neutral/bearish)."
        )
        return await bridge.infer(**request.model_dump())

    @app.post("/analyze/sec_filing")
    async def analyze_sec_filing(request: InferenceRequest):
        request.task_type = "sec_filing"
        request.system_prompt = (
            "You are an SEC filing analyst. Parse the filing and extract: filing_type, "
            "material_changes (list), risk_factors (list with severity), insider_transactions, "
            "related_party_transactions, and investment_implications."
        )
        return await bridge.infer(**request.model_dump())

    @app.post("/generate/trade_thesis")
    async def generate_trade_thesis(request: InferenceRequest):
        request.task_type = "trade_thesis"
        request.system_prompt = (
            "You are a quantitative portfolio manager at Metadron Capital. Generate a trade thesis "
            "with: ticker, direction (long/short), conviction (1-10), entry_price, stop_loss, "
            "target_price, timeframe, catalyst, risk_factors, and position_size_suggestion."
        )
        return await bridge.infer(**request.model_dump())

    @app.post("/generate/narrative")
    async def generate_narrative(request: InferenceRequest):
        request.task_type = "narrative"
        request.system_prompt = (
            "You are the chief strategist at Metadron Capital. Generate a market narrative "
            "covering: regime_assessment, macro_outlook, sector_rotation, key_risks, "
            "opportunities, and portfolio_positioning. Be concise and actionable."
        )
        return await bridge.infer(**request.model_dump())

    return app


# ─── Entry Point ──────────────────────────────────────────────────────

def main():
    """Run the LLM inference bridge as a PM2-managed service."""
    import uvicorn

    port = int(os.environ.get("LLM_BRIDGE_PORT", "8002"))
    host = os.environ.get("LLM_BRIDGE_HOST", "0.0.0.0")

    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Starting LLM Inference Bridge on {host}:{port}")
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
