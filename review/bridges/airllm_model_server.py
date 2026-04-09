"""
Metadron Capital — Air-LLM Standalone Model Server

Provides a dedicated FastAPI service for Air-LLM inference on port 8003.
Loads meta-llama/Llama-3.1-70B (or configurable model) using the Air-LLM
layer-by-layer inference engine for efficient 70B+ models on limited VRAM.

PM2 manages this as a persistent service. The LLM Inference Bridge can
also load Air-LLM in-process, but this standalone server avoids
double-loading and allows independent scaling/GPU assignment.

Usage:
    python3 -m uvicorn engine.bridges.airllm_model_server:create_app \
        --factory --host 0.0.0.0 --port 8003 --log-level info
"""

import os
import sys
import time
import signal
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("airllm-server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

PLATFORM_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── Air-LLM Availability ──────────────────────────────────────────

_airllm_available = False
try:
    sys.path.insert(0, str(PLATFORM_ROOT / "intelligence_platform" / "Air-LLM" / "air_llm"))
    import airllm
    _airllm_available = True
    logger.info("Air-LLM library: AVAILABLE")
except ImportError:
    logger.warning("Air-LLM library not found — server will start in degraded mode")


# ─── Model Manager ─────────────────────────────────────────────────

class AirLLMModelManager:
    """Manages lazy-loading and inference for a single Air-LLM model."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id: str = os.environ.get("AIRLLM_MODEL_PATH", "meta-llama/Llama-3.1-70B")
        self.loaded: bool = False
        self.load_time_s: float = 0.0
        self.request_count: int = 0
        self.error_count: int = 0
        self.total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    def load(self):
        """Lazy-load the Air-LLM model on first request."""
        if self.loaded:
            return

        if not _airllm_available:
            raise RuntimeError(
                "Air-LLM library not installed. "
                "Ensure intelligence_platform/Air-LLM/air_llm is on sys.path."
            )

        logger.info(f"Loading Air-LLM model: {self.model_id} ...")
        start = time.time()

        try:
            from airllm import AutoModel
            self.model = AutoModel.from_pretrained(self.model_id)
            self.load_time_s = round(time.time() - start, 2)
            self.loaded = True
            logger.info(f"Air-LLM model loaded in {self.load_time_s}s")
        except Exception as e:
            logger.error(f"Failed to load Air-LLM model: {e}")
            raise

        # Load tokenizer separately
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            logger.info("Tokenizer loaded")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> dict:
        """Run text generation inference."""
        self.load()

        start = time.time()
        self.request_count += 1

        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

            kwargs = {"max_new_tokens": max_tokens}
            if temperature != 1.0:
                kwargs["temperature"] = temperature
                kwargs["do_sample"] = True

            output = self.model.generate(input_ids, **kwargs)
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Strip the prompt echo if present
            if text.startswith(prompt):
                text = text[len(prompt):]

            latency_ms = (time.time() - start) * 1000
            self.total_latency_ms += latency_ms

            return {
                "text": text.strip(),
                "tokens_generated": len(self.tokenizer.encode(text)),
                "latency_ms": round(latency_ms, 1),
            }

        except Exception as e:
            self.error_count += 1
            latency_ms = (time.time() - start) * 1000
            self.total_latency_ms += latency_ms
            logger.error(f"Generation error: {e}")
            raise

    def get_info(self) -> dict:
        """Return model metadata and stats."""
        return {
            "model_id": self.model_id,
            "loaded": self.loaded,
            "airllm_available": _airllm_available,
            "load_time_s": self.load_time_s,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "gpu_device": os.environ.get("CUDA_VISIBLE_DEVICES", "auto"),
        }


# ─── FastAPI Service (PM2 managed) ─────────────────────────────────

def create_app():
    """Create FastAPI app for the Air-LLM model server."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(
        title="Metadron Air-LLM Model Server",
        description="Standalone Air-LLM inference for 70B+ models on limited VRAM",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    manager = AirLLMModelManager()

    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 512
        temperature: float = 0.7

    @app.get("/health")
    async def health():
        return {
            "status": "healthy" if _airllm_available else "degraded",
            "service": "airllm-model-server",
            "model_loaded": manager.loaded,
            "model_id": manager.model_id,
            "airllm_available": _airllm_available,
        }

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        if not _airllm_available:
            raise HTTPException(
                status_code=503,
                detail="Air-LLM library not available. Install from intelligence_platform/Air-LLM/",
            )

        try:
            import asyncio
            result = await asyncio.to_thread(
                manager.generate,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/model-info")
    async def model_info():
        return manager.get_info()

    return app


# ─── Entry Point ───────────────────────────────────────────────────

def main():
    """Run the Air-LLM model server as a PM2-managed service."""
    import uvicorn

    port = int(os.environ.get("AIRLLM_PORT", "8003"))
    host = os.environ.get("AIRLLM_HOST", "0.0.0.0")

    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info(f"Starting Air-LLM Model Server on {host}:{port}")
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
