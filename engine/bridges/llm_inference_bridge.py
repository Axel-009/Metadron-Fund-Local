"""
Metadron Capital — Unified LLM Inference Bridge

Provides a single interface for all LLM inference across the platform:
  - Qwen 2.5-7b Omni (local GPU, multimodal: text/image/audio/video)
  - Air-LLM (local GPU, efficient 70B+ inference on limited VRAM)
  - Anthropic Claude (cloud API, primary reasoning engine)

The bridge auto-routes requests to the best available backend based on
task type, model availability, and latency requirements.

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

# ─── Backend Registry ────────────────────────────────────────────────

@dataclass
class LLMBackend:
    name: str
    backend_type: str  # "local_qwen", "local_airllm", "cloud_anthropic"
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
    """Unified LLM inference across Qwen, Air-LLM, and Anthropic."""

    def __init__(self):
        self.backends: dict[str, LLMBackend] = {}
        self._qwen_model = None
        self._qwen_processor = None
        self._airllm_model = None
        self._anthropic_client = None
        self._initialize_backends()

    def _initialize_backends(self):
        """Register all available LLM backends."""
        # Qwen 2.5-7b Omni (local multimodal)
        self.backends["qwen"] = LLMBackend(
            name="Qwen 2.5-7b Omni",
            backend_type="local_qwen",
            model_id=os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-Omni-7B"),
            capabilities=["text", "image", "audio", "video", "speech_synthesis"],
        )

        # Air-LLM (local efficient inference)
        self.backends["airllm"] = LLMBackend(
            name="Air-LLM",
            backend_type="local_airllm",
            model_id=os.environ.get("AIRLLM_MODEL_PATH", "meta-llama/Llama-3.1-70B"),
            capabilities=["text", "reasoning", "long_context"],
        )

        # Anthropic Claude (cloud API — primary)
        self.backends["anthropic"] = LLMBackend(
            name="Anthropic Claude",
            backend_type="cloud_anthropic",
            model_id=os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6"),
            capabilities=["text", "reasoning", "code", "analysis", "long_context"],
        )

        self._probe_backends()

    def _probe_backends(self):
        """Check which backends are available."""
        # Anthropic — check for API key
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key and api_key.startswith("sk-"):
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=api_key)
                self.backends["anthropic"].available = True
                logger.info("Anthropic Claude backend: AVAILABLE")
            except ImportError:
                logger.warning("anthropic package not installed")
        else:
            logger.info("Anthropic Claude backend: NO API KEY")

        # Qwen — check for model files / transformers
        try:
            import torch
            if torch.cuda.is_available():
                self.backends["qwen"].available = True
                logger.info(f"Qwen backend: AVAILABLE (GPU: {torch.cuda.get_device_name(0)})")
            else:
                logger.info("Qwen backend: NO GPU (CPU-only mode available)")
                self.backends["qwen"].available = True  # CPU fallback
        except ImportError:
            logger.info("Qwen backend: torch not installed")

        # Air-LLM — check for airllm package
        try:
            sys.path.insert(0, str(PLATFORM_ROOT / "intelligence_platform" / "Air-LLM" / "air_llm"))
            import airllm
            self.backends["airllm"].available = True
            logger.info("Air-LLM backend: AVAILABLE")
        except ImportError:
            logger.info("Air-LLM backend: airllm not installed")

    def _load_qwen(self):
        """Lazy-load Qwen model on first inference request."""
        if self._qwen_model is not None:
            return

        logger.info("Loading Qwen 2.5-7b model...")
        try:
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
            model_path = self.backends["qwen"].model_id

            import torch
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            self._qwen_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path, device_map=device_map, torch_dtype="auto"
            )
            self._qwen_processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
            logger.info("Qwen model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            self.backends["qwen"].available = False

    def _load_airllm(self):
        """Lazy-load Air-LLM model."""
        if self._airllm_model is not None:
            return

        logger.info("Loading Air-LLM model...")
        try:
            sys.path.insert(0, str(PLATFORM_ROOT / "intelligence_platform" / "Air-LLM" / "air_llm"))
            from airllm import AutoModel
            model_path = self.backends["airllm"].model_id
            self._airllm_model = AutoModel.from_pretrained(model_path)
            logger.info("Air-LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Air-LLM model: {e}")
            self.backends["airllm"].available = False

    def select_backend(self, task_type: str, preferred: Optional[str] = None) -> Optional[str]:
        """Select the best backend for a given task type.

        Routing logic:
          - Multimodal (image/audio/video) → Qwen
          - Long reasoning / analysis → Anthropic (primary) > Air-LLM (fallback)
          - Sentiment / NLP → Anthropic > Air-LLM > Qwen
          - Code generation → Anthropic
          - General text → Anthropic > Qwen > Air-LLM
        """
        if preferred and preferred in self.backends and self.backends[preferred].available:
            return preferred

        task_routing = {
            "multimodal": ["qwen", "anthropic"],
            "image": ["qwen", "anthropic"],
            "audio": ["qwen"],
            "video": ["qwen"],
            "speech": ["qwen"],
            "reasoning": ["anthropic", "airllm", "qwen"],
            "analysis": ["anthropic", "airllm"],
            "sentiment": ["anthropic", "airllm", "qwen"],
            "code": ["anthropic"],
            "text": ["anthropic", "qwen", "airllm"],
            "earnings": ["anthropic", "airllm"],
            "sec_filing": ["anthropic", "airllm"],
            "trade_thesis": ["anthropic", "airllm"],
            "narrative": ["anthropic", "airllm", "qwen"],
        }

        candidates = task_routing.get(task_type, ["anthropic", "qwen", "airllm"])
        for candidate in candidates:
            if candidate in self.backends and self.backends[candidate].available:
                return candidate

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
        """Run inference on the best available backend.

        Returns:
            {
                "text": str,           # Generated text
                "backend": str,        # Which backend was used
                "latency_ms": float,   # Response time
                "tokens_used": int,    # Approximate token count
                "task_type": str,      # Task classification
            }
        """
        backend_name = self.select_backend(task_type, preferred_backend)
        if not backend_name:
            return {
                "text": "",
                "backend": "none",
                "error": "No available backend for this task type",
                "latency_ms": 0,
                "tokens_used": 0,
                "task_type": task_type,
            }

        backend = self.backends[backend_name]
        start_time = time.time()

        try:
            if backend_name == "anthropic":
                result = await self._infer_anthropic(prompt, max_tokens, temperature, system_prompt)
            elif backend_name == "qwen":
                result = await self._infer_qwen(prompt, max_tokens, images, audio)
            elif backend_name == "airllm":
                result = await self._infer_airllm(prompt, max_tokens, temperature)
            else:
                result = {"text": "", "tokens_used": 0, "error": f"Unknown backend: {backend_name}"}

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
            }

        except Exception as e:
            backend.request_count += 1
            backend.error_count += 1
            logger.error(f"Inference error on {backend_name}: {e}")

            # Try fallback
            fallback = self.select_backend(task_type)
            if fallback and fallback != backend_name:
                logger.info(f"Falling back from {backend_name} to {fallback}")
                return await self.infer(prompt, task_type, fallback, max_tokens, temperature, system_prompt)

            return {
                "text": "",
                "backend": backend_name,
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000,
                "tokens_used": 0,
                "task_type": task_type,
            }

    async def _infer_anthropic(self, prompt: str, max_tokens: int, temperature: float,
                                system_prompt: Optional[str]) -> dict:
        """Inference via Anthropic Claude API."""
        if not self._anthropic_client:
            raise RuntimeError("Anthropic client not initialized")

        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.backends["anthropic"].model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await asyncio.to_thread(
            self._anthropic_client.messages.create, **kwargs
        )

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        return {
            "text": text,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        }

    async def _infer_qwen(self, prompt: str, max_tokens: int,
                           images: Optional[list], audio: Optional[str]) -> dict:
        """Inference via local Qwen 2.5-7b model."""
        self._load_qwen()
        if not self._qwen_model:
            raise RuntimeError("Qwen model not loaded")

        def _generate():
            inputs = self._qwen_processor(text=prompt, return_tensors="pt")
            inputs = inputs.to(self._qwen_model.device)
            output_ids = self._qwen_model.generate(
                **inputs, max_new_tokens=max_tokens
            )
            generated = self._qwen_processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            return generated[0] if generated else ""

        text = await asyncio.to_thread(_generate)
        return {"text": text, "tokens_used": len(text.split()) * 2}

    async def _infer_airllm(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        """Inference via Air-LLM (efficient large model inference)."""
        self._load_airllm()
        if not self._airllm_model:
            raise RuntimeError("Air-LLM model not loaded")

        def _generate():
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.backends["airllm"].model_id)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            output = self._airllm_model.generate(
                input_ids, max_new_tokens=max_tokens, temperature=temperature
            )
            return tokenizer.decode(output[0], skip_special_tokens=True)

        text = await asyncio.to_thread(_generate)
        return {"text": text, "tokens_used": len(text.split()) * 2}

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
        description="Unified LLM inference across Qwen 2.5-7b, Air-LLM, and Anthropic Claude",
        version="1.0.0",
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
