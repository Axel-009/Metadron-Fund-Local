"""Air-LLM Backend — Model Compression & Inference.

Air-LLM enables running large language models (70B+ parameters)
on limited hardware through layer-by-layer inference with disk offloading.

Used in Metadron Capital for:
    - Mavrock HLM (Hidden Language Model) inference
    - Earnings call transcript analysis
    - SEC filing narrative extraction
    - Research report summarization

Dependencies:
    pip install airllm

Usage:
    from backends.airllm.airllm_backend import AirLLMBackend
    backend = AirLLMBackend()
    result = backend.analyze_text("AAPL Q4 earnings call transcript...")
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from airllm import AutoModel as AirAutoModel
    _HAS_AIRLLM = True
    logger.info("Air-LLM loaded successfully")
except (ImportError, ModuleNotFoundError) as e:
    _HAS_AIRLLM = False
    AirAutoModel = None
    if "optimum" in str(e) or "bettertransformer" in str(e).lower():
        logger.warning(
            "Air-LLM import failed due to missing optimum.bettertransformer "
            "(removed in newer optimum versions). "
            "Consider: pip install optimum<1.21 or pip install --upgrade airllm"
        )
    else:
        logger.warning(f"Air-LLM not available — install with: pip install airllm ({e})")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

from backends.transformers_bert.local_finbert import LocalFinBERT

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "airllm"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class AirLLMBackend:
    """Air-LLM model serving backend for large language model inference.

    Supports disk-offloaded inference for models that don't fit in VRAM.
    Falls back to smaller HuggingFace models if Air-LLM unavailable.
    """

    # Models sorted by capability (largest = most capable)
    MODEL_HIERARCHY = {
        "large": "meta-llama/Meta-Llama-3-70B-Instruct",
        "medium": "meta-llama/Meta-Llama-3-8B-Instruct",
        "small": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "sentiment": "ProsusAI/finbert",
    }

    def __init__(self, model_size: str = "small", cache_dir: Optional[str] = None):
        self.model_size = model_size
        self.cache_dir = cache_dir or str(MODELS_DIR)
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._sentiment_model = LocalFinBERT()  # always available

        self._init_model()

    def _init_model(self):
        """Initialize the model backend."""
        model_name = self.MODEL_HIERARCHY.get(self.model_size, self.MODEL_HIERARCHY["small"])

        # Try Air-LLM for large models
        if _HAS_AIRLLM and AirAutoModel is not None and self.model_size in ("large", "medium"):
            try:
                self._model = AirAutoModel.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                )
                logger.info(f"Air-LLM model loaded: {model_name}")
                return
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(
                    f"Air-LLM model load failed due to missing dependency: {e}"
                )
            except Exception as e:
                logger.warning(f"Air-LLM model load failed: {e}")

        # Fallback to transformers pipeline for small models
        if _HAS_TRANSFORMERS:
            try:
                self._pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    device=-1,  # CPU
                    max_new_tokens=256,
                    truncation=True,
                )
                logger.info(f"Transformers pipeline loaded: {model_name}")
                return
            except Exception as e:
                logger.warning(f"Transformers pipeline failed: {e}")

        logger.warning("No LLM backend available — using rule-based analysis")

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text completion."""
        if self._model is not None:
            # Air-LLM inference
            try:
                import torch
                inputs = self._model.tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs, max_new_tokens=max_tokens,
                    )
                return self._model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Air-LLM generation failed: {e}")

        if self._pipeline is not None:
            try:
                result = self._pipeline(prompt, max_new_tokens=max_tokens)
                return result[0]["generated_text"]
            except Exception as e:
                logger.warning(f"Pipeline generation failed: {e}")

        return ""

    def analyze_narrative(self, text: str, context: str = "earnings_call") -> dict:
        """Analyze narrative tone and extract key signals.

        Used by Mavrock HLM for hidden language pattern detection.

        Args:
            text: Document text (earnings call, filing, etc.)
            context: Type of document.

        Returns:
            Dict with tone, key phrases, confidence shifts, etc.
        """
        # Key financial phrases indicating confidence/concern
        confidence_markers = [
            "we are confident", "strong execution", "ahead of plan",
            "record results", "momentum", "accelerating growth",
            "well positioned", "robust pipeline",
        ]
        concern_markers = [
            "challenging", "headwinds", "uncertainty", "cautious",
            "deceleration", "softening", "restructuring", "impairment",
            "risk factors", "we expect pressure", "macro concerns",
        ]
        hedging_markers = [
            "may", "might", "could", "potentially", "subject to",
            "no assurance", "forward-looking", "approximately",
        ]

        text_lower = text.lower()

        confidence_hits = [m for m in confidence_markers if m in text_lower]
        concern_hits = [m for m in concern_markers if m in text_lower]
        hedging_hits = [m for m in hedging_markers if m in text_lower]

        total_markers = len(confidence_hits) + len(concern_hits)
        if total_markers == 0:
            tone = 0.0
        else:
            tone = (len(confidence_hits) - len(concern_hits)) / total_markers

        hedging_ratio = len(hedging_hits) / max(len(text.split()), 1) * 100

        result = {
            "context": context,
            "tone": tone,                           # [-1, +1]
            "confidence_markers": confidence_hits,
            "concern_markers": concern_hits,
            "hedging_density": hedging_ratio,       # % of hedging words
            "confidence_score": len(confidence_hits) / max(total_markers, 1),
            "concern_score": len(concern_hits) / max(total_markers, 1),
            "word_count": len(text.split()),
            "signal": "bullish" if tone > 0.2 else ("bearish" if tone < -0.2 else "neutral"),
        }

        # If LLM available, enhance with deeper analysis
        if self._pipeline is not None or self._model is not None:
            prompt = (
                f"Analyze this {context} excerpt for hidden signals. "
                f"What is management's true confidence level? "
                f"Are there any red flags or unusually positive language?\n\n"
                f"Text: {text[:1000]}\n\n"
                f"Analysis:"
            )
            try:
                llm_analysis = self.generate(prompt, max_tokens=200)
                result["llm_analysis"] = llm_analysis
            except Exception:
                pass

        return result

    def analyze_sentiment(self, text: str) -> dict[str, str | float]:
        """Analyze financial sentiment of a text string.

        Tries the HuggingFace FinBERT pipeline first (if the ``sentiment``
        model was loaded), then falls back to LocalFinBERT (always available).

        Returns:
            {"label": "positive"/"negative"/"neutral", "score": 0.0-1.0}
        """
        # Try HF sentiment pipeline if available
        if self._pipeline is not None and self.model_size == "sentiment":
            try:
                result = self._pipeline(text[:512])[0]
                return {"label": result["label"].lower(), "score": result["score"]}
            except Exception:
                pass

        # LocalFinBERT fallback (always works)
        return self._sentiment_model.predict(text)

    def compare_narratives(self, current: str, previous: str,
                            context: str = "earnings_call") -> dict:
        """Compare two documents to detect narrative shifts.

        Key for Mavrock HLM: detecting when management tone changes
        between quarters, which often precedes guidance revisions.
        """
        current_analysis = self.analyze_narrative(current, context)
        previous_analysis = self.analyze_narrative(previous, context)

        tone_shift = current_analysis["tone"] - previous_analysis["tone"]
        hedging_shift = current_analysis["hedging_density"] - previous_analysis["hedging_density"]
        confidence_shift = current_analysis["confidence_score"] - previous_analysis["confidence_score"]

        return {
            "tone_shift": tone_shift,
            "hedging_shift": hedging_shift,
            "confidence_shift": confidence_shift,
            "current": current_analysis,
            "previous": previous_analysis,
            "signal": (
                "improving" if tone_shift > 0.15 else
                ("deteriorating" if tone_shift < -0.15 else "stable")
            ),
            "direction": int(np.sign(tone_shift)) if abs(tone_shift) > 0.1 else 0,
        }
