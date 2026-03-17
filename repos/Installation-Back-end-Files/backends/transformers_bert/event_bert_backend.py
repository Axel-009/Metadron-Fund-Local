"""TradeTheEvent BERT Backend — Event Detection & Classification.

Fine-tuned BERT model for financial event detection from text.

Event types detected:
    - Earnings surprises (beat/miss/inline)
    - M&A announcements
    - FDA approvals/rejections
    - Analyst upgrades/downgrades
    - Share buyback announcements
    - Management changes
    - Guidance revisions
    - Regulatory actions
    - Dividend changes
    - Macro events (rate decisions, employment data)

Dependencies:
    pip install transformers torch

Usage:
    from backends.transformers_bert.event_bert_backend import EventBERTBackend
    backend = EventBERTBackend()
    events = backend.detect_events(["AAPL reports Q4 earnings beating estimates by 15%"])
"""

import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    logger.warning("PyTorch not available")

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, AutoModel,
    )
    _HAS_TRANSFORMERS = True
    logger.info("Transformers library loaded")
except ImportError:
    _HAS_TRANSFORMERS = False
    logger.warning("Transformers not available — install with: pip install transformers")

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints" / "event_bert"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DetectedEvent:
    """A financial event detected from text."""
    text: str
    event_type: str
    ticker: Optional[str] = None
    sentiment: float = 0.0       # [-1, 1]
    confidence: float = 0.0      # [0, 1]
    magnitude: float = 0.0       # expected price impact [0, 1]
    direction: int = 0           # -1, 0, +1
    metadata: dict = field(default_factory=dict)


class EventBERTBackend:
    """BERT-based financial event detection and classification.

    Uses pre-trained FinBERT for sentiment + custom classifier for event types.
    """

    EVENT_TYPES = [
        "earnings_surprise", "ma_announcement", "fda_decision",
        "analyst_rating", "buyback", "management_change",
        "guidance_revision", "regulatory_action", "dividend_change",
        "macro_event", "other",
    ]

    # Keywords for rule-based fallback
    EVENT_KEYWORDS = {
        "earnings_surprise": ["earnings", "revenue", "beat", "miss", "EPS", "quarterly results"],
        "ma_announcement": ["acquire", "merger", "takeover", "buyout", "deal"],
        "fda_decision": ["FDA", "approval", "clinical trial", "phase", "drug"],
        "analyst_rating": ["upgrade", "downgrade", "price target", "outperform", "underperform"],
        "buyback": ["buyback", "repurchase", "share repurchase"],
        "management_change": ["CEO", "CFO", "resign", "appoint", "executive"],
        "guidance_revision": ["guidance", "outlook", "forecast", "raised", "lowered"],
        "regulatory_action": ["SEC", "fine", "investigation", "compliance", "regulation"],
        "dividend_change": ["dividend", "payout", "yield increase", "dividend cut"],
        "macro_event": ["Fed", "interest rate", "inflation", "unemployment", "GDP", "CPI"],
    }

    SENTIMENT_WORDS = {
        "positive": ["beat", "surpass", "upgrade", "approval", "growth", "raised",
                     "outperform", "strong", "record", "exceed", "bullish"],
        "negative": ["miss", "downgrade", "rejection", "decline", "cut", "lowered",
                     "underperform", "weak", "warning", "bearish", "loss"],
    }

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self._sentiment_pipeline = None
        self._tokenizer = None
        self._model = None
        self._initialized = False

        self._init_models()

    def _init_models(self):
        """Initialize BERT models."""
        if not _HAS_TRANSFORMERS or not _HAS_TORCH:
            logger.warning("BERT backend in rule-based fallback mode")
            return

        try:
            # Check for local checkpoint first, avoid network download
            local_model_path = CHECKPOINTS_DIR / "finbert"
            model_source = str(local_model_path) if local_model_path.exists() else self.model_name
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_source,
                tokenizer=model_source,
                device=-1,  # CPU
                truncation=True,
                max_length=512,
            )
            self._initialized = True
            logger.info(f"FinBERT sentiment pipeline loaded: {model_source}")
        except (OSError, ConnectionError, Exception) as e:
            logger.warning(f"FinBERT init failed (using rule-based fallback): {e}")
            self._sentiment_pipeline = None

        # Try to load custom event classifier if checkpoint exists
        checkpoint_path = CHECKPOINTS_DIR / "event_classifier"
        if checkpoint_path.exists():
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    str(checkpoint_path),
                    num_labels=len(self.EVENT_TYPES),
                )
                self._model.eval()
                logger.info("Custom event classifier loaded from checkpoint")
            except Exception as e:
                logger.warning(f"Custom event classifier load failed: {e}")

    def detect_events(self, texts: list[str],
                       tickers: Optional[list[str]] = None) -> list[DetectedEvent]:
        """Detect financial events from text.

        Args:
            texts: List of news headlines / article text.
            tickers: Optional list of associated tickers.

        Returns:
            List of DetectedEvent.
        """
        events = []

        for i, text in enumerate(texts):
            ticker = tickers[i] if tickers and i < len(tickers) else None

            # Event type classification
            event_type = self._classify_event_type(text)

            # Sentiment analysis
            sentiment, sent_confidence = self._analyze_sentiment(text)

            # Magnitude estimation
            magnitude = self._estimate_magnitude(event_type, sentiment, sent_confidence)

            # Direction
            direction = int(np.sign(sentiment)) if abs(sentiment) > 0.1 else 0

            events.append(DetectedEvent(
                text=text,
                event_type=event_type,
                ticker=ticker,
                sentiment=sentiment,
                confidence=sent_confidence,
                magnitude=magnitude,
                direction=direction,
                metadata={
                    "model": self.model_name if self._initialized else "rule_based",
                },
            ))

        return events

    def _classify_event_type(self, text: str) -> str:
        """Classify the event type of a text."""
        # If custom model available, use it
        if self._model is not None and self._tokenizer is not None:
            try:
                inputs = self._tokenizer(text, return_tensors="pt",
                                          truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    predicted = torch.argmax(probs, dim=-1).item()
                    return self.EVENT_TYPES[predicted]
            except Exception:
                pass

        # Rule-based fallback
        text_lower = text.lower()
        best_type = "other"
        best_score = 0

        for event_type, keywords in self.EVENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > best_score:
                best_score = score
                best_type = event_type

        return best_type

    def _analyze_sentiment(self, text: str) -> tuple[float, float]:
        """Analyze sentiment of text. Returns (sentiment, confidence)."""
        # FinBERT pipeline
        if self._sentiment_pipeline:
            try:
                result = self._sentiment_pipeline(text[:512])[0]
                label = result["label"].lower()
                score = result["score"]

                if label == "positive":
                    return score, score
                elif label == "negative":
                    return -score, score
                else:
                    return 0.0, score
            except Exception:
                pass

        # Rule-based fallback
        text_lower = text.lower()
        pos_count = sum(1 for w in self.SENTIMENT_WORDS["positive"] if w in text_lower)
        neg_count = sum(1 for w in self.SENTIMENT_WORDS["negative"] if w in text_lower)
        total = pos_count + neg_count
        if total == 0:
            return 0.0, 0.3

        sentiment = (pos_count - neg_count) / total
        confidence = min(total / 5, 1.0) * 0.7  # cap at 0.7 for rule-based
        return sentiment, confidence

    def _estimate_magnitude(self, event_type: str, sentiment: float,
                             confidence: float) -> float:
        """Estimate expected price impact magnitude."""
        # Base magnitude by event type
        base_magnitude = {
            "earnings_surprise": 0.6,
            "ma_announcement": 0.8,
            "fda_decision": 0.9,
            "analyst_rating": 0.3,
            "buyback": 0.2,
            "management_change": 0.4,
            "guidance_revision": 0.5,
            "regulatory_action": 0.5,
            "dividend_change": 0.2,
            "macro_event": 0.4,
            "other": 0.1,
        }

        base = base_magnitude.get(event_type, 0.1)
        return base * abs(sentiment) * confidence

    def batch_analyze(self, news_data: list[dict]) -> list[DetectedEvent]:
        """Analyze a batch of news items.

        Args:
            news_data: List of dicts with 'text' and optional 'ticker' keys.

        Returns:
            List of DetectedEvent sorted by magnitude.
        """
        texts = [item.get("text", item.get("title", "")) for item in news_data]
        tickers = [item.get("ticker") for item in news_data]

        events = self.detect_events(texts, tickers)
        events.sort(key=lambda e: e.magnitude * e.confidence, reverse=True)
        return events

    def get_trading_signals(self, events: list[DetectedEvent],
                             min_confidence: float = 0.4) -> dict[str, dict]:
        """Convert detected events into trading signals.

        Returns dict of ticker -> signal dict.
        """
        signals = {}

        for event in events:
            if event.confidence < min_confidence or not event.ticker:
                continue

            ticker = event.ticker
            if ticker not in signals or event.magnitude > signals[ticker].get("magnitude", 0):
                signals[ticker] = {
                    "event_type": event.event_type,
                    "sentiment": event.sentiment,
                    "confidence": event.confidence,
                    "magnitude": event.magnitude,
                    "direction": event.direction,
                    "text": event.text[:100],
                }

        return signals
