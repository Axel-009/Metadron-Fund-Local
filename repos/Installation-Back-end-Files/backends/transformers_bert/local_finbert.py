"""Local FinBERT-equivalent Sentiment Analyzer — Rule-based Financial Sentiment.

A curated dictionary-based financial sentiment classifier that produces
the same output interface as ProsusAI/finbert:
    {"label": "positive"/"negative"/"neutral", "score": 0.0-1.0}

Designed to work 100% offline with zero network access.  When the real
FinBERT model cannot be downloaded (proxy block, air-gapped environment,
etc.) this module provides a high-quality fallback that covers the vast
majority of financial news sentiment patterns.

Features:
    - ~200 curated financial sentiment words (bullish / bearish)
    - Negation handling  ("not bullish" -> bearish)
    - Intensifier handling  ("very bullish", "strongly bearish")
    - Diminisher handling  ("slightly bearish", "somewhat positive")
    - Bigram / phrase matching  ("price target raised", "missed estimates")
    - Returns identical interface to HuggingFace pipeline output

Usage:
    from backends.transformers_bert.local_finbert import LocalFinBERT

    model = LocalFinBERT()
    result = model("AAPL reports record earnings beating estimates by 15%")
    # {"label": "positive", "score": 0.82}

    # Batch mode (same as HuggingFace pipeline)
    results = model(["Good earnings", "Stock crashed"])
    # [{"label": "positive", ...}, {"label": "negative", ...}]
"""

from __future__ import annotations

import logging
import math
import re
from typing import Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentiment dictionaries
# ---------------------------------------------------------------------------

# Bullish / positive words  (weight > 0)
POSITIVE_WORDS: dict[str, float] = {
    # Earnings & results
    "beat": 0.7, "beats": 0.7, "beating": 0.7, "topped": 0.65,
    "surpass": 0.7, "surpassed": 0.7, "surpasses": 0.7,
    "exceed": 0.7, "exceeded": 0.7, "exceeds": 0.7, "exceeding": 0.7,
    "outperform": 0.7, "outperformed": 0.7, "outperforms": 0.7,
    "record": 0.5, "blowout": 0.8, "stellar": 0.75,
    "blockbuster": 0.75, "smashed": 0.7,
    # Growth & momentum
    "growth": 0.5, "growing": 0.45, "grew": 0.5,
    "accelerate": 0.6, "accelerating": 0.6, "acceleration": 0.6,
    "momentum": 0.5, "expand": 0.5, "expanding": 0.5, "expansion": 0.5,
    "surge": 0.7, "surged": 0.7, "surging": 0.7,
    "soar": 0.7, "soared": 0.7, "soaring": 0.7,
    "spike": 0.55, "spiked": 0.55, "spiking": 0.55,
    "rally": 0.6, "rallied": 0.6, "rallying": 0.6,
    "boom": 0.65, "booming": 0.65,
    "jump": 0.5, "jumped": 0.5, "jumps": 0.5,
    "climb": 0.45, "climbed": 0.45, "climbing": 0.45,
    "gain": 0.45, "gained": 0.45, "gains": 0.45,
    "rise": 0.45, "risen": 0.45, "rising": 0.45, "rose": 0.45,
    "increase": 0.4, "increased": 0.4, "increases": 0.4, "increasing": 0.4,
    "higher": 0.35, "high": 0.3,
    # Analyst & ratings
    "upgrade": 0.7, "upgraded": 0.7, "upgrades": 0.7,
    "overweight": 0.55, "outperform": 0.6,
    "buy": 0.5, "bullish": 0.65, "optimistic": 0.55,
    "positive": 0.5, "favorable": 0.5, "favourable": 0.5,
    "recommend": 0.4, "recommended": 0.4,
    # Guidance & outlook
    "raised": 0.6, "raises": 0.6, "raise": 0.55,
    "upbeat": 0.55, "confident": 0.5, "confidence": 0.5,
    "robust": 0.55, "resilient": 0.5, "resilience": 0.5,
    # Corporate actions (positive)
    "approval": 0.6, "approved": 0.6, "approves": 0.6,
    "breakthrough": 0.7, "innovation": 0.45, "innovative": 0.45,
    "buyback": 0.4, "repurchase": 0.4,
    "dividend": 0.35, "special dividend": 0.5,
    "acquisition": 0.4, "merger": 0.35,
    "partnership": 0.4, "collaboration": 0.35,
    "launch": 0.4, "launched": 0.4, "launching": 0.4,
    "contract": 0.35, "awarded": 0.45,
    # Strength indicators
    "strong": 0.5, "stronger": 0.55, "strongest": 0.6,
    "solid": 0.45, "healthy": 0.4, "stable": 0.3,
    "improve": 0.45, "improved": 0.45, "improving": 0.45, "improvement": 0.45,
    "recover": 0.5, "recovered": 0.5, "recovery": 0.5, "recovering": 0.5,
    "rebound": 0.5, "rebounded": 0.5, "rebounding": 0.5,
    "profit": 0.4, "profitable": 0.45, "profitability": 0.45,
    "margin expansion": 0.55, "upside": 0.55,
    "tailwind": 0.45, "tailwinds": 0.45,
    "breakout": 0.5, "all-time high": 0.6,
    "new high": 0.5, "52-week high": 0.5,
    "outpace": 0.5, "outpaced": 0.5, "outpacing": 0.5,
    "win": 0.45, "wins": 0.45, "winning": 0.45, "won": 0.45,
    "success": 0.5, "successful": 0.5,
}

# Bearish / negative words  (weight > 0, applied as negative sentiment)
NEGATIVE_WORDS: dict[str, float] = {
    # Earnings & results
    "miss": 0.7, "missed": 0.7, "misses": 0.7, "missing": 0.6,
    "disappoint": 0.7, "disappointed": 0.7, "disappointing": 0.7, "disappointment": 0.7,
    "shortfall": 0.65, "fell short": 0.65,
    "underperform": 0.7, "underperformed": 0.7, "underperforms": 0.7,
    "weak": 0.55, "weaker": 0.6, "weakest": 0.65, "weakness": 0.55,
    "soft": 0.4, "softer": 0.45, "softness": 0.45, "softening": 0.45,
    # Decline & loss
    "decline": 0.55, "declined": 0.55, "declining": 0.55, "declines": 0.55,
    "drop": 0.5, "dropped": 0.5, "dropping": 0.5, "drops": 0.5,
    "fall": 0.5, "fallen": 0.5, "falling": 0.5, "falls": 0.5,
    "plunge": 0.75, "plunged": 0.75, "plunging": 0.75,
    "crash": 0.85, "crashed": 0.85, "crashing": 0.85,
    "tumble": 0.65, "tumbled": 0.65, "tumbling": 0.65,
    "collapse": 0.8, "collapsed": 0.8, "collapsing": 0.8,
    "sink": 0.55, "sank": 0.55, "sinking": 0.55,
    "slide": 0.5, "slid": 0.5, "sliding": 0.5,
    "slump": 0.6, "slumped": 0.6, "slumping": 0.6,
    "selloff": 0.65, "sell-off": 0.65,
    "rout": 0.7, "bloodbath": 0.8, "carnage": 0.8,
    "loss": 0.55, "losses": 0.55, "lost": 0.5, "losing": 0.5,
    "decrease": 0.4, "decreased": 0.4, "decreasing": 0.4,
    "lower": 0.35, "low": 0.3,
    "down": 0.35, "downside": 0.5,
    "negative": 0.45, "worse": 0.5, "worst": 0.6,
    "reduce": 0.4, "reduced": 0.4, "reducing": 0.4, "reduction": 0.4,
    "shrink": 0.5, "shrinking": 0.5, "shrank": 0.5,
    "contraction": 0.5, "contract": 0.45, "contracting": 0.45,
    "erode": 0.5, "eroded": 0.5, "erosion": 0.5,
    # Analyst & ratings
    "downgrade": 0.7, "downgraded": 0.7, "downgrades": 0.7,
    "underweight": 0.55, "sell": 0.5,
    "bearish": 0.65, "pessimistic": 0.55,
    # Guidance & outlook
    "lowered": 0.6, "lowers": 0.6, "cut": 0.55, "cuts": 0.55,
    "warning": 0.65, "warns": 0.6, "warned": 0.6, "caution": 0.45,
    "cautious": 0.4, "concern": 0.45, "concerned": 0.45, "concerns": 0.45,
    "uncertainty": 0.45, "uncertain": 0.45,
    "risk": 0.35, "risks": 0.35, "risky": 0.4,
    "volatile": 0.4, "volatility": 0.35,
    "headwind": 0.5, "headwinds": 0.5,
    "pressure": 0.4, "pressured": 0.45, "pressures": 0.4,
    # Corporate distress
    "bankruptcy": 0.9, "bankrupt": 0.9, "insolvent": 0.85, "insolvency": 0.85,
    "default": 0.8, "defaulted": 0.8,
    "layoff": 0.6, "layoffs": 0.6, "restructuring": 0.5,
    "impairment": 0.6, "writedown": 0.6, "write-down": 0.6,
    "writeoff": 0.65, "write-off": 0.65,
    "recall": 0.55, "recalled": 0.55,
    "lawsuit": 0.5, "litigation": 0.5, "sued": 0.5, "fine": 0.5, "fined": 0.55,
    "penalty": 0.5, "penalties": 0.5,
    "investigation": 0.55, "probe": 0.5, "subpoena": 0.6,
    "fraud": 0.8, "scandal": 0.75, "misconduct": 0.7,
    "violation": 0.55, "violations": 0.55,
    "rejection": 0.65, "rejected": 0.65, "reject": 0.6,
    "delay": 0.4, "delayed": 0.4, "delays": 0.4,
    "suspend": 0.6, "suspended": 0.6, "suspension": 0.6,
    "halt": 0.55, "halted": 0.55,
    "delist": 0.75, "delisted": 0.75, "delisting": 0.75,
    # Market stress
    "recession": 0.7, "slowdown": 0.5, "stagnation": 0.5,
    "inflation": 0.4, "deflation": 0.45,
    "crisis": 0.75, "contagion": 0.65,
    "bubble": 0.55, "overvalued": 0.5,
    "52-week low": 0.5, "new low": 0.5,
    "margin compression": 0.55,
    "debt": 0.35, "leverage": 0.3, "overleveraged": 0.6,
}

# Multi-word phrases scored as a unit (checked before single words)
POSITIVE_PHRASES: dict[str, float] = {
    "beat estimates": 0.75, "beats estimates": 0.75, "beating estimates": 0.75,
    "beat expectations": 0.75, "beats expectations": 0.75,
    "better than expected": 0.7, "above expectations": 0.65,
    "top line growth": 0.5, "bottom line growth": 0.55,
    "price target raised": 0.65, "raised price target": 0.65,
    "raised guidance": 0.65, "raises guidance": 0.65,
    "upward revision": 0.6, "revised higher": 0.6,
    "strong demand": 0.6, "strong growth": 0.6, "strong results": 0.6,
    "record revenue": 0.65, "record earnings": 0.65, "record profit": 0.65,
    "all time high": 0.6, "all-time high": 0.6,
    "market share gains": 0.55, "gaining market share": 0.55,
    "margin expansion": 0.55, "margins improved": 0.55,
    "positive surprise": 0.65, "upside surprise": 0.7,
    "ahead of schedule": 0.5, "above consensus": 0.6,
    "double upgrade": 0.8, "initiated with buy": 0.6,
    "share buyback": 0.4, "share repurchase program": 0.45,
    "special dividend": 0.5, "dividend increase": 0.5,
    "fda approval": 0.7, "fda approved": 0.7,
    "phase 3 success": 0.7, "clinical trial success": 0.7,
    "strategic acquisition": 0.45, "accretive acquisition": 0.55,
    "cost savings": 0.45, "efficiency gains": 0.45,
}

NEGATIVE_PHRASES: dict[str, float] = {
    "missed estimates": 0.75, "misses estimates": 0.75, "missing estimates": 0.75,
    "missed expectations": 0.75, "misses expectations": 0.75,
    "worse than expected": 0.7, "below expectations": 0.65,
    "fell short of": 0.65, "falls short of": 0.65,
    "price target cut": 0.65, "price target lowered": 0.65,
    "cut price target": 0.65, "lowered price target": 0.65,
    "lowered guidance": 0.65, "lowers guidance": 0.65, "cut guidance": 0.7,
    "downward revision": 0.6, "revised lower": 0.6, "revised down": 0.6,
    "weak demand": 0.6, "slowing growth": 0.55, "weak results": 0.6,
    "revenue miss": 0.65, "earnings miss": 0.65, "profit warning": 0.7,
    "going concern": 0.8, "material weakness": 0.7,
    "negative surprise": 0.65, "downside surprise": 0.7,
    "behind schedule": 0.5, "below consensus": 0.6,
    "double downgrade": 0.8, "initiated with sell": 0.6,
    "dividend cut": 0.6, "dividend suspended": 0.7,
    "fda rejection": 0.75, "fda rejected": 0.75,
    "clinical trial failure": 0.75, "failed trial": 0.75,
    "chapter 11": 0.85, "chapter 7": 0.9,
    "margin pressure": 0.55, "margin compression": 0.55,
    "supply chain disruption": 0.5, "supply chain issues": 0.45,
    "accounting irregularities": 0.75, "restated earnings": 0.7,
    "sec investigation": 0.65, "regulatory scrutiny": 0.5,
    "debt downgrade": 0.7, "credit downgrade": 0.7,
    "mass layoffs": 0.65, "job cuts": 0.55,
    "short seller report": 0.6, "short interest rising": 0.5,
}

# ---------------------------------------------------------------------------
# Negation, intensifier, and diminisher words
# ---------------------------------------------------------------------------

NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nor", "hardly", "barely", "scarcely",
    "without", "lack", "lacks", "lacking", "failed", "fail", "fails",
    "unable", "cannot", "can't", "couldn't", "didn't", "doesn't",
    "don't", "won't", "wouldn't", "shouldn't", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't",
})

# Window (in tokens) over which a negation applies
_NEGATION_WINDOW = 3

INTENSIFIERS: dict[str, float] = {
    "very": 1.3, "extremely": 1.5, "highly": 1.3, "significantly": 1.3,
    "substantially": 1.3, "considerably": 1.25, "remarkably": 1.35,
    "sharply": 1.35, "dramatically": 1.4, "massively": 1.45,
    "strongly": 1.3, "hugely": 1.4, "overwhelmingly": 1.45,
    "exceptionally": 1.4, "aggressively": 1.3, "profoundly": 1.35,
    "deeply": 1.3, "steeply": 1.3,
}

DIMINISHERS: dict[str, float] = {
    "slightly": 0.6, "somewhat": 0.65, "marginally": 0.6,
    "modestly": 0.65, "mildly": 0.6, "partially": 0.65,
    "a bit": 0.6, "a little": 0.6, "relatively": 0.7,
    "fairly": 0.75, "moderately": 0.7,
}


# ---------------------------------------------------------------------------
# Tokenizer (simple, whitespace + punctuation aware)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z0-9\-']+|[.,!?;:]")


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer that preserves hyphenated words."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Core sentiment scorer
# ---------------------------------------------------------------------------

def _score_text(text: str) -> tuple[float, float, float]:
    """Score a single text string.

    Returns:
        (positive_score, negative_score, neutral_residual)
        All in [0, 1], summing to ~1.0.
    """
    text_lower = text.lower()
    tokens = _tokenize(text)
    n_tokens = len(tokens)

    if n_tokens == 0:
        return 0.0, 0.0, 1.0

    pos_total = 0.0
    neg_total = 0.0

    # ---------- Phase 1: phrase matching ----------
    matched_spans: list[tuple[int, int]] = []  # (start_char, end_char)

    for phrase, weight in POSITIVE_PHRASES.items():
        idx = text_lower.find(phrase)
        while idx != -1:
            # Check for negation in the 4 words before the phrase
            prefix = text_lower[max(0, idx - 40):idx]
            prefix_tokens = _tokenize(prefix)
            if any(t in NEGATION_WORDS for t in prefix_tokens[-_NEGATION_WINDOW:]):
                neg_total += weight * 0.8  # negated positive -> negative
            else:
                # Check for intensifier/diminisher in prefix
                mult = 1.0
                for pt in prefix_tokens[-2:]:
                    if pt in INTENSIFIERS:
                        mult = INTENSIFIERS[pt]
                    elif pt in DIMINISHERS:
                        mult = DIMINISHERS[pt]
                pos_total += weight * mult
            matched_spans.append((idx, idx + len(phrase)))
            idx = text_lower.find(phrase, idx + len(phrase))

    for phrase, weight in NEGATIVE_PHRASES.items():
        idx = text_lower.find(phrase)
        while idx != -1:
            prefix = text_lower[max(0, idx - 40):idx]
            prefix_tokens = _tokenize(prefix)
            if any(t in NEGATION_WORDS for t in prefix_tokens[-_NEGATION_WINDOW:]):
                pos_total += weight * 0.8  # negated negative -> positive
            else:
                mult = 1.0
                for pt in prefix_tokens[-2:]:
                    if pt in INTENSIFIERS:
                        mult = INTENSIFIERS[pt]
                    elif pt in DIMINISHERS:
                        mult = DIMINISHERS[pt]
                neg_total += weight * mult
            matched_spans.append((idx, idx + len(phrase)))
            idx = text_lower.find(phrase, idx + len(phrase))

    # ---------- Phase 2: single-word matching ----------
    # Build a set of character positions already covered by phrases
    covered = set()
    for start, end in matched_spans:
        covered.update(range(start, end))

    # Track negation state
    negation_countdown = 0

    # Track intensifier/diminisher for next sentiment word
    pending_modifier = 1.0

    for i, token in enumerate(tokens):
        # Approximate character position
        # (Not perfect, but good enough for overlap check)
        char_pos = text_lower.find(token, sum(len(tokens[j]) + 1 for j in range(i)) if i > 0 else 0)
        if char_pos >= 0 and char_pos in covered:
            continue  # already scored as part of a phrase

        # Check negation
        if token in NEGATION_WORDS:
            negation_countdown = _NEGATION_WINDOW
            continue

        # Check intensifier / diminisher
        if token in INTENSIFIERS:
            pending_modifier = INTENSIFIERS[token]
            continue
        if token in DIMINISHERS:
            pending_modifier = DIMINISHERS[token]
            continue

        is_negated = negation_countdown > 0

        if token in POSITIVE_WORDS:
            weight = POSITIVE_WORDS[token] * pending_modifier
            if is_negated:
                neg_total += weight * 0.8
            else:
                pos_total += weight
            pending_modifier = 1.0
        elif token in NEGATIVE_WORDS:
            weight = NEGATIVE_WORDS[token] * pending_modifier
            if is_negated:
                pos_total += weight * 0.8
            else:
                neg_total += weight
            pending_modifier = 1.0
        else:
            # Non-sentiment word: reset modifier (don't carry across)
            pending_modifier = 1.0

        if negation_countdown > 0:
            negation_countdown -= 1

    # ---------- Phase 3: normalize to [0, 1] probabilities ----------
    raw_total = pos_total + neg_total
    if raw_total == 0:
        return 0.0, 0.0, 1.0

    # Sigmoid-like normalization: more words = higher confidence, capped.
    # Tuned so that 1 strong word (~0.7) yields ~0.55 dominant score,
    # 2 strong words (~1.4) yields ~0.75, and 3+ saturates near 0.90.
    scale = 1.0 - math.exp(-raw_total / 0.9)  # asymptotes to 1.0

    if pos_total > neg_total:
        pos_score = scale * (pos_total / raw_total)
        neg_score = scale * (neg_total / raw_total)
    elif neg_total > pos_total:
        neg_score = scale * (neg_total / raw_total)
        pos_score = scale * (pos_total / raw_total)
    else:
        # Equal positive and negative — lean neutral
        pos_score = scale * 0.35
        neg_score = scale * 0.35

    neutral_score = max(0.0, 1.0 - pos_score - neg_score)
    return pos_score, neg_score, neutral_score


# ---------------------------------------------------------------------------
# Public API — matches HuggingFace pipeline interface
# ---------------------------------------------------------------------------

class LocalFinBERT:
    """Local rule-based FinBERT-equivalent sentiment analyzer.

    Returns the same interface as ``transformers.pipeline("sentiment-analysis")``:
        Single text:  [{"label": "positive", "score": 0.82}]
        Batch:        [[{"label": ...}, ...], ...]

    The returned list always contains one dict per input text (matching
    the top-1 label convention of the HF pipeline).
    """

    def __init__(self):
        logger.info("LocalFinBERT initialized (dictionary-based, no network required)")

    def __call__(
        self,
        texts: Union[str, list[str]],
        **kwargs,
    ) -> list[dict[str, Union[str, float]]]:
        """Analyze sentiment.

        Args:
            texts: A single string or list of strings.
            **kwargs: Accepted for HuggingFace pipeline compatibility
                      (truncation, max_length, etc.) — silently ignored.

        Returns:
            List of dicts, one per input text:
                [{"label": "positive"/"negative"/"neutral", "score": float}]
        """
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for text in texts:
            pos, neg, neu = _score_text(text)

            # Pick the dominant label
            if pos >= neg and pos >= neu:
                label = "positive"
                score = pos
            elif neg >= pos and neg >= neu:
                label = "negative"
                score = neg
            else:
                label = "neutral"
                score = neu

            # Ensure minimum score floor (HF models rarely return < 0.33)
            score = max(score, 0.34)
            # Cap at 0.99 (rule-based shouldn't claim perfect confidence)
            score = min(score, 0.99)

            results.append({"label": label, "score": round(score, 4)})

        return results

    def predict(self, text: str) -> dict[str, Union[str, float]]:
        """Convenience: analyze a single text and return one result dict."""
        return self(text)[0]

    @staticmethod
    def is_available() -> bool:
        """Always available — no external dependencies."""
        return True
