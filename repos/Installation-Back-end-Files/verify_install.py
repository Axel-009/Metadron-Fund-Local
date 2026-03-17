#!/usr/bin/env python3
"""Verify all backend installations for Metadron Capital.

Run: python3 verify_install.py
"""

import sys

BACKENDS = [
    ("numpy", "NumPy (core)"),
    ("pandas", "Pandas (core)"),
    ("scipy", "SciPy (core)"),
    ("sklearn", "scikit-learn (core)"),
    ("yfinance", "yfinance (data fallback)"),
    ("torch", "PyTorch (BERT/LLM)"),
    ("transformers", "Transformers (FinBERT)"),
    ("qlib", "QLIB (factor mining)"),
]

# These may fail on first import due to network/Julia requirements
OPTIONAL_BACKENDS = [
    ("openbb", "OpenBB SDK (data universe)"),
    ("camel", "CAMEL-AI (MiroFish agents)"),
    ("pysr", "PySR (AI-Newton symbolic regression) — needs Julia runtime"),
    ("airllm", "Air-LLM (model compression)"),
    ("lightgbm", "LightGBM (QLIB models)"),
    ("xgboost", "XGBoost (QLIB models)"),
]


def check():
    print("=" * 60)
    print("Metadron Capital — Backend Installation Verification")
    print("=" * 60)

    ok, fail = 0, 0

    print("\nRequired backends:")
    for pkg, name in BACKENDS:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "OK")
            print(f"  ✅ {name:45s} {ver}")
            ok += 1
        except Exception as e:
            print(f"  ❌ {name:45s} {str(e)[:40]}")
            fail += 1

    print("\nOptional backends:")
    opt_ok = 0
    for pkg, name in OPTIONAL_BACKENDS:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "OK")
            print(f"  ✅ {name:45s} {ver}")
            opt_ok += 1
        except Exception as e:
            print(f"  ⚠️  {name:45s} {str(e)[:40]}")

    print(f"\n{'=' * 60}")
    print(f"Required: {ok}/{ok+fail} installed")
    print(f"Optional: {opt_ok}/{len(OPTIONAL_BACKENDS)} installed")
    print(f"{'=' * 60}")

    # Check bridges
    print("\nBridge connectivity:")
    bridges = [
        ("backends.openbb.openbb_backend", "OpenBBBackend"),
        ("backends.camel_oasis.mirofish_backend", "MiroFishDualSimulation"),
        ("backends.pysr_newton.newton_backend", "AINewtonEngine"),
        ("backends.qlib.qlib_backend", "QLIBBackend"),
        ("backends.transformers_bert.event_bert_backend", "EventBERTBackend"),
        ("backends.airllm.airllm_backend", "AirLLMBackend"),
    ]

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    for module, cls in bridges:
        try:
            mod = __import__(module, fromlist=[cls])
            obj = getattr(mod, cls)
            print(f"  ✅ {cls:40s} importable")
        except Exception as e:
            print(f"  ⚠️  {cls:40s} {str(e)[:40]}")

    return fail == 0


if __name__ == "__main__":
    success = check()
    sys.exit(0 if success else 1)
