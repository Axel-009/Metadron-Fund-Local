#!/usr/bin/env python3
"""
Metadron Capital — Session Bootstrap Script
Run this at the start of any new session to verify platform state.

Usage:
    python3 bootstrap.py
    python3 bootstrap.py --full    # Run tests too
"""

import sys
import os
import subprocess
from pathlib import Path

PLATFORM_ROOT = Path(__file__).parent
REPOS_DIR = PLATFORM_ROOT / "repos"

LAYER_MAP = {
    "layer1_data": {
        "Financial-Data": "yfinance market data pipeline",
        "open-bb": "OpenBB investment research terminal",
        "hedgefund-tracker": "SEC 13F institutional flow tracker",
    },
    "layer2_signals": {
        "Mav-Analysis": "MaverickMCP technical analysis engine",
        "quant-trading": "Quantitative strategy library",
        "stock-chain": "Time-series chain analysis",
        "CTA-code": "CTA/trend-following signals",
    },
    "layer3_ml": {
        "QLIB": "Microsoft Qlib quantitative ML framework",
        "Stock-prediction": "DL price prediction models",
        "ML-Macro-Market": "Macro-to-market regime classification",
        "AI-Newton": "Physics-inspired financial models",
    },
    "layer4_portfolio": {
        "ai-hedgefund": "Multi-agent AI hedge fund",
        "financial-distressed-repo": "Company default prediction",
        "sophisticated-distress-analysis": "Advanced distress analytics",
    },
    "layer5_infra": {
        "Kserve": "Kubernetes ML model serving",
        "nividia-repo": "NVIDIA GPU-optimized DL",
        "Air-LLM": "Memory-efficient LLM inference",
    },
    "layer6_agents": {
        "Ruflo-agents": "claude-flow multi-agent orchestration",
    },
}


def check_repo(layer: str, repo: str) -> tuple[bool, int]:
    """Check if a repo exists and count its files."""
    repo_path = REPOS_DIR / layer / repo
    if not repo_path.exists():
        return False, 0
    file_count = sum(1 for _ in repo_path.rglob("*") if _.is_file() and "__pycache__" not in str(_))
    return True, file_count


def check_core() -> bool:
    """Verify core platform modules."""
    try:
        sys.path.insert(0, str(PLATFORM_ROOT))
        from core.platform import MetadronPlatform
        from core.signals import SignalEngine
        from core.portfolio import PortfolioEngine
        return True
    except ImportError as e:
        print(f"  Core import error: {e}")
        return False


def run_tests() -> int:
    """Run platform tests."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(PLATFORM_ROOT),
        capture_output=True, text=True
    )
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    return result.returncode


def main():
    full_mode = "--full" in sys.argv

    print("=" * 70)
    print("METADRON CAPITAL — SESSION BOOTSTRAP")
    print("=" * 70)
    print()

    # Check core
    print("Checking core platform modules...")
    core_ok = check_core()
    print(f"  Core: {'OK' if core_ok else 'FAILED'}")
    print()

    # Check repos
    total_files = 0
    total_repos = 0
    missing_repos = []

    for layer, repos in LAYER_MAP.items():
        layer_name = layer.replace("_", " ").upper()
        print(f"━━━ {layer_name} ━━━")
        for repo, description in repos.items():
            exists, count = check_repo(layer, repo)
            total_files += count
            if exists:
                total_repos += 1
                print(f"  ● {repo:<40} {count:>6} files  [{description}]")
            else:
                missing_repos.append(repo)
                print(f"  ✗ {repo:<40} MISSING")
        print()

    # Summary
    print("=" * 70)
    print(f"REPOS: {total_repos}/18 present | FILES: {total_files:,} | CORE: {'OK' if core_ok else 'FAILED'}")
    if missing_repos:
        print(f"MISSING: {', '.join(missing_repos)}")
    print("=" * 70)

    if full_mode:
        print("\nRunning platform tests...")
        return run_tests()

    if total_repos == 18 and core_ok:
        print("\nPlatform ready. Run with --full to execute tests.")
        return 0
    else:
        print("\nPlatform has issues. Check missing repos or core imports.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
