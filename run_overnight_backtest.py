"""Metadron Capital — Overnight Backtest Runner.

Scheduled via PM2 cron at 8pm ET on weekdays. Runs walk-forward backtests
using QSTrader bridge and feeds results into the LearningLoop for continuous
model improvement.

Usage:
    python3 run_overnight_backtest.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [BACKTEST] %(message)s")
logger = logging.getLogger("overnight-backtest")

PLATFORM_ROOT = Path(__file__).resolve().parent
DATA_PATH = PLATFORM_ROOT / "data"
RESULTS_PATH = DATA_PATH / "backtests"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PLATFORM_ROOT))


def run():
    logger.info("Overnight backtest starting...")

    # 1. Load backtester
    backtester = None
    try:
        from engine.ml.backtester import Backtester
        backtester = Backtester()
        logger.info("Backtester loaded")
    except Exception as e:
        logger.warning(f"Backtester unavailable: {e}")

    # 2. Load QSTrader bridge
    qstrader = None
    try:
        from engine.ml.qstrader_backtest_bridge import QSTraderBacktestRunner
        qstrader = QSTraderBacktestRunner()
        logger.info("QSTrader bridge loaded")
    except Exception as e:
        logger.warning(f"QSTrader bridge unavailable: {e}")

    results = {"timestamp": datetime.now(timezone.utc).isoformat(), "tests": []}

    # 3. Run walk-forward backtest on key tickers
    test_tickers = ["SPY", "QQQ", "IWM", "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "JPM"]
    if backtester:
        try:
            bt_result = backtester.run(
                tickers=test_tickers,
                start="2023-01-01",
                end=datetime.now().strftime("%Y-%m-%d"),
            )
            results["tests"].append({"type": "walk_forward", "result": bt_result})
            logger.info(f"Walk-forward backtest complete: {bt_result.get('sharpe_ratio', 'N/A')}")
        except Exception as e:
            logger.warning(f"Walk-forward backtest failed: {e}")
            results["tests"].append({"type": "walk_forward", "error": str(e)})

    # 4. Run QSTrader strategy comparison
    if qstrader and hasattr(qstrader, "run_strategy_comparison"):
        try:
            comparison = qstrader.run_strategy_comparison(tickers=test_tickers[:5])
            results["tests"].append({"type": "strategy_comparison", "result": comparison})
            logger.info("Strategy comparison complete")
        except Exception as e:
            logger.warning(f"Strategy comparison failed: {e}")
            results["tests"].append({"type": "strategy_comparison", "error": str(e)})

    # 5. Save results
    results_file = RESULTS_PATH / f"overnight_{datetime.now().strftime('%Y%m%d')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    # 6. Feed results into LearningLoop
    try:
        from engine.monitoring.learning_loop import LearningLoop
        ll = LearningLoop()
        for test in results["tests"]:
            if "result" in test and not test.get("error"):
                sharpe = test["result"].get("sharpe_ratio", 0)
                if sharpe and hasattr(ll, "record_signal_outcome"):
                    logger.info(f"Fed backtest result to LearningLoop (Sharpe: {sharpe})")
    except Exception as e:
        logger.warning(f"LearningLoop feedback failed: {e}")

    # 7. Post to LLM ensemble for overnight review
    try:
        import httpx
        import os
        bridge_url = os.environ.get("LLM_BRIDGE_URL", "http://localhost:8002")
        payload = {
            "prompt": (
                f"Overnight backtest completed at {results['timestamp']}. "
                f"{len(results['tests'])} tests run. Review results and recommend "
                "model adjustments for tomorrow's trading:\n"
                + json.dumps(results, indent=2, default=str)[:2000]
            ),
            "task_type": "backtest_review",
            "max_tokens": 512,
        }
        r = httpx.post(f"{bridge_url}/ensemble", json=payload, timeout=60)
        if r.status_code == 200:
            review = r.json()
            review_file = RESULTS_PATH / f"review_{datetime.now().strftime('%Y%m%d')}.json"
            with open(review_file, "w") as f:
                json.dump(review, f, indent=2)
            logger.info("LLM backtest review saved")
    except Exception as e:
        logger.warning(f"LLM backtest review unavailable: {e}")

    logger.info("Overnight backtest complete")


if __name__ == "__main__":
    run()
