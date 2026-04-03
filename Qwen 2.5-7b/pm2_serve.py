"""
Qwen 2.5-7b — PM2-Compatible Model Server Entry Point

This script wraps the Qwen web demo for PM2 process management.
PM2 handles restarts, logging, and monitoring.

Usage (standalone):
    python pm2_serve.py --checkpoint-path Qwen/Qwen2.5-Omni-7B

Usage (via PM2):
    pm2 start ecosystem.config.cjs --only qwen-model-server
"""

import os
import sys
import signal
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("qwen-pm2")


def signal_handler(signum, frame):
    """Handle graceful shutdown signals from PM2."""
    sig_name = signal.Signals(signum).name
    logger.info(f"Received {sig_name}, shutting down gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def health_check():
    """Return health status for PM2 monitoring."""
    return {
        "status": "healthy",
        "model": os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-Omni-7B"),
        "port": int(os.environ.get("QWEN_SERVER_PORT", "7860")),
        "gpu_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Qwen 2.5-7b PM2 Model Server")
    parser.add_argument("--checkpoint-path", type=str,
                        default=os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-Omni-7B"))
    parser.add_argument("--server-port", type=int,
                        default=int(os.environ.get("QWEN_SERVER_PORT", "7860")))
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--flash-attn2", action="store_true")
    args = parser.parse_args()

    logger.info(f"Starting Qwen 2.5-7b model server...")
    logger.info(f"  Model: {args.checkpoint_path}")
    logger.info(f"  Port: {args.server_port}")
    logger.info(f"  GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")

    try:
        sys.argv = [
            "web_demo.py",
            "--checkpoint-path", args.checkpoint_path,
            "--server-port", str(args.server_port),
            "--server-name", args.server_name,
        ]
        if args.cpu_only:
            sys.argv.append("--cpu-only")
        if args.flash_attn2:
            sys.argv.append("--flash-attn2")

        from web_demo import _load_model_processor, _launch_demo
        from argparse import ArgumentParser as AP

        demo_parser = AP()
        demo_parser.add_argument("--checkpoint-path", type=str, default=args.checkpoint_path)
        demo_parser.add_argument("--server-port", type=int, default=args.server_port)
        demo_parser.add_argument("--server-name", type=str, default=args.server_name)
        demo_parser.add_argument("--cpu-only", action="store_true", default=args.cpu_only)
        demo_parser.add_argument("--flash-attn2", action="store_true", default=args.flash_attn2)
        demo_args = demo_parser.parse_args([])

        logger.info("Loading model and processor...")
        model, processor = _load_model_processor(demo_args)
        logger.info("Model loaded. Launching Gradio demo...")
        _launch_demo(demo_args, model, processor)

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install requirements: pip install -r requirements_web_demo.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start model server: {e}")
        sys.exit(1)
