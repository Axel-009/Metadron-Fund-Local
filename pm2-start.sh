#!/usr/bin/env bash
# Metadron Capital — PM2 Platform Launcher
set -euo pipefail
PLATFORM_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PLATFORM_ROOT"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${CYAN}[PM2]${NC} $1"; }
ok()   { echo -e "${GREEN}  OK${NC} $1"; }
mkdir -p "$PLATFORM_ROOT/logs/pm2"
if ! command -v pm2 &>/dev/null; then echo "PM2 not found. Install: npm install -g pm2"; exit 1; fi
ECOSYSTEM="$PLATFORM_ROOT/ecosystem.config.cjs"
print_banner() {
  echo -e "${BOLD}${CYAN}"
  echo "  METADRON CAPITAL — PM2 PROCESS MANAGER"
  echo "  Engine API :8001 | Frontend :5000 | MiroFish :5001/:5174"
  echo "  Qwen :7860 | LLM Bridge :8002 | Air-LLM :8003"
  echo -e "${NC}"
}
start_core() { pm2 start "$ECOSYSTEM" --only "engine-api,express-frontend"; }
start_intelligence() { pm2 start "$ECOSYSTEM" --only "llm-inference-bridge,ainewton-service,learning-loop,metadron-cube,airllm-model-server"; }
start_trading() { pm2 start "$ECOSYSTEM" --only "live-loop,market-open,market-close,hourly-tasks,platform-orchestrator"; }
case "${1:-}" in
  --core) print_banner; start_core ;;
  --intelligence) print_banner; start_core; start_intelligence ;;
  --trading) print_banner; start_core; start_intelligence; start_trading ;;
  --full) print_banner; pm2 start "$ECOSYSTEM" ;;
  --status) pm2 list ;;
  --stop) pm2 stop all ;;
  --restart) pm2 restart all ;;
  --logs) pm2 logs --lines 50 ;;
  --save) pm2 save ;;
  --startup) pm2 startup; pm2 save ;;
  --help|-h) echo "Usage: ./pm2-start.sh [--core|--intelligence|--trading|--full|--status|--stop|--restart|--logs|--save|--startup]" ;;
  *) print_banner; pm2 start "$ECOSYSTEM"; pm2 list ;;
esac
