#!/usr/bin/env bash
# Metadron Capital — One-command platform setup
# Usage: ./setup.sh [--skip-deps] [--skip-vault]
set -euo pipefail

PLATFORM_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PLATFORM_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

step() { echo -e "\n${CYAN}[$1/5]${NC} $2"; }
ok()   { echo -e "  ${GREEN}OK${NC} $1"; }
warn() { echo -e "  ${YELLOW}WARNING${NC} $1"; }
fail() { echo -e "  ${RED}FAIL${NC} $1"; exit 1; }

SKIP_DEPS=false
SKIP_VAULT=false
for arg in "$@"; do
  case "$arg" in
    --skip-deps)  SKIP_DEPS=true ;;
    --skip-vault) SKIP_VAULT=true ;;
  esac
done

echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  Metadron Capital — Platform Setup${NC}"
echo -e "${CYAN}================================================${NC}"

# ── Step 1: Decrypt vault ─────────────────────────────────────────────
step 1 "Decrypting API key vault"

if [ "$SKIP_VAULT" = true ]; then
  warn "Skipped (--skip-vault)"
elif [ -f "$PLATFORM_ROOT/.env" ] && grep -q "^ANTHROPIC_API_KEY=sk-" "$PLATFORM_ROOT/.env" 2>/dev/null; then
  ok ".env already has a real ANTHROPIC_API_KEY — skipping decrypt"
elif [ ! -f "$PLATFORM_ROOT/.env.vault.gpg" ]; then
  fail ".env.vault.gpg not found — no vault to decrypt"
else
  echo -n "  Enter vault password: "
  read -rs VAULT_PASS
  echo ""

  if echo "$VAULT_PASS" | gpg --batch --yes --passphrase-fd 0 -d "$PLATFORM_ROOT/.env.vault.gpg" > "$PLATFORM_ROOT/.env" 2>/dev/null; then
    ok "Decrypted .env.vault.gpg → .env"
  else
    fail "Wrong password or corrupt vault"
  fi
fi

# ── Step 2: Propagate .env to sub-modules ─────────────────────────────
step 2 "Propagating keys to sub-modules"

ANTHROPIC_KEY=$(grep "^ANTHROPIC_API_KEY=" "$PLATFORM_ROOT/.env" 2>/dev/null | cut -d= -f2-)
ZEP_KEY=$(grep "^ZEP_API_KEY=" "$PLATFORM_ROOT/.env" 2>/dev/null | cut -d= -f2-)

# MiroFish (has its own .env)
MIROFISH_DIRS=(
  "$PLATFORM_ROOT/intelligence_platform/MiroFish"
  "$PLATFORM_ROOT/repos/layer6_agents/MiroFish"
)
for dir in "${MIROFISH_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    cat > "$dir/.env" <<MFEOF
# --- Unified LLM (Anthropic Claude Opus 4.6) ---
ANTHROPIC_API_KEY=${ANTHROPIC_KEY}

# --- Zep Memory Graph ---
ZEP_API_KEY=${ZEP_KEY}
MFEOF
    ok "$(basename "$(dirname "$dir")")/MiroFish/.env"
  fi
done

# Mirofish monorepo copy (under /mirofish)
if [ -d "$PLATFORM_ROOT/mirofish/backend" ]; then
  cat > "$PLATFORM_ROOT/mirofish/.env" <<MFEOF2
ANTHROPIC_API_KEY=${ANTHROPIC_KEY}
ZEP_API_KEY=${ZEP_KEY}
MFEOF2
  ok "mirofish/.env"
fi

ok "All sub-module .env files synced"

# ── Step 3: Install Python dependencies ───────────────────────────────
step 3 "Installing Python dependencies"

if [ "$SKIP_DEPS" = true ]; then
  warn "Skipped (--skip-deps)"
elif command -v pip &>/dev/null; then
  # Check if already installed
  if python3 -c "import anthropic, langchain_anthropic, openbb" 2>/dev/null; then
    ok "Core dependencies already installed — skipping"
  else
    echo "  Installing (this may take a few minutes)..."
    pip install -e "$PLATFORM_ROOT" --quiet 2>&1 | tail -3
    ok "Dependencies installed"
  fi
else
  warn "pip not found — install manually: pip install -e ."
fi

# ── Step 4: Verify platform health ───────────────────────────────────
step 4 "Verifying platform health"

# Check key is set
if [ -n "$ANTHROPIC_KEY" ] && [ "$ANTHROPIC_KEY" != "your_anthropic_api_key" ]; then
  ok "ANTHROPIC_API_KEY set (${ANTHROPIC_KEY:0:12}...)"
else
  warn "ANTHROPIC_API_KEY not set or still placeholder"
fi

if [ -n "$ZEP_KEY" ] && [ "$ZEP_KEY" != "your_zep_api_key_here" ]; then
  ok "ZEP_API_KEY set (${ZEP_KEY:0:12}...)"
else
  warn "ZEP_API_KEY not set or still placeholder"
fi

# Quick import check
if python3 -c "
import sys
sys.path.insert(0, '$PLATFORM_ROOT')
try:
    from engine.data.universe_engine import UniverseEngine
    print('  OK engine imports')
except Exception as e:
    print(f'  WARN engine import: {e}')
" 2>/dev/null; then
  true
else
  warn "Engine import check failed (may need dependencies)"
fi

# ── Step 5: Summary ──────────────────────────────────────────────────
step 5 "Ready"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Platform setup complete${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "  Run the signal pipeline:  ${CYAN}python3 run_open.py${NC}"
echo -e "  Run tests:                ${CYAN}python3 -m pytest tests/ -v${NC}"
echo -e "  Platform health check:    ${CYAN}python3 bootstrap.py${NC}"
echo -e "  Start MiroFish:           ${CYAN}cd mirofish/backend && python run.py${NC}"
echo ""
echo -e "  Re-encrypt vault after adding new keys:"
echo -e "    ${YELLOW}gpg --symmetric --cipher-algo AES256 -o .env.vault.gpg .env${NC}"
echo ""
