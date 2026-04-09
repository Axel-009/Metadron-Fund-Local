#!/usr/bin/env bash
# =============================================================================
# deploy-update.sh — Deploy code updates to Metadron Capital (GEX44)
#
# USAGE (run as root or via sudo):
#   sudo bash deploy-update.sh [--branch <branch>] [--skip-build] [--skip-pip]
#
# WHAT THIS DOES:
#   1.  git pull origin <branch>
#   2.  pip install -r requirements.txt  (if requirements.txt changed)
#   3.  npm install                      (if package.json / package-lock changed)
#   4.  npm run build                    (recompile React + Express)
#   5.  Sync marketing/ static files
#   6.  pm2 restart all                  (zero-downtime where possible)
#   7.  Print status report
#
# ROLLBACK:
#   git -C /opt/metadron checkout <previous-commit>
#   pm2 restart all
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_DIR="/opt/metadron"
APP_USER="metadron"
VENV_DIR="$APP_DIR/.venv"
BRANCH="${BRANCH:-main}"
SKIP_BUILD=false
SKIP_PIP=false
LOG_FILE="/var/log/metadron/deploy-$(date +%Y%m%d-%H%M%S).log"

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*" | tee -a "$LOG_FILE"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" | tee -a "$LOG_FILE"; }
error() { echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Parse CLI flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch)      BRANCH="$2";       shift 2 ;;
        --skip-build)  SKIP_BUILD=true;   shift ;;
        --skip-pip)    SKIP_PIP=true;     shift ;;
        *) warn "Unknown argument: $1";   shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Must be root (or run via sudo)
# ---------------------------------------------------------------------------
[[ $EUID -eq 0 ]] || error "Run with sudo: sudo bash deploy-update.sh"

# ---------------------------------------------------------------------------
# Ensure log directory exists
# ---------------------------------------------------------------------------
mkdir -p /var/log/metadron
chown "$APP_USER:$APP_USER" /var/log/metadron

echo "=================================================================" | tee -a "$LOG_FILE"
echo "  Metadron Capital — Deploy Update" | tee -a "$LOG_FILE"
echo "  Date:   $(date -u '+%Y-%m-%d %H:%M UTC')" | tee -a "$LOG_FILE"
echo "  Branch: $BRANCH" | tee -a "$LOG_FILE"
echo "=================================================================" | tee -a "$LOG_FILE"

# ---------------------------------------------------------------------------
# Step 1 — git pull
# ---------------------------------------------------------------------------
info "[1/7] Fetching latest code from origin/$BRANCH..."

cd "$APP_DIR"

# Capture the current commit so we can diff what changed
PREV_COMMIT=$(sudo -u "$APP_USER" git rev-parse HEAD)

sudo -u "$APP_USER" git fetch origin
sudo -u "$APP_USER" git checkout "$BRANCH"
sudo -u "$APP_USER" git pull origin "$BRANCH"

NEW_COMMIT=$(sudo -u "$APP_USER" git rev-parse HEAD)

if [[ "$PREV_COMMIT" == "$NEW_COMMIT" ]]; then
    warn "No new commits on $BRANCH — already up to date."
    warn "Run with FORCE=true to redeploy current commit: FORCE=true bash deploy-update.sh"
    [[ "${FORCE:-false}" == "true" ]] || exit 0
fi

info "Updated: $PREV_COMMIT → $NEW_COMMIT"
info "Changed files:"
sudo -u "$APP_USER" git diff --name-only "$PREV_COMMIT" "$NEW_COMMIT" | tee -a "$LOG_FILE"

# Detect what changed to skip unnecessary steps
REQUIREMENTS_CHANGED=false
PACKAGE_JSON_CHANGED=false

if sudo -u "$APP_USER" git diff --name-only "$PREV_COMMIT" "$NEW_COMMIT" | grep -q "requirements"; then
    REQUIREMENTS_CHANGED=true
fi
if sudo -u "$APP_USER" git diff --name-only "$PREV_COMMIT" "$NEW_COMMIT" | grep -qE "package(-lock)?\.json"; then
    PACKAGE_JSON_CHANGED=true
fi

# ---------------------------------------------------------------------------
# Step 2 — pip install (only if requirements changed, or forced)
# ---------------------------------------------------------------------------
if [[ "$SKIP_PIP" == "true" ]]; then
    info "[2/7] Skipping pip install (--skip-pip flag set)."
elif [[ "$REQUIREMENTS_CHANGED" == "true" ]] || [[ "${FORCE:-false}" == "true" ]]; then
    info "[2/7] requirements.txt changed — running pip install..."
    sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --quiet -r "$APP_DIR/requirements.txt" \
        2>&1 | tee -a "$LOG_FILE"
    info "Python dependencies updated."
else
    info "[2/7] requirements.txt unchanged — skipping pip install."
fi

# ---------------------------------------------------------------------------
# Step 3 — npm install (only if package files changed)
# ---------------------------------------------------------------------------
if [[ "$PACKAGE_JSON_CHANGED" == "true" ]] || [[ "${FORCE:-false}" == "true" ]]; then
    info "[3/7] package.json changed — running npm install..."
    sudo -u "$APP_USER" npm install --prefer-offline --silent \
        2>&1 | tail -10 | tee -a "$LOG_FILE"
    info "Node dependencies updated."
else
    info "[3/7] package.json unchanged — skipping npm install."
fi

# ---------------------------------------------------------------------------
# Step 4 — npm run build
# ---------------------------------------------------------------------------
if [[ "$SKIP_BUILD" == "true" ]]; then
    info "[4/7] Skipping build (--skip-build flag set)."
else
    info "[4/7] Building production assets..."
    sudo -u "$APP_USER" npm run build 2>&1 | tee -a "$LOG_FILE"
    info "Build complete."
fi

# ---------------------------------------------------------------------------
# Step 5 — Sync marketing static files
# ---------------------------------------------------------------------------
info "[5/7] Syncing marketing static files to /opt/metadron/marketing/..."

# If the repo has a dedicated marketing dist directory, sync it here.
# Adjust the source path to match your repo structure.
if [[ -d "$APP_DIR/marketing/dist" ]]; then
    rsync -av --delete "$APP_DIR/marketing/dist/" /opt/metadron/marketing/ \
        2>&1 | tee -a "$LOG_FILE"
elif [[ -d "$APP_DIR/dist/marketing" ]]; then
    rsync -av --delete "$APP_DIR/dist/marketing/" /opt/metadron/marketing/ \
        2>&1 | tee -a "$LOG_FILE"
else
    warn "Marketing dist directory not found — check your build output path."
    warn "Expected: $APP_DIR/marketing/dist or $APP_DIR/dist/marketing"
fi

chown -R "$APP_USER:$APP_USER" /opt/metadron/marketing/
info "Marketing files synced."

# ---------------------------------------------------------------------------
# Step 6 — pm2 restart all (graceful restart)
# ---------------------------------------------------------------------------
info "[6/7] Restarting PM2 processes..."

# Use reload instead of restart where possible (zero-downtime for cluster mode)
sudo -u "$APP_USER" pm2 reload all --update-env 2>&1 | tee -a "$LOG_FILE" || \
    sudo -u "$APP_USER" pm2 restart all --update-env 2>&1 | tee -a "$LOG_FILE"

# Save the updated process list
sudo -u "$APP_USER" pm2 save

info "PM2 processes reloaded."

# ---------------------------------------------------------------------------
# Step 7 — Status report
# ---------------------------------------------------------------------------
info "[7/7] Deployment complete. Status:"
echo
echo "── PM2 Status ────────────────────────────────────────────────"
sudo -u "$APP_USER" pm2 list 2>&1 | tee -a "$LOG_FILE"
echo
echo "── NGINX Status ──────────────────────────────────────────────"
systemctl status nginx --no-pager -l 2>&1 | head -20 | tee -a "$LOG_FILE"
echo
echo "── Commit Info ───────────────────────────────────────────────"
sudo -u "$APP_USER" git -C "$APP_DIR" log -1 --oneline 2>&1 | tee -a "$LOG_FILE"
echo

info "Deploy log saved to: $LOG_FILE"
echo "================================================================="
info "Deployment of $BRANCH complete at $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "================================================================="
