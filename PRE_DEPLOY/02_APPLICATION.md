# Step 2 — Deploy the Application

## 2.1 Clone the Repository

```bash
# As metadron user
su - metadron

# Clone to /opt/metadron
sudo mkdir -p /opt/metadron
sudo chown metadron:metadron /opt/metadron
cd /opt
git clone https://github.com/Axel-009/Metadron-Fund-Local.git metadron
cd /opt/metadron
```

## 2.2 Python Dependencies

This installs EVERYTHING the platform needs. Takes 10-15 minutes.

```bash
cd /opt/metadron

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support FIRST (large download ~2GB)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install all platform dependencies
pip install -r requirements.txt

# Install additional packages from pyproject.toml that aren't in requirements.txt
pip install \
    joblib pyarrow requests tiktoken \
    statsmodels fredapi \
    plotly matplotlib seaborn \
    pytest pytest-asyncio

# Install OpenBB providers
pip install \
    openbb-fred openbb-sec openbb-cboe openbb-nasdaq \
    openbb-fmp openbb-tiingo openbb-polygon openbb-oecd

# Verify critical imports
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import fastapi; print(f'FastAPI {fastapi.__version__}')"
python3 -c "import sqlalchemy; print(f'SQLAlchemy {sqlalchemy.__version__}')"
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"
python3 -c "import xgboost; print(f'XGBoost {xgboost.__version__}')"
python3 -c "import transformers; print(f'Transformers {transformers.__version__}')"
```

If PyTorch says `CUDA: False`, your NVIDIA driver or CUDA toolkit isn't installed correctly. Go back to Step 1.4.

## 2.3 Node.js Dependencies

```bash
cd /opt/metadron

# Install root dependencies (Express server + React client)
npm install

# Build the frontend for production
npm run build
```

This creates `dist/index.cjs` (the production Express server that serves the React terminal).

## 2.4 Create Required Directories

```bash
mkdir -p /opt/metadron/logs/pm2
mkdir -p /opt/metadron/data/{vault,security/ledger,archive/token_usage,learning/archive,paul_patterns,graphify,backtests,trades,regime,discoveries,universe_cache,agents}
mkdir -p /opt/metadron/logs/{errors,returns,reports,paper_broker,alpaca_broker,research_bots,learning_loop,live_loop,l7_execution,l7_learning,conviction_overrides,enforcement,agent_factory,agent_scorecard,missed_opportunities}
```

## 2.5 Environment Configuration

```bash
cd /opt/metadron

# Copy the production template
cp review/deployment/hetzner/.env.production.example .env.production

# Edit with your actual values
nano .env.production
```

**You MUST set these values (the platform won't work without them):**

```bash
# REQUIRED — Trading
ALPACA_API_KEY=your_actual_alpaca_key
ALPACA_SECRET_KEY=your_actual_alpaca_secret
ALPACA_PAPER_TRADE=True    # Set to False for live trading (be sure!)

# REQUIRED — Market Data
FMP_API_KEY=your_actual_fmp_key

# RECOMMENDED — AI Intelligence
XIAOMI_MIMO_API_KEY=your_xiaomi_key    # Brain Power orchestrator

# SECURITY — Change these!
METADRON_HMAC_KEY=generate_a_random_64_char_string_here
SESSION_SECRET=generate_another_random_string
JWT_SECRET=and_another_one

# TIMEZONE
TZ=America/New_York
```

Generate random secrets:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
# Run 3 times, use each output for HMAC_KEY, SESSION_SECRET, JWT_SECRET
```

## 2.6 Initialize Database

```bash
cd /opt/metadron
source venv/bin/activate

# The database creates itself on first API start, but let's verify
python3 -c "
from engine.db.database import engine, Base
from engine.db import tables
if engine:
    Base.metadata.create_all(bind=engine)
    print('Database initialized: data/metadron.db')
else:
    print('ERROR: Database engine failed to create')
"
```

## 2.7 Run Tests

```bash
cd /opt/metadron
source venv/bin/activate
python3 -m pytest tests/ -v
```

Expected: 164+ passed. The 1 Alpaca test failure and ~20 router test errors
are normal if you haven't started the API server yet.

## 2.8 Start PM2 Ecosystem

```bash
cd /opt/metadron
source venv/bin/activate

# Set production environment
export NODE_ENV=production

# Start all services
pm2 start ecosystem.config.cjs --env production

# Check they're all running
pm2 status

# You should see 14-16 services, most showing "online"
# Model servers may show "waiting restart" until GPU models load (this is normal)

# Save the process list for auto-restart on reboot
pm2 save
```

## 2.9 Install PM2 Exporter

```bash
# Install the Prometheus exporter for PM2
npm install -g @burningtree/pm2-prometheus-exporter

# Or alternatively:
pip install pm2-prometheus-exporter

# Start it (binds to port 9209)
pm2-prometheus-exporter &

# Verify
curl -s http://localhost:9209/metrics | head -5
```

## 2.10 Apply Netdata Config

```bash
# Copy Metadron-specific Netdata config
sudo cp /opt/metadron/review/deployment/hetzner/netdata/netdata.conf /etc/netdata/
sudo mkdir -p /etc/netdata/go.d/
sudo cp /opt/metadron/review/deployment/hetzner/netdata/go.d/nvidia_smi.conf /etc/netdata/go.d/
sudo systemctl restart netdata
```

## 2.11 Install Systemd Service (Auto-Start on Boot)

```bash
# Copy the service file
sudo cp /opt/metadron/review/deployment/hetzner/systemd/metadron-pm2.service /etc/systemd/system/

# Edit paths if your NVM version differs
sudo nano /etc/systemd/system/metadron-pm2.service
# Make sure the PATH line matches your Node version:
# /home/metadron/.nvm/versions/node/v20.x.x/bin

# Enable
sudo systemctl daemon-reload
sudo systemctl enable metadron-pm2

# Test
sudo systemctl start metadron-pm2
sudo systemctl status metadron-pm2
```

## 2.12 Install Claude Code + Jarvis

```bash
# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Install MCP plugins for agent reasoning
claude mcp add sequential-thinking -- npx -y @modelcontextprotocol/server-sequential-thinking
claude mcp add --scope user context7 -- npx -y @upstash/context7-mcp

# Install Jarvis assistant
npm install -g @anthropic-ai/jarvis
```

## 2.13 Generate Knowledge Graph (Optional — First Time)

```bash
cd /opt/metadron
source venv/bin/activate
graphify .
# Takes 1-5 minutes. Creates graphify-out/graph.json
# After this, the GRAPHIFY tab will show data
# PM2 also regenerates this nightly at 2am
```

## 2.14 Verify Everything

```bash
# Check all PM2 services
pm2 status

# Check Engine API is responding
curl -s http://localhost:8001/api/engine/health
# Should return: {"status":"ok","timestamp":"..."}

# Check frontend is responding
curl -s http://localhost:5000/ | head -5
# Should return HTML

# Check model servers
curl -s http://localhost:8002/health  # LLM Bridge
curl -s http://localhost:8004/health  # Qwen
curl -s http://localhost:8005/health  # Llama

# Check logs for errors
pm2 logs --lines 20
```

If Engine API returns health OK, proceed to Step 3.
