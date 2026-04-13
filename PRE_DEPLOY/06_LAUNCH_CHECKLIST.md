# Step 6 — Launch Checklist

## Pre-Launch Verification

Run through this checklist before going live. Every item must pass.

### Server Health

```bash
# On GEX44 as metadron user
cd /opt/metadron
source venv/bin/activate

# 1. GPU is working
nvidia-smi
# Must show: RTX 4000 SFF Ada, 20GB, driver loaded

# 2. All PM2 services running
pm2 status
# Must show: 14+ services "online"
# Model servers may show "waiting restart" on first boot (normal — they lazy-load)

# 3. Engine API healthy
curl -s http://localhost:8001/api/engine/health
# Must return: {"status":"ok","timestamp":"..."}

# 4. Frontend accessible
curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/
# Must return: 200

# 5. LLM Bridge healthy
curl -s http://localhost:8002/health | python3 -m json.tool
# Must return JSON with backend status

# 6. WireGuard tunnel active
sudo wg show wg0
# Must show: peer with recent handshake
ping -c 3 10.0.0.2
# Must respond (Contabo)
```

### Monitoring Health (Contabo)

```bash
# On Contabo
# 7. Docker containers running
docker compose ps
# Must show 7 services "Up"

# 8. Prometheus scraping GEX44
curl -s http://localhost:9090/api/v1/targets | python3 -c "
import json,sys
data=json.load(sys.stdin)
for t in data['data']['activeTargets']:
    print(f\"{t['labels'].get('job','?'):30s} {t['health']:6s} {t['lastScrape'][:19]}\")
"
# All targets should show "up"

# 9. Grafana accessible
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/health
# Must return: 200
```

### API Keys Configured

```bash
# 10. Check via API
curl -s http://localhost:8001/api/engine/vault/status | python3 -c "
import json,sys
data=json.load(sys.stdin)
for slot,info in data.get('slots',{}).items():
    status = 'SET' if info.get('configured') else 'MISSING'
    req = 'REQUIRED' if info.get('required') else 'optional'
    print(f'  {slot:30s} {status:8s} ({req})')
print(f\"\\n  Configured: {data.get('configured',0)}/{data.get('total_slots',0)}\")
"
# ALPACA_API_KEY, ALPACA_SECRET_KEY, FMP_API_KEY must show SET
```

### Security System

```bash
# 11. Security module active
curl -s http://localhost:8001/api/engine/security/status | python3 -c "
import json,sys
data=json.load(sys.stdin)
print(f\"System healthy: {data.get('healthy')}\")
print(f\"Phase chain: {'OK' if not data.get('phase_chain',{}).get('broken') else 'BROKEN'}\")
print(f\"Circuit breaker: {'OK' if not data.get('circuit_breaker',{}).get('locked') else 'LOCKED'}\")
print(f\"Token meter: {data.get('token_meter',{}).get('daily_used',0)} / {data.get('token_meter',{}).get('daily_cap',0)}\")
"
```

### External Access

```bash
# 12. From your local machine (not the server)
curl -I https://metadroncapital.com
# Must return: HTTP/2 200

curl -I https://metadroncapital.com/terminal/
# Must return: HTTP/2 200

curl https://metadroncapital.com/api/engine/health
# Must return: {"status":"ok"}

# 13. Grafana
# Open https://monitor.metadroncapital.com in browser
# Login with admin credentials
# Check dashboards show data
```

## First Boot Sequence

On the first trading day:

1. **8:00 AM ET** — Platform should already be running (PM2 auto-start)
2. **Check PM2**: `pm2 status` — all services online
3. **Check API**: `curl http://localhost:8001/api/engine/health`
4. **Open terminal**: `https://metadroncapital.com/terminal/`
5. **VAULT tab**: Verify all keys show configured
6. **SECURITY tab**: Verify "ALL SYSTEMS SECURE"
7. **LIVE tab**: Should show NAV, P&L, positions
8. **9:30 AM ET** — `market-open` PM2 cron fires automatically
9. **Monitor**: Watch LIVE tab for first signals and trades
10. **WRAP tab**: News feed should show live articles

## If Something Goes Wrong

```bash
# Check PM2 logs for errors
pm2 logs engine-api --lines 50
pm2 logs live-loop --lines 50

# Restart a specific service
pm2 restart engine-api
pm2 restart live-loop

# Restart everything
pm2 restart all

# Check TECH tab in the terminal for engine health
# https://metadroncapital.com/terminal/#/tech

# Emergency: stop all trading (keeps monitoring)
pm2 stop live-loop

# Nuclear: stop everything
pm2 stop all
```

## Overnight Schedule (Automatic)

These run via PM2 cron — no manual action needed:

| Time (ET) | Service | What it does |
|-----------|---------|-------------|
| 16:00 | market-close | EOD reconciliation, P&L snapshot |
| 20:00 | overnight-backtest | Walk-forward backtest → LearningLoop |
| 21:00 | autoresearch | Karpathy model training (5-min budget) |
| 02:00 | graphify-nightly | Regenerate codebase knowledge graph |
| 09:30 | market-open | Full pipeline refresh, signal flush |
| Every hour | hourly-tasks | Sector scan, anomaly detection |

## Summary

If all 13 checks pass, your platform is live:
- Trading engine running 24/7
- AI models serving inference
- Market data flowing from FMP
- Trades executing via Alpaca (paper or live)
- Monitoring dashboards on Grafana
- Alerts configured for Slack/Email
- Security module protecting all endpoints
- Learning loop improving continuously
