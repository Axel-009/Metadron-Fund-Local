# Netdata Setup — GEX44 (Rocky Linux 9)

## Install Netdata

```bash
# One-line install (recommended)
curl -fsSL https://get.netdata.cloud/kickstart.sh | bash

# Verify running
systemctl status netdata
```

## Apply Metadron Config

```bash
# Copy config files
sudo cp netdata.conf /etc/netdata/netdata.conf
sudo mkdir -p /etc/netdata/go.d/
sudo cp go.d/nvidia_smi.conf /etc/netdata/go.d/nvidia_smi.conf

# Restart
sudo systemctl restart netdata
```

## Verify GPU Monitoring

```bash
# Check nvidia-smi is accessible
nvidia-smi

# Check Netdata GPU charts
curl -s http://localhost:19999/api/v1/charts | grep nvidia
```

## Verify Prometheus Export

```bash
# This is what Contabo Prometheus scrapes
curl -s "http://localhost:19999/api/v1/allmetrics?format=prometheus" | head -20
```

## Firewall (if UFW active)

```bash
# Allow Netdata from WireGuard subnet only
sudo ufw allow from 10.0.0.0/24 to any port 19999
```

## What Netdata Monitors on GEX44

| Category | Metrics |
|----------|---------|
| CPU | Per-core utilization, frequency, context switches |
| RAM | Used/available/cached, swap, NUMA |
| Disk | I/O latency, throughput, IOPS per device |
| Network | Bandwidth per interface, TCP connections, retransmits |
| GPU (nvidia-smi) | Utilization %, VRAM used/total, temperature, power draw, clock speeds |
| Processes | Per-process CPU/RAM (tracks PM2 services) |

## Integration with Contabo Monitoring

Contabo Prometheus scrapes `http://GEX44_IP:19999/api/v1/allmetrics?format=prometheus`
every 15 seconds via WireGuard VPN. Grafana dashboards on Contabo visualize all metrics.
