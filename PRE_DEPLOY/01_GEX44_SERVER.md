# Step 1 — GEX44 Server Setup (Rocky Linux 9)

## 1.1 Initial Server Access

When Hetzner provisions your GEX44, you'll get:
- An IP address (your public IPv4)
- Root password (or SSH key if you configured one)

SSH in:
```bash
ssh root@YOUR_IP_ADDRESS
```

## 1.2 Create Service Account

Never run the platform as root. Create a dedicated user:
```bash
# Create user
useradd -m -s /bin/bash metadron
passwd metadron
# Give sudo access
usermod -aG wheel metadron
# Switch to that user
su - metadron
```

From now on, everything runs as the `metadron` user unless noted.

## 1.3 System Updates

```bash
sudo dnf update -y
sudo dnf install -y epel-release
sudo dnf install -y \
    git curl wget unzip tar \
    gcc gcc-c++ make cmake \
    openssl-devel libffi-devel bzip2-devel \
    readline-devel sqlite-devel zlib-devel \
    libpq-devel \
    cairo pango gdk-pixbuf2 \
    firewalld
```

## 1.4 NVIDIA Driver + CUDA

The GEX44 has an NVIDIA RTX 4000 SFF Ada (20GB VRAM). You need the driver and CUDA toolkit.

```bash
# Add NVIDIA repo for Rocky Linux 9
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

# Install NVIDIA driver + CUDA toolkit
sudo dnf install -y cuda-toolkit-12-4 nvidia-driver

# Reboot to load the driver
sudo reboot
```

After reboot, verify:
```bash
nvidia-smi
```

You should see:
- Driver Version: 550.x or higher
- CUDA Version: 12.4
- GPU: NVIDIA RTX 4000 SFF Ada, 20GB

If `nvidia-smi` doesn't work, the driver didn't install correctly. Check `dmesg | grep -i nvidia`.

## 1.5 Python 3.11

Rocky Linux 9 ships with Python 3.9. You need 3.11:

```bash
sudo dnf install -y python3.11 python3.11-pip python3.11-devel

# Make it the default python3
sudo alternatives --set python3 /usr/bin/python3.11

# Verify
python3 --version
# Should show: Python 3.11.x

# Upgrade pip
python3 -m pip install --upgrade pip
```

## 1.6 Node.js 20 (via NVM)

```bash
# Install NVM
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

# Reload shell
source ~/.bashrc

# Install Node 20
nvm install 20
nvm use 20
nvm alias default 20

# Verify
node --version
# Should show: v20.x.x

npm --version
# Should show: 10.x.x
```

## 1.7 PM2 (Process Manager)

```bash
npm install -g pm2

# Verify
pm2 --version
```

## 1.8 Firewall

```bash
sudo systemctl enable --now firewalld

# Allow SSH
sudo firewall-cmd --permanent --add-service=ssh

# Allow HTTP/HTTPS (NGINX)
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https

# Allow WireGuard
sudo firewall-cmd --permanent --add-port=51820/udp

# Monitoring ports (restrict to WireGuard subnet later)
sudo firewall-cmd --permanent --add-port=9100/tcp
sudo firewall-cmd --permanent --add-port=9113/tcp
sudo firewall-cmd --permanent --add-port=9209/tcp
sudo firewall-cmd --permanent --add-port=9835/tcp
sudo firewall-cmd --permanent --add-port=19999/tcp

# Engine ports
sudo firewall-cmd --permanent --add-port=8001/tcp
sudo firewall-cmd --permanent --add-port=8002/tcp
sudo firewall-cmd --permanent --add-port=8003/tcp
sudo firewall-cmd --permanent --add-port=8004/tcp
sudo firewall-cmd --permanent --add-port=8005/tcp

# Apply
sudo firewall-cmd --reload

# Verify
sudo firewall-cmd --list-all
```

## 1.9 WireGuard VPN

This creates an encrypted tunnel between GEX44 and Contabo for monitoring.

```bash
sudo dnf install -y wireguard-tools

# Generate keys
cd /etc/wireguard
umask 077
wg genkey | tee private.key | wg pubkey > public.key

# Show your public key (you'll need this for Contabo setup)
cat public.key
```

Create the config — you'll fill in Contabo's public key later (Step 4):
```bash
sudo nano /etc/wireguard/wg0.conf
```

Contents:
```ini
[Interface]
Address = 10.0.0.1/24
ListenPort = 51820
PrivateKey = <PASTE YOUR PRIVATE KEY FROM /etc/wireguard/private.key>

[Peer]
# Contabo monitoring VPS
PublicKey = <CONTABO PUBLIC KEY - fill in during Step 4>
AllowedIPs = 10.0.0.2/32
```

Enable (after Contabo is set up):
```bash
sudo systemctl enable --now wg-quick@wg0
```

## 1.10 Prometheus Exporters

These run on GEX44 and expose metrics for Contabo's Prometheus to scrape.

### Node Exporter (CPU/RAM/disk)
```bash
cd /tmp
wget https://github.com/prometheus/node_exporter/releases/download/v1.8.2/node_exporter-1.8.2.linux-amd64.tar.gz
tar xzf node_exporter-1.8.2.linux-amd64.tar.gz
sudo mv node_exporter-1.8.2.linux-amd64/node_exporter /usr/local/bin/

# Create systemd service
sudo tee /etc/systemd/system/node_exporter.service << 'EOF'
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=metadron
ExecStart=/usr/local/bin/node_exporter
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now node_exporter

# Verify: should return metrics
curl -s http://localhost:9100/metrics | head -5
```

### NVIDIA GPU Exporter
```bash
cd /tmp
wget https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v1.2.1/nvidia_gpu_exporter_1.2.1_linux_x86_64.tar.gz
tar xzf nvidia_gpu_exporter_1.2.1_linux_x86_64.tar.gz
sudo mv nvidia_gpu_exporter /usr/local/bin/

sudo tee /etc/systemd/system/nvidia_gpu_exporter.service << 'EOF'
[Unit]
Description=NVIDIA GPU Exporter
After=network.target

[Service]
User=metadron
ExecStart=/usr/local/bin/nvidia_gpu_exporter
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now nvidia_gpu_exporter
curl -s http://localhost:9835/metrics | head -5
```

### Netdata
```bash
curl -fsSL https://get.netdata.cloud/kickstart.sh | bash

# Apply Metadron config (after cloning repo in Step 2)
# sudo cp /opt/metadron/review/deployment/hetzner/netdata/netdata.conf /etc/netdata/
# sudo cp /opt/metadron/review/deployment/hetzner/netdata/go.d/nvidia_smi.conf /etc/netdata/go.d/
# sudo systemctl restart netdata

# Verify
curl -s http://localhost:19999/api/v1/info | head -5
```

### NGINX Exporter (installed after NGINX in Step 3)

### PM2 Exporter (installed after PM2 ecosystem in Step 2)

## 1.11 Verify Everything

```bash
# System
python3 --version        # 3.11.x
node --version           # v20.x.x
pm2 --version            # 5.x
nvidia-smi               # RTX 4000 SFF Ada, 20GB
nvcc --version           # CUDA 12.4

# Services
systemctl status node_exporter      # active
systemctl status nvidia_gpu_exporter # active
systemctl status netdata            # active
systemctl status firewalld          # active

# Ports
ss -tlnp | grep -E "9100|9835|19999"
```

If all pass, proceed to Step 2.
