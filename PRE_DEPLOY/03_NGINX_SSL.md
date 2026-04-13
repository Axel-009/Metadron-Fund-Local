# Step 3 — NGINX, SSL, Landing Page

## 3.1 Install NGINX

```bash
sudo dnf install -y nginx
sudo systemctl enable nginx
```

## 3.2 Cloudflare SSL Certificates

You need Cloudflare Origin Certificates so NGINX can serve HTTPS.

1. Go to **Cloudflare Dashboard** → your domain → **SSL/TLS** → **Origin Server**
2. Click **Create Certificate**
3. Hostnames: `metadroncapital.com, *.metadroncapital.com`
4. Validity: 15 years
5. Key format: RSA (2048)
6. Click **Create** — download the certificate and private key

Place them on the server:
```bash
sudo mkdir -p /etc/ssl/cloudflare

# Paste the certificate
sudo nano /etc/ssl/cloudflare/origin.pem
# Paste the full certificate text, save

# Paste the private key
sudo nano /etc/ssl/cloudflare/origin-key.pem
# Paste the full key text, save

# Lock down permissions
sudo chmod 600 /etc/ssl/cloudflare/origin-key.pem
sudo chmod 644 /etc/ssl/cloudflare/origin.pem
```

In Cloudflare Dashboard:
- **SSL/TLS** → **Overview** → Set mode to **Full (Strict)**
- This tells Cloudflare to validate your origin certificate

## 3.3 Marketing / Landing Page

The landing page is a static site. NGINX serves it directly (no Express needed).

```bash
# Create the marketing site directory
sudo mkdir -p /opt/metadron/marketing

# Copy your static files (HTML, CSS, JS, images) here
# These are the login page + landing page files
sudo cp -r /opt/metadron/client/dist/* /opt/metadron/marketing/ 2>/dev/null || true

# Or if you have a separate marketing site:
# sudo cp -r /path/to/your/marketing/site/* /opt/metadron/marketing/
```

The NGINX config serves:
- `/` → Static marketing site (landing page + login)
- `/terminal/` → React terminal (proxied to Express :5000)
- `/api/engine/` → Engine API (proxied through Express to FastAPI :8001)

## 3.4 Deploy NGINX Config

```bash
# Copy the Metadron NGINX config
sudo cp /opt/metadron/review/deployment/hetzner/nginx/metadroncapital.conf /etc/nginx/conf.d/

# Edit to replace YOUR_DOMAIN if needed
sudo nano /etc/nginx/conf.d/metadroncapital.conf

# Test the config
sudo nginx -t
# Should say: syntax is ok, test is successful

# Start NGINX
sudo systemctl restart nginx
```

## 3.5 Install NGINX Prometheus Exporter

```bash
cd /tmp
wget https://github.com/nginxinc/nginx-prometheus-exporter/releases/download/v1.3.0/nginx-prometheus-exporter_1.3.0_linux_amd64.tar.gz
tar xzf nginx-prometheus-exporter_1.3.0_linux_amd64.tar.gz
sudo mv nginx-prometheus-exporter /usr/local/bin/

sudo tee /etc/systemd/system/nginx_exporter.service << 'EOF'
[Unit]
Description=NGINX Prometheus Exporter
After=nginx.service

[Service]
User=metadron
ExecStart=/usr/local/bin/nginx-prometheus-exporter -nginx.scrape-uri=http://127.0.0.1/nginx_status
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now nginx_exporter
```

## 3.6 How the Reverse Proxy Works

```
User's browser
    ↓ HTTPS (port 443)
Cloudflare CDN (DDoS protection + caching)
    ↓ HTTPS (Origin Certificate)
NGINX on GEX44
    ├── /                 → /opt/metadron/marketing/ (static HTML)
    ├── /terminal/        → Express :5000 (React terminal, WebSocket)
    ├── /api/engine/      → Express :5000 → FastAPI :8001
    ├── /metrics          → node_exporter :9100 (WireGuard only)
    └── /nginx_status     → NGINX stub_status (WireGuard only)
```

The landing page loads instantly (static files, no server processing).
When a user logs in and enters the terminal, they're proxied to Express
which serves the React app and forwards API calls to FastAPI.

## 3.7 Verify

```bash
# From another machine, test your domain
curl -I https://metadroncapital.com
# Should return: HTTP/2 200

# Test the terminal path
curl -I https://metadroncapital.com/terminal/
# Should return: HTTP/2 200

# Test the API through NGINX
curl https://metadroncapital.com/api/engine/health
# Should return: {"status":"ok","timestamp":"..."}
```

If you get certificate errors, check:
1. Cloudflare SSL mode is "Full (Strict)"
2. Origin certificate is correctly placed
3. NGINX config points to the right cert paths
