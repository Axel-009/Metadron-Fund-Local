import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import http from "http";

const ENGINE_API_HOST = process.env.ENGINE_API_HOST || "127.0.0.1";
const ENGINE_API_PORT = parseInt(process.env.ENGINE_API_PORT || "8001", 10);

/**
 * Proxy requests matching /api/engine/* to the Python FastAPI server.
 * Falls through to Express mock routes if the Python server is unreachable.
 */
function proxyToEngine(
  app: Express,
  prefix: string = "/api/engine"
): void {
  app.all(`${prefix}/{*path}`, (req, res, next) => {
    const targetPath = req.originalUrl;
    const options: http.RequestOptions = {
      hostname: ENGINE_API_HOST,
      port: ENGINE_API_PORT,
      path: targetPath,
      method: req.method,
      headers: {
        ...req.headers,
        host: `${ENGINE_API_HOST}:${ENGINE_API_PORT}`,
      },
      timeout: 30000,
    };

    const proxyReq = http.request(options, (proxyRes) => {
      res.writeHead(proxyRes.statusCode || 500, proxyRes.headers);
      proxyRes.pipe(res, { end: true });
    });

    proxyReq.on("error", (_err) => {
      // Python server not running — fall through to mock routes
      next();
    });

    proxyReq.on("timeout", () => {
      proxyReq.destroy();
      next();
    });

    if (req.body && Object.keys(req.body).length > 0) {
      proxyReq.write(JSON.stringify(req.body));
    }
    proxyReq.end();
  });
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {

  // ─── Proxy /api/engine/* to Python FastAPI ─────────────
  proxyToEngine(app);

  // ─── Legacy mock endpoints (fallback if Python is down) ─
  // These return static data so the frontend can render even without the engine.

  app.get("/api/portfolio/live", (_req, res) => {
    res.json({
      nav: 128450320 + Math.floor(Math.random() * 100000),
      dailyPnl: 842150 + Math.floor(Math.random() * 50000 - 25000),
      dailyPnlPct: 0.66 + (Math.random() * 0.2 - 0.1),
      positions: 47,
      exposure: 0.92,
      cash: 10250000,
    });
  });

  app.get("/api/signals/stream", (_req, res) => {
    const signals = Array.from({ length: 10 }, (_, i) => ({
      timestamp: new Date(Date.now() - i * 60000).toISOString(),
      signal: ["MOMENTUM", "REVERSION", "BREAKOUT", "PAIRS"][Math.floor(Math.random() * 4)],
      asset: ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"][Math.floor(Math.random() * 5)],
      direction: Math.random() > 0.5 ? "long" : "short",
      confidence: +(Math.random() * 40 + 60).toFixed(1),
    }));
    res.json(signals);
  });

  app.get("/api/engines/status", (_req, res) => {
    const engines = [
      { id: "L1", name: "Data Ingestion", level: "L1", status: "online", latency: 1.2, cpu: 34, memory: 42, errors: 0 },
      { id: "L2", name: "Signal Gen", level: "L2", status: "online", latency: 3.4, cpu: 67, memory: 58, errors: 0 },
      { id: "L3", name: "Risk Engine", level: "L3", status: "online", latency: 2.1, cpu: 45, memory: 51, errors: 1 },
      { id: "L4", name: "Execution", level: "L4", status: "online", latency: 0.8, cpu: 28, memory: 35, errors: 0 },
      { id: "L5", name: "ML Pipeline", level: "L5", status: "degraded", latency: 12.4, cpu: 89, memory: 82, errors: 3 },
      { id: "L6", name: "Backtest", level: "L6", status: "online", latency: 45.2, cpu: 72, memory: 68, errors: 0 },
      { id: "L7", name: "Reporting", level: "L7", status: "online", latency: 8.6, cpu: 22, memory: 31, errors: 0 },
    ];
    res.json(engines);
  });

  app.get("/api/market/wrap", (_req, res) => {
    res.json({
      direction: "risk-on",
      narrative: "Markets are in a risk-on regime with strong breadth improvement. Tech continues to lead.",
      vix: 14.22,
      spxChange: 0.84,
      regime: "low_vol",
    });
  });

  app.get("/api/risk/portfolio", (_req, res) => {
    res.json({
      pnl: 45750,
      var_pct: 1.22,
      calmar: 1.55,
      maxDrawdown: -8.78,
      beta: 0.92,
      sharpe: 1.82,
      sortino: 2.45,
      totalMargin: 4250000,
    });
  });

  app.get("/api/allocation/basket", (_req, res) => {
    const holdings = [
      { ticker: "AAPL", name: "Apple Inc", weight: 8.5, shares: 1200, price: 189.45, change: 1.2, sector: "Technology" },
      { ticker: "MSFT", name: "Microsoft Corp", weight: 7.8, shares: 800, price: 420.12, change: 0.8, sector: "Technology" },
      { ticker: "NVDA", name: "NVIDIA Corp", weight: 6.2, shares: 500, price: 875.30, change: 2.4, sector: "Technology" },
    ];
    res.json(holdings);
  });

  app.get("/api/ml/anomalies", (_req, res) => {
    const anomalies = [
      { id: "1", asset: "SMCI", type: "Volume Spike", zScore: 4.2, detected: "2m ago", severity: "critical" },
      { id: "2", asset: "GME", type: "Options Flow", zScore: 3.8, detected: "15m ago", severity: "high" },
      { id: "3", asset: "BTC/USD", type: "Correlation Break", zScore: 2.9, detected: "1h ago", severity: "medium" },
    ];
    res.json(anomalies);
  });

  app.get("/api/reports", (_req, res) => {
    const reports = [
      { id: "1", name: "Platinum Report", type: "Executive", description: "Comprehensive portfolio overview", lastGenerated: "Apr 01, 2026 — 18:00", status: "ready" },
      { id: "2", name: "Daily P&L Report", type: "Operations", description: "Detailed daily P&L breakdown", lastGenerated: "Apr 01, 2026 — 16:30", status: "ready" },
    ];
    res.json(reports);
  });

  return httpServer;
}
