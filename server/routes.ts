import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import http from "http";

const ENGINE_API_HOST = process.env.ENGINE_API_HOST || "127.0.0.1";
const ENGINE_API_PORT = parseInt(process.env.ENGINE_API_PORT || "8001", 10);

/**
 * Proxy all /api/engine/* requests to the Python FastAPI server.
 * Returns 502 if the engine server is unreachable — frontend handles the error.
 */
function proxyToEngine(
  app: Express,
  prefix: string = "/api/engine"
): void {
  app.all(`${prefix}/{*path}`, (req, res) => {
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
      res.status(502).json({
        error: "Engine API unavailable",
        detail: "Python engine server is not running on port " + ENGINE_API_PORT,
        timestamp: new Date().toISOString(),
      });
    });

    proxyReq.on("timeout", () => {
      proxyReq.destroy();
      res.status(504).json({
        error: "Engine API timeout",
        detail: "Request to engine server timed out after 30s",
        timestamp: new Date().toISOString(),
      });
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

  // All API traffic goes through the Python engine — no mock fallbacks
  proxyToEngine(app);

  // Allocation engine endpoints → Python FastAPI
  // Covers: /api/allocation/rules, /status, /slate, /scan/*, /collateral/*
  proxyToEngine(app, "/api/allocation");

  return httpServer;
}
