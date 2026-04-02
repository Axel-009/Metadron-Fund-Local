import { sqliteTable, text, integer } from "drizzle-orm/sqlite-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = sqliteTable("users", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

// ─── Dashboard Types ───────────────────────────────────────

export interface OrderBookEntry {
  price: number;
  size: number;
  total: number;
  side: "bid" | "ask";
}

export interface MarketNode {
  id: string;
  label: string;
  x: number;
  y: number;
  radius: number;
  sector: string;
  momentum: number;
}

export interface MarketEdge {
  source: string;
  target: string;
  weight: number;
}

export interface OrderDistribution {
  type: string;
  value: number;
  color: string;
}

export interface RiskMetrics {
  pnl: number;
  var_pct: number;
  calmar: number;
  maxDrawdown: number;
  beta: number;
  sector: string;
  dailyReturns: number;
  doi: number;
  riskPct: number;
  betPct: number;
}

export interface ExecutionEntry {
  pair: string;
  ctrl: number;
  sens: number;
  status: number;
  statusColor: "positive" | "negative";
}

export interface EngineStatus {
  id: string;
  name: string;
  level: string;
  status: "online" | "degraded" | "offline";
  latency: number;
  cpu: number;
  memory: number;
  errors: number;
}

export interface PortfolioState {
  nav: number;
  dailyPnl: number;
  dailyPnlPct: number;
  positions: number;
  exposure: number;
  cash: number;
}

export interface SignalEntry {
  timestamp: string;
  signal: string;
  asset: string;
  direction: "long" | "short";
  confidence: number;
}

export interface ReportItem {
  id: string;
  name: string;
  type: string;
  description: string;
  lastGenerated: string;
  status: "ready" | "generating" | "scheduled";
}

export interface SectorPerformance {
  sector: string;
  daily: number;
  weekly: number;
  monthly: number;
  weight: number;
}

export interface BasketHolding {
  ticker: string;
  name: string;
  weight: number;
  shares: number;
  price: number;
  change: number;
  sector: string;
}

export interface AnomalyEntry {
  id: string;
  asset: string;
  type: string;
  zScore: number;
  detected: string;
  severity: "low" | "medium" | "high" | "critical";
}

export interface RelativeValuePair {
  pair: string;
  spreadZScore: number;
  halfLife: number;
  signal: "long" | "short" | "neutral";
  pnl: number;
}
