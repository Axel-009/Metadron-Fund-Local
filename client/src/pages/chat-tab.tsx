import { DashboardPanel } from "@/components/dashboard-panel";
import { useState, useEffect, useRef, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  agent_id: string;
}

interface Recommendation {
  id: number;
  source_agent: string;
  content: string;
  type: string;
  requires_approval: boolean;
  approved: boolean;
  dismissed: boolean;
  prefix: string;
  timestamp: string;
}

interface RufloAgent {
  agent_id: string;
  name: string;
  current_task: string;
  status: "active" | "idle" | "error";
  last_heartbeat: string;
  signals_processed: number;
}

// ═══════════════════════════════════════════════════════════════════════════
// NanoClaw Chat Panel
// ═══════════════════════════════════════════════════════════════════════════

function NanoClawPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [connected, setConnected] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Fetch initial message history
  useEffect(() => {
    fetch("/api/chat/nanoclaw/messages")
      .then((r) => r.json())
      .then((data) => {
        if (data.messages) setMessages(data.messages);
        setConnected(true);
      })
      .catch(() => setConnected(false));

    fetch("/api/chat/recommendations")
      .then((r) => r.json())
      .then((data) => {
        if (data.recommendations) setRecommendations(data.recommendations);
      })
      .catch(() => {});
  }, []);

  // Poll recommendations
  useEffect(() => {
    const iv = setInterval(() => {
      fetch("/api/chat/recommendations")
        .then((r) => r.json())
        .then((data) => {
          if (data.recommendations) setRecommendations(data.recommendations);
        })
        .catch(() => {});
    }, 15000);
    return () => clearInterval(iv);
  }, []);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(async () => {
    const msg = input.trim();
    if (!msg || streaming) return;

    setInput("");
    setMessages((prev) => [
      ...prev,
      { role: "user", content: msg, timestamp: new Date().toISOString(), agent_id: "aj" },
    ]);
    setStreaming(true);

    try {
      const res = await fetch("/api/chat/nanoclaw/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });

      if (!res.ok) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `[Error: ${res.status}]`, timestamp: new Date().toISOString(), agent_id: "nanoclaw" },
        ]);
        setStreaming(false);
        return;
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let fullResponse = "";
      let messageAdded = false;

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const text = decoder.decode(value, { stream: true });
          const lines = text.split("\n");

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const payload = JSON.parse(line.slice(6));
              if (payload.type === "token") {
                fullResponse += payload.content;
                if (!messageAdded) {
                  setMessages((prev) => [
                    ...prev,
                    { role: "assistant", content: fullResponse, timestamp: new Date().toISOString(), agent_id: "nanoclaw" },
                  ]);
                  messageAdded = true;
                } else {
                  setMessages((prev) => {
                    const copy = [...prev];
                    copy[copy.length - 1] = { ...copy[copy.length - 1], content: fullResponse };
                    return copy;
                  });
                }
              } else if (payload.type === "done") {
                break;
              } else if (payload.type === "error") {
                fullResponse += `\n[Error: ${payload.content}]`;
                setMessages((prev) => {
                  const copy = [...prev];
                  copy[copy.length - 1] = { ...copy[copy.length - 1], content: fullResponse };
                  return copy;
                });
              }
            } catch {
              // skip malformed SSE lines
            }
          }
        }
      }
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `[Connection error: ${e}]`, timestamp: new Date().toISOString(), agent_id: "nanoclaw" },
      ]);
    }

    setStreaming(false);
  }, [input, streaming]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleApprove = async (id: number) => {
    try {
      await fetch(`/api/chat/recommendations/${id}/approve`, { method: "POST" });
      setRecommendations((prev) => prev.map((r) => (r.id === id ? { ...r, approved: true } : r)));
    } catch {}
  };

  const handleDismiss = async (id: number) => {
    try {
      await fetch(`/api/chat/recommendations/${id}/dismiss`, { method: "POST" });
      setRecommendations((prev) => prev.filter((r) => r.id !== id));
    } catch {}
  };

  const pendingRecs = recommendations.filter((r) => !r.approved && !r.dismissed);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-terminal-border/50 flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono font-bold tracking-wider text-terminal-accent">NANOCLAW OPERATOR</span>
          <div className={`flex items-center gap-1 ${connected ? "text-terminal-positive" : "text-terminal-text-faint"}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-terminal-positive animate-pulse" : "bg-terminal-text-faint"}`} />
            <span className="text-[8px] font-mono">{connected ? "ONLINE" : "OFFLINE"}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[8px] font-mono text-terminal-accent/70 px-1.5 py-0.5 rounded border border-terminal-accent/20 bg-terminal-accent/5">
            WRITE (with permission)
          </span>
          <span className="text-[7px] font-mono text-terminal-text-faint">
            WRITE ACTIONS REQUIRE EXPLICIT INSTRUCTION
          </span>
        </div>
      </div>

      {/* Message Stream */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2 min-h-0">
        {messages.length === 0 && pendingRecs.length === 0 && (
          <div className="text-center py-12 text-terminal-text-faint text-[10px] font-mono">
            NanoClaw operator ready. Type a message to begin.
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[75%] rounded px-3 py-2 ${
                msg.role === "user"
                  ? "bg-terminal-accent/10 border border-terminal-accent/30 text-terminal-text-primary"
                  : "bg-terminal-surface border border-terminal-border/50"
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className={`text-[8px] font-mono font-bold tracking-wider ${
                  msg.role === "user" ? "text-terminal-accent" : "text-terminal-accent"
                }`}>
                  {msg.role === "user" ? "AJ" : "NANOCLAW"}
                </span>
                <span className="text-[7px] font-mono text-terminal-text-faint">
                  {new Date(msg.timestamp).toLocaleTimeString("en-US", { hour12: false })}
                </span>
              </div>
              <div className="text-[11px] font-mono text-terminal-text-primary whitespace-pre-wrap leading-relaxed">
                {msg.content}
              </div>
            </div>
          </div>
        ))}

        {/* CEO Recommendation cards */}
        {pendingRecs.map((rec) => (
          <div
            key={`rec-${rec.id}`}
            className="rounded px-3 py-2 border border-amber-400/30 bg-amber-400/5"
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[8px] font-mono font-bold tracking-wider text-amber-400">
                CEO RECOMMENDATION
              </span>
              <span className="text-[7px] font-mono text-terminal-text-faint">
                {new Date(rec.timestamp).toLocaleTimeString("en-US", { hour12: false })}
              </span>
            </div>
            <div className="text-[11px] font-mono text-terminal-text-primary whitespace-pre-wrap leading-relaxed mb-2">
              {rec.content}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => handleApprove(rec.id)}
                className="px-2 py-0.5 text-[9px] font-mono font-bold rounded border border-terminal-positive/40 text-terminal-positive bg-terminal-positive/10 hover:bg-terminal-positive/20 transition-colors"
              >
                APPROVE
              </button>
              <button
                onClick={() => handleDismiss(rec.id)}
                className="px-2 py-0.5 text-[9px] font-mono font-bold rounded border border-terminal-text-faint/40 text-terminal-text-faint hover:text-terminal-text-muted hover:bg-white/[0.03] transition-colors"
              >
                DISMISS
              </button>
            </div>
          </div>
        ))}

        {streaming && (
          <div className="flex items-center gap-2 px-3 py-1">
            <span className="w-1.5 h-1.5 rounded-full bg-terminal-accent animate-pulse" />
            <span className="text-[9px] font-mono text-terminal-accent/60">NanoClaw is responding...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="flex-shrink-0 border-t border-terminal-border/50 p-2">
        <div className="flex items-end gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Instruct NanoClaw..."
            rows={1}
            className="flex-1 bg-terminal-surface border border-terminal-border rounded px-3 py-2 text-[11px] font-mono text-terminal-text-primary placeholder:text-terminal-text-faint/50 resize-none focus:outline-none focus:border-terminal-accent/40 transition-colors"
            style={{ minHeight: "36px", maxHeight: "120px" }}
            disabled={streaming}
          />
          <button
            onClick={sendMessage}
            disabled={streaming || !input.trim()}
            className="px-3 py-2 rounded bg-terminal-accent/20 border border-terminal-accent/40 text-terminal-accent text-[10px] font-mono font-bold hover:bg-terminal-accent/30 disabled:opacity-30 disabled:cursor-not-allowed transition-colors flex-shrink-0"
          >
            SEND
          </button>
        </div>
        <div className="flex items-center justify-between mt-1">
          <span className="text-[7px] font-mono text-terminal-text-faint">
            Enter to send, Shift+Enter for newline
          </span>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Ruflo Agent Panel
// ═══════════════════════════════════════════════════════════════════════════

function RufloPanel() {
  const [agents, setAgents] = useState<RufloAgent[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [responses, setResponses] = useState<Array<{ agent_id: string; name: string; response: string; timestamp: string }>>([]);
  const [sending, setSending] = useState(false);

  // Fetch agents
  useEffect(() => {
    const fetchAgents = () => {
      fetch("/api/chat/ruflo/agents")
        .then((r) => r.json())
        .then((data) => {
          if (data.agents) setAgents(data.agents);
        })
        .catch(() => {});
    };
    fetchAgents();
    const iv = setInterval(fetchAgents, 15000);
    return () => clearInterval(iv);
  }, []);

  const sendMessage = useCallback(async () => {
    const msg = input.trim();
    if (!msg || sending) return;

    setInput("");
    setSending(true);

    try {
      const target = selectedAgent || "all";
      const res = await fetch("/api/chat/ruflo/send", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent_id: target, message: msg }),
      });
      const data = await res.json();

      if (data.responses) {
        setResponses((prev) => [...prev, ...data.responses].slice(-50));
      } else if (data.response) {
        setResponses((prev) => [...prev, data].slice(-50));
      }
    } catch {}

    setSending(false);
  }, [input, selectedAgent, sending]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const statusColor = (status: string) => {
    if (status === "active") return "bg-terminal-positive";
    if (status === "idle") return "bg-terminal-warning";
    return "bg-terminal-negative";
  };

  const statusText = (status: string) => {
    if (status === "active") return "text-terminal-positive";
    if (status === "idle") return "text-terminal-warning";
    return "text-terminal-negative";
  };

  return (
    <div className="flex h-full">
      {/* Left: Agent list */}
      <div className="w-56 border-r border-terminal-border/50 flex flex-col flex-shrink-0">
        <div className="px-3 py-2 border-b border-terminal-border/50">
          <span className="text-[10px] font-mono font-bold tracking-wider text-terminal-blue">RUFLO AGENT SWARM</span>
        </div>
        <div className="flex-1 overflow-y-auto">
          {/* Broadcast option */}
          <button
            onClick={() => setSelectedAgent(null)}
            className={`w-full px-3 py-2 text-left border-b border-terminal-border/20 transition-colors ${
              selectedAgent === null ? "bg-terminal-blue/10 border-l-2 border-l-terminal-blue" : "hover:bg-white/[0.02]"
            }`}
          >
            <div className="text-[9px] font-mono font-bold text-terminal-blue">@ALL BROADCAST</div>
            <div className="text-[8px] font-mono text-terminal-text-faint">Message all agents</div>
          </button>

          {agents.map((agent) => (
            <button
              key={agent.agent_id}
              onClick={() => setSelectedAgent(agent.agent_id)}
              className={`w-full px-3 py-2 text-left border-b border-terminal-border/20 transition-colors ${
                selectedAgent === agent.agent_id ? "bg-terminal-blue/10 border-l-2 border-l-terminal-blue" : "hover:bg-white/[0.02]"
              }`}
            >
              <div className="flex items-center gap-1.5 mb-0.5">
                <span className={`w-1.5 h-1.5 rounded-full ${statusColor(agent.status)}`} />
                <span className="text-[9px] font-mono font-bold text-terminal-text-primary">{agent.name}</span>
              </div>
              <div className="text-[8px] font-mono text-terminal-text-faint truncate">{agent.current_task}</div>
              <div className="flex items-center gap-2 mt-0.5">
                <span className={`text-[7px] font-mono ${statusText(agent.status)}`}>{agent.status.toUpperCase()}</span>
                <span className="text-[7px] font-mono text-terminal-text-faint">{agent.signals_processed} signals</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Right: Chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="px-3 py-2 border-b border-terminal-border/50 flex-shrink-0">
          <span className="text-[9px] font-mono text-terminal-text-muted">
            {selectedAgent
              ? `Chatting with: ${agents.find((a) => a.agent_id === selectedAgent)?.name || selectedAgent}`
              : "Broadcasting to all Ruflo agents"}
          </span>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-2 min-h-0">
          {responses.length === 0 && (
            <div className="text-center py-12 text-terminal-text-faint text-[10px] font-mono">
              Select an agent or broadcast to all. No write commands accepted — Ruflo agents are read-only.
            </div>
          )}

          {responses.map((resp, i) => (
            <div key={i} className="bg-terminal-surface border border-terminal-border/50 rounded px-3 py-2">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[8px] font-mono font-bold tracking-wider text-terminal-blue">{resp.name}</span>
                <span className="text-[7px] font-mono text-terminal-text-faint">
                  {new Date(resp.timestamp).toLocaleTimeString("en-US", { hour12: false })}
                </span>
              </div>
              <div className="text-[11px] font-mono text-terminal-text-primary whitespace-pre-wrap leading-relaxed">
                {resp.response}
              </div>
            </div>
          ))}
        </div>

        {/* Input */}
        <div className="flex-shrink-0 border-t border-terminal-border/50 p-2">
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={selectedAgent ? `Message @${agents.find((a) => a.agent_id === selectedAgent)?.name || selectedAgent}...` : "Message @all Ruflo agents..."}
              rows={1}
              className="flex-1 bg-terminal-surface border border-terminal-border rounded px-3 py-2 text-[11px] font-mono text-terminal-text-primary placeholder:text-terminal-text-faint/50 resize-none focus:outline-none focus:border-terminal-blue/40 transition-colors"
              style={{ minHeight: "36px", maxHeight: "120px" }}
              disabled={sending}
            />
            <button
              onClick={sendMessage}
              disabled={sending || !input.trim()}
              className="px-3 py-2 rounded bg-terminal-blue/20 border border-terminal-blue/40 text-terminal-blue text-[10px] font-mono font-bold hover:bg-terminal-blue/30 disabled:opacity-30 disabled:cursor-not-allowed transition-colors flex-shrink-0"
            >
              SEND
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Chat Tab
// ═══════════════════════════════════════════════════════════════════════════

export default function ChatTab() {
  const [activeChannel, setActiveChannel] = useState<"nanoclaw" | "ruflo">("nanoclaw");

  return (
    <div className="h-full p-[2px] overflow-hidden" data-testid="chat-tab">
      <DashboardPanel
        title="CHAT"
        noPadding
        headerRight={
          <div className="flex items-center gap-1">
            <button
              onClick={() => setActiveChannel("nanoclaw")}
              className={`px-2 py-0.5 text-[9px] font-mono font-bold rounded-sm transition-colors ${
                activeChannel === "nanoclaw"
                  ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30"
                  : "text-terminal-text-muted hover:text-terminal-text-primary hover:bg-white/[0.03]"
              }`}
            >
              NANOCLAW
            </button>
            <button
              onClick={() => setActiveChannel("ruflo")}
              className={`px-2 py-0.5 text-[9px] font-mono font-bold rounded-sm transition-colors ${
                activeChannel === "ruflo"
                  ? "bg-terminal-blue/15 text-terminal-blue border border-terminal-blue/30"
                  : "text-terminal-text-muted hover:text-terminal-text-primary hover:bg-white/[0.03]"
              }`}
            >
              RUFLO
            </button>
          </div>
        }
        className="h-full"
      >
        {activeChannel === "nanoclaw" ? <NanoClawPanel /> : <RufloPanel />}
      </DashboardPanel>
    </div>
  );
}
