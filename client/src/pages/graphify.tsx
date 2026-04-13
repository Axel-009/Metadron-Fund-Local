import { useState } from "react";
import { useEngineQuery } from "@/hooks/use-engine-api";

// ═══════════ TYPES ═══════════

interface GodNode {
  id: string;
  label: string;
  degree: number;
}

interface GraphifyStatus {
  available: boolean;
  graph_exists: boolean;
  report_exists: boolean;
  graph_size_kb: number;
  god_nodes: GodNode[];
  god_node_count: number;
  timestamp: string;
}

interface GraphifyReport {
  report: string;
  has_report: boolean;
  timestamp: string;
}

interface QueryResult {
  question: string;
  answer: string;
  timestamp: string;
  error?: string;
}

// ═══════════ SUB-COMPONENTS ═══════════

function StatusPanel({ status }: { status: GraphifyStatus | null }) {
  if (!status) return <div className="text-terminal-text-muted text-xs p-4">Loading status...</div>;

  return (
    <div className="border border-terminal-border rounded p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-terminal-text-primary text-xs font-bold tracking-wider">KNOWLEDGE GRAPH STATUS</h3>
        <span className={`text-[10px] px-2 py-0.5 rounded ${status.available ? "bg-terminal-positive/20 text-terminal-positive" : "bg-terminal-negative/20 text-terminal-negative"}`}>
          {status.available ? "ONLINE" : "OFFLINE"}
        </span>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[10px]">
        <div>
          <span className="text-terminal-text-muted block">Graph File</span>
          <span className={status.graph_exists ? "text-terminal-positive" : "text-terminal-negative"}>
            {status.graph_exists ? `EXISTS (${status.graph_size_kb}KB)` : "NOT GENERATED"}
          </span>
        </div>
        <div>
          <span className="text-terminal-text-muted block">Report</span>
          <span className={status.report_exists ? "text-terminal-positive" : "text-terminal-text-faint"}>
            {status.report_exists ? "AVAILABLE" : "NOT GENERATED"}
          </span>
        </div>
        <div>
          <span className="text-terminal-text-muted block">God Nodes</span>
          <span className="text-terminal-text-primary">{status.god_node_count}</span>
        </div>
        <div>
          <span className="text-terminal-text-muted block">Last Updated</span>
          <span className="text-terminal-text-faint">{status.timestamp ? new Date(status.timestamp).toLocaleTimeString() : "—"}</span>
        </div>
      </div>
    </div>
  );
}

function GenerateButton() {
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleGenerate = async () => {
    setGenerating(true);
    setResult(null);
    try {
      const res = await fetch("/api/engine/graphify/generate", { method: "POST" });
      const data = await res.json();
      setResult(data.status === "generating"
        ? `Graph generation started (PID: ${data.pid}). Refresh status in 1-5 minutes.`
        : data.message || "Unknown error");
    } catch (e) {
      setResult(`Failed: ${e}`);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="border border-terminal-border rounded p-4 space-y-2">
      <h3 className="text-terminal-text-primary text-xs font-bold tracking-wider">GENERATE GRAPH</h3>
      <p className="text-[10px] text-terminal-text-muted">
        Run <code className="text-terminal-accent">graphify .</code> to scan the codebase and build the knowledge graph.
        This identifies god nodes, architectural connections, and concept relationships.
      </p>
      <button
        onClick={handleGenerate}
        disabled={generating}
        className="px-3 py-1.5 text-[10px] font-mono rounded bg-terminal-accent/20 text-terminal-accent border border-terminal-accent/30 hover:bg-terminal-accent/30 disabled:opacity-40"
      >
        {generating ? "GENERATING..." : "GENERATE KNOWLEDGE GRAPH"}
      </button>
      {result && <div className="text-[10px] text-terminal-text-faint mt-1">{result}</div>}
    </div>
  );
}

function GodNodesTable({ nodes }: { nodes: GodNode[] }) {
  if (!nodes || nodes.length === 0) {
    return <div className="text-terminal-text-muted text-[10px] p-4">No god nodes — generate the graph first.</div>;
  }

  const maxDegree = Math.max(...nodes.map(n => n.degree));

  return (
    <div className="border border-terminal-border rounded overflow-hidden">
      <div className="px-4 py-2 border-b border-terminal-border">
        <h3 className="text-terminal-text-primary text-xs font-bold tracking-wider">GOD NODES — Highest Connectivity</h3>
        <p className="text-[10px] text-terminal-text-muted">Concepts with the most connections in the codebase graph</p>
      </div>
      <div className="max-h-[400px] overflow-y-auto">
        <table className="w-full text-[10px]">
          <thead className="sticky top-0 bg-terminal-bg">
            <tr className="text-terminal-text-muted border-b border-terminal-border">
              <th className="text-left px-4 py-1.5 w-10">#</th>
              <th className="text-left px-4 py-1.5">NODE</th>
              <th className="text-left px-4 py-1.5">LABEL</th>
              <th className="text-right px-4 py-1.5 w-20">DEGREE</th>
              <th className="text-left px-4 py-1.5 w-40">CONNECTIVITY</th>
            </tr>
          </thead>
          <tbody>
            {nodes.map((node, i) => (
              <tr key={node.id || i} className="border-b border-terminal-border/30 hover:bg-terminal-accent/5">
                <td className="px-4 py-1.5 text-terminal-text-muted">{i + 1}</td>
                <td className="px-4 py-1.5 text-terminal-accent font-mono">{node.id}</td>
                <td className="px-4 py-1.5 text-terminal-text-primary">{node.label}</td>
                <td className="px-4 py-1.5 text-right text-terminal-text-primary">{node.degree}</td>
                <td className="px-4 py-1.5">
                  <div className="w-full bg-terminal-border/30 rounded-full h-1.5">
                    <div
                      className="bg-terminal-accent h-1.5 rounded-full"
                      style={{ width: `${maxDegree > 0 ? (node.degree / maxDegree) * 100 : 0}%` }}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function QueryInterface() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<QueryResult[]>([]);
  const [loading, setLoading] = useState(false);

  const handleQuery = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/engine/graphify/query?q=${encodeURIComponent(query)}`);
      const data: QueryResult = await res.json();
      setResults(prev => [data, ...prev].slice(0, 20));
    } catch (e) {
      setResults(prev => [{ question: query, answer: `Error: ${e}`, timestamp: new Date().toISOString() }, ...prev]);
    } finally {
      setLoading(false);
      setQuery("");
    }
  };

  return (
    <div className="border border-terminal-border rounded p-4 space-y-3">
      <h3 className="text-terminal-text-primary text-xs font-bold tracking-wider">QUERY KNOWLEDGE GRAPH</h3>
      <div className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleQuery()}
          placeholder="Ask about the codebase architecture..."
          className="flex-1 bg-terminal-bg border border-terminal-border rounded px-3 py-1.5 text-[10px] text-terminal-text-primary placeholder:text-terminal-text-muted focus:border-terminal-accent outline-none"
        />
        <button
          onClick={handleQuery}
          disabled={loading || !query.trim()}
          className="px-3 py-1.5 text-[10px] font-mono rounded bg-terminal-accent/20 text-terminal-accent border border-terminal-accent/30 hover:bg-terminal-accent/30 disabled:opacity-40"
        >
          {loading ? "..." : "QUERY"}
        </button>
      </div>
      {results.length > 0 && (
        <div className="space-y-2 max-h-[300px] overflow-y-auto">
          {results.map((r, i) => (
            <div key={i} className="border border-terminal-border/30 rounded p-2 text-[10px]">
              <div className="text-terminal-accent font-mono mb-1">Q: {r.question}</div>
              <div className="text-terminal-text-primary whitespace-pre-wrap">{r.answer || r.error || "No answer"}</div>
              <div className="text-terminal-text-muted mt-1">{new Date(r.timestamp).toLocaleTimeString()}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ReportViewer({ report }: { report: GraphifyReport | null }) {
  if (!report || !report.has_report) {
    return (
      <div className="border border-terminal-border rounded p-4">
        <h3 className="text-terminal-text-primary text-xs font-bold tracking-wider mb-2">GRAPH REPORT</h3>
        <p className="text-[10px] text-terminal-text-muted">No report available. Generate the graph first.</p>
      </div>
    );
  }

  return (
    <div className="border border-terminal-border rounded overflow-hidden">
      <div className="px-4 py-2 border-b border-terminal-border">
        <h3 className="text-terminal-text-primary text-xs font-bold tracking-wider">GRAPH REPORT</h3>
      </div>
      <div className="p-4 max-h-[400px] overflow-y-auto">
        <pre className="text-[10px] text-terminal-text-primary whitespace-pre-wrap font-mono leading-relaxed">
          {report.report}
        </pre>
      </div>
    </div>
  );
}

// ═══════════ MAIN PAGE ═══════════

export default function GraphifyPage() {
  const { data: status } = useEngineQuery<GraphifyStatus>("/graphify/status", 60_000);
  const { data: report } = useEngineQuery<GraphifyReport>("/graphify/report", 120_000);

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-terminal-text-primary text-sm font-bold tracking-wider">GRAPHIFY KNOWLEDGE GRAPH</h1>
          <p className="text-[10px] text-terminal-text-muted">Codebase architecture analysis, concept mapping, and agent-queryable knowledge graph</p>
        </div>
      </div>

      {/* Status + Generate */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <StatusPanel status={status} />
        <GenerateButton />
      </div>

      {/* God Nodes */}
      <GodNodesTable nodes={status?.god_nodes || []} />

      {/* Query + Report side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <QueryInterface />
        <ReportViewer report={report} />
      </div>
    </div>
  );
}
