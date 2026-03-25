"use client";

import { useEffect, useState } from "react";

import { IssueCreator } from "./components/IssueCreator";
import { IssueList } from "./components/IssueList";
import { TraceViewer } from "./components/TraceViewer";
import { Issue, TraceStats, api } from "../lib/api";

export default function Dashboard() {
  const [issues, setIssues] = useState<Issue[]>([]);
  const [stats, setStats] = useState<TraceStats | null>(null);
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      const [issueData, statsData] = await Promise.all([api.listIssues(), api.getTraceStats()]);
      setIssues(issueData);
      setStats(statsData);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to connect to API");
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleIssueCreated = (issue: Issue) => {
    setIssues((prev) => [issue, ...prev]);
    loadData();
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">AgentWatch</h1>
        <p className="text-gray-400 mt-1 text-sm">Few-shot silent failure detection for AI agents</p>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-950/50 border border-red-800 rounded-lg text-red-300 text-sm">{error}</div>
      )}

      {/* Stats bar */}
      {stats && (
        <div className="grid grid-cols-4 gap-4 mb-8">
          <StatCard label="Total traces" value={stats.total_traces.toLocaleString()} />
          <StatCard label="Unique users" value={stats.unique_users.toLocaleString()} />
          <StatCard label="Avg tokens" value={Math.round(stats.avg_tokens).toLocaleString()} />
          <StatCard label="Avg latency" value={`${Math.round(stats.avg_duration_ms)}ms`} />
        </div>
      )}

      {/* Issue Creator */}
      <IssueCreator onCreated={handleIssueCreated} />

      {/* Issues List */}
      <IssueList issues={issues} onSelectTrace={(id) => setSelectedTraceId(id)} />

      {/* Trace Viewer Modal */}
      {selectedTraceId && <TraceViewer traceId={selectedTraceId} onClose={() => setSelectedTraceId(null)} />}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="text-xl font-semibold mt-1">{value}</div>
    </div>
  );
}

