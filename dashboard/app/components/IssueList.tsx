"use client";

import { useEffect, useState } from "react";

import { Issue, IssueFlag, api } from "../../lib/api";

interface Props {
  issues: Issue[];
  onSelectTrace: (traceId: string) => void;
}

export function IssueList({ issues, onSelectTrace }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [flags, setFlags] = useState<IssueFlag[]>([]);
  const [loadingFlags, setLoadingFlags] = useState(false);

  const handleExpand = async (issueId: string) => {
    if (expanded === issueId) {
      setExpanded(null);
      return;
    }
    setExpanded(issueId);
    setLoadingFlags(true);
    try {
      const data = await api.getIssueFlags(issueId, 20);
      setFlags(data);
    } catch {
      setFlags([]);
    } finally {
      setLoadingFlags(false);
    }
  };

  if (issues.length === 0) {
    return <div className="text-center py-12 text-gray-600 text-sm">No issues tracked yet. Create one above to get started.</div>;
  }

  return (
    <div className="space-y-4">
      <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wide">Tracked issues</h2>

      {issues.map((issue) => (
        <div key={issue.issue_id} className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
          {/* Issue header */}
          <button
            onClick={() => handleExpand(issue.issue_id)}
            className="w-full px-6 py-4 flex items-center justify-between text-left hover:bg-gray-800/50 transition-colors"
          >
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-gray-100 truncate">{issue.description}</div>
              <div className="flex gap-4 mt-2 text-xs text-gray-500">
                <span>
                  <span className="text-red-400 font-medium">{issue.n_flagged}</span> flagged
                </span>
                <span>{(issue.frequency * 100).toFixed(1)}% of traces</span>
                <span>{issue.n_affected_users} users affected</span>
                <span>
                  Best: {issue.best_approach} (F1: {(issue.metrics.f1 ?? 0).toFixed(2)})
                </span>
              </div>
            </div>

            {/* Status badge */}
            <div className="ml-4 flex items-center gap-3">
              <StatusBadge status={issue.status} />
              <svg
                className={`w-4 h-4 text-gray-500 transition-transform ${expanded === issue.issue_id ? "rotate-180" : ""}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </button>

          {/* Expanded: flagged traces */}
          {expanded === issue.issue_id && (
            <div className="border-t border-gray-800 px-6 py-4">
              {/* Metrics row */}
              <div className="grid grid-cols-5 gap-4 mb-4">
                <MiniStat label="Accuracy" value={`${((issue.metrics.accuracy ?? 0) * 100).toFixed(1)}%`} />
                <MiniStat label="Precision" value={`${((issue.metrics.precision ?? 0) * 100).toFixed(1)}%`} />
                <MiniStat label="Recall" value={`${((issue.metrics.recall ?? 0) * 100).toFixed(1)}%`} />
                <MiniStat label="F1 Score" value={`${((issue.metrics.f1 ?? 0) * 100).toFixed(1)}%`} />
                <MiniStat label="Inference" value={`${(issue.metrics.inference_time_per_trace_ms ?? 0).toFixed(1)}ms`} />
              </div>

              {/* Flagged traces table */}
              <div className="text-xs text-gray-500 uppercase tracking-wide mb-2">Flagged traces</div>
              {loadingFlags ? (
                <div className="text-sm text-gray-500 py-4">Loading...</div>
              ) : (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {flags.map((f) => (
                    <button
                      key={f.trace_id}
                      onClick={() => onSelectTrace(f.trace_id)}
                      className="w-full text-left bg-gray-950 border border-gray-800 rounded-md px-4 py-3 hover:border-gray-600 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-mono text-gray-500">{f.trace_id.slice(0, 12)}...</span>
                        <div className="flex items-center gap-2">
                          <ConfidenceBadge value={f.confidence} />
                          {f.failure_type !== "unknown" && (
                            <span className="text-xs px-2 py-0.5 bg-gray-800 rounded text-gray-400">{f.failure_type}</span>
                          )}
                        </div>
                      </div>
                      <div className="text-sm text-gray-300 truncate">{f.user_query}</div>
                      <div className="text-xs text-gray-500 mt-1 truncate">{f.agent_response}</div>
                      <div className="text-xs text-gray-600 mt-1">
                        Tools: {f.tools_called.length > 0 ? f.tools_called.join(", ") : "None"}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    active: "bg-green-900/50 text-green-400 border-green-800",
    training: "bg-blue-900/50 text-blue-400 border-blue-800",
    labeling: "bg-yellow-900/50 text-yellow-400 border-yellow-800",
    pending: "bg-gray-800 text-gray-400 border-gray-700",
  };
  return <span className={`text-xs px-2 py-0.5 rounded border ${colors[status] || colors.pending}`}>{status}</span>;
}

function ConfidenceBadge({ value }: { value: number }) {
  const color = value > 0.8 ? "text-red-400" : value > 0.6 ? "text-yellow-400" : "text-gray-400";
  return <span className={`text-xs font-mono ${color}`}>{(value * 100).toFixed(0)}%</span>;
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-gray-600">{label}</div>
      <div className="text-sm font-medium">{value}</div>
    </div>
  );
}

