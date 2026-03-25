"use client";

import { useEffect, useState } from "react";

import { SpanDetail, TraceDetail, api } from "../../lib/api";

interface Props {
  traceId: string;
  onClose: () => void;
}

export function TraceViewer({ traceId, onClose }: Props) {
  const [trace, setTrace] = useState<TraceDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getTrace(traceId)
      .then(setTrace)
      .catch(() => setTrace(null))
      .finally(() => setLoading(false));
  }, [traceId]);

  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-6">
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-3xl max-h-[85vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800">
          <div>
            <h3 className="text-sm font-medium">Trace detail</h3>
            <span className="text-xs font-mono text-gray-500">{traceId}</span>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300 transition-colors">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {loading ? (
            <div className="text-sm text-gray-500 py-8 text-center">Loading trace...</div>
          ) : !trace ? (
            <div className="text-sm text-red-400 py-8 text-center">Trace not found</div>
          ) : (
            <div className="space-y-4">
              {/* Summary */}
              <div className="grid grid-cols-3 gap-4">
                <SummaryItem label="Tokens" value={trace.total_tokens.toLocaleString()} />
                <SummaryItem label="Duration" value={`${Math.round(trace.total_duration_ms)}ms`} />
                <SummaryItem
                  label="Failure type"
                  value={String(trace.metadata?.failure_type ?? "unknown")}
                  highlight={trace.metadata?.failure_type !== "none"}
                />
              </div>

              {/* Spans */}
              <div className="text-xs text-gray-500 uppercase tracking-wide mt-6 mb-2">Execution trace</div>
              <div className="space-y-2">
                {trace.spans.map((span, i) => (
                  <SpanCard key={span.span_id || i} span={span} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SummaryItem({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div>
      <div className="text-xs text-gray-600">{label}</div>
      <div className={`text-sm font-medium ${highlight ? "text-red-400" : ""}`}>{value}</div>
    </div>
  );
}

function SpanCard({ span }: { span: SpanDetail }) {
  const kindColors: Record<string, string> = {
    user_message: "border-l-blue-500",
    agent_response: "border-l-green-500",
    tool_call: "border-l-amber-500",
    llm_call: "border-l-purple-500",
    retrieval: "border-l-teal-500",
    agent_step: "border-l-gray-500",
  };

  const borderColor = kindColors[span.kind] || "border-l-gray-600";

  return (
    <div className={`bg-gray-950 border border-gray-800 border-l-2 ${borderColor} rounded-md px-4 py-3`}>
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-gray-500">{span.kind}</span>
          {span.tool_name && (
            <span className="text-xs px-1.5 py-0.5 bg-amber-900/30 text-amber-400 rounded">{span.tool_name}</span>
          )}
          {span.model && (
            <span className="text-xs px-1.5 py-0.5 bg-purple-900/30 text-purple-400 rounded">{span.model}</span>
          )}
        </div>
        {span.input_tokens != null && (
          <span className="text-xs text-gray-600">
            {span.input_tokens}+{span.output_tokens} tokens
          </span>
        )}
      </div>

      {span.content && (
        <div className="text-sm text-gray-300 mt-1 whitespace-pre-wrap break-words">
          {span.content.length > 300 ? span.content.slice(0, 300) + "..." : span.content}
        </div>
      )}

      {span.tool_input && (
        <div className="text-xs text-gray-500 mt-1">
          Input: {span.tool_input.length > 150 ? span.tool_input.slice(0, 150) + "..." : span.tool_input}
        </div>
      )}
      {span.tool_output && (
        <div className="text-xs text-gray-500 mt-1">
          Output: {span.tool_output.length > 150 ? span.tool_output.slice(0, 150) + "..." : span.tool_output}
        </div>
      )}
    </div>
  );
}

