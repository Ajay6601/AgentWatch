"use client";

import { useState } from "react";

import { Issue, api } from "../../lib/api";

interface Props {
  onCreated: (issue: Issue) => void;
}

const PRESETS = [
  {
    label: "Laziness",
    description:
      "Agent is being lazy — giving short, generic responses like 'check the docs' without actually searching the knowledge base or using any available tools.",
  },
  {
    label: "Forgetting",
    description:
      "Agent is forgetting context — asking for information the user already provided earlier in the conversation, or ignoring prior messages.",
  },
  {
    label: "Hallucination",
    description:
      "Agent is hallucinating — confidently stating incorrect information, citing features that don't exist, or making up policy details.",
  },
  {
    label: "Tool loop",
    description:
      "Agent is stuck in a loop — calling the same tool repeatedly with similar inputs without making progress toward answering the user's question.",
  },
];

export function IssueCreator({ onCreated }: Props) {
  const [description, setDescription] = useState("");
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState("");

  const handleCreate = async () => {
    if (!description.trim()) return;
    setLoading(true);
    setStage("Sending traces to LLM judge...");

    try {
      // Simulate stage updates (the actual stages happen server-side)
      const stageTimer = setInterval(() => {
        setStage((prev) => {
          if (prev.includes("judge")) return "Training few-shot classifiers...";
          if (prev.includes("Training")) return "Classifying all traces...";
          return prev;
        });
      }, 3000);

      const issue = await api.createIssue(description);
      clearInterval(stageTimer);
      onCreated(issue);
      setDescription("");
      setStage("");
    } catch (e) {
      setStage(`Error: ${e instanceof Error ? e.message : "Unknown error"}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 mb-8">
      <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wide mb-4">Track a new issue</h2>

      {/* Presets */}
      <div className="flex gap-2 mb-4 flex-wrap">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            onClick={() => setDescription(p.description)}
            className="px-3 py-1.5 text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-md transition-colors"
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="flex gap-3">
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe the issue in natural language... e.g. 'agent is giving vague answers without using tools'"
          className="flex-1 bg-gray-950 border border-gray-700 rounded-lg px-4 py-3 text-sm text-gray-100 placeholder-gray-600 resize-none focus:outline-none focus:border-blue-600 transition-colors"
          rows={2}
          disabled={loading}
        />
        <button
          onClick={handleCreate}
          disabled={loading || !description.trim()}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium rounded-lg transition-colors self-end"
        >
          {loading ? "Training..." : "Track issue"}
        </button>
      </div>

      {/* Progress */}
      {loading && stage && (
        <div className="mt-3 flex items-center gap-2 text-xs text-gray-400">
          <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          {stage}
        </div>
      )}
    </div>
  );
}

