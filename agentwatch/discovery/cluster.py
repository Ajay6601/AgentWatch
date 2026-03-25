"""
Unknown Unknowns Discovery: find failure patterns nobody asked about.

This maps to Raindrop's "discovering unknown unknowns" — automatically
surface clusters of similar agent behavior that don't match any tracked issue.

Uses HDBSCAN for density-based clustering on trace embeddings, then
summarizes each cluster to suggest new issues to track.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import hdbscan
import numpy as np
from sentence_transformers import SentenceTransformer

from agentwatch.config import get_settings
from agentwatch.traces.models import Trace


@dataclass
class DiscoveredCluster:
    """A cluster of similar traces that might represent an undiscovered issue."""

    cluster_id: int
    size: int
    percentage: float
    representative_traces: list[str]  # trace_ids
    representative_texts: list[str]  # trace texts for display
    suggested_label: str = ""
    avg_confidence: float = 0.0


@dataclass
class DiscoveryResult:
    """Results from unsupervised discovery."""

    n_traces: int
    n_clusters: int
    n_noise: int
    clusters: list[DiscoveredCluster]


def discover_unknown_issues(
    traces: list[Trace],
    known_issue_trace_ids: set[str] | None = None,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> DiscoveryResult:
    """
    Cluster traces to find unknown failure patterns.

    1. Embed all traces with sentence-transformer
    2. Run HDBSCAN (density-based, doesn't need k)
    3. Filter out traces already assigned to known issues
    4. Return clusters ranked by size
    """
    settings = get_settings()
    encoder = SentenceTransformer(settings.embedding_model, device=settings.device)

    # Optionally filter out traces with known issues
    if known_issue_trace_ids:
        traces = [t for t in traces if t.trace_id not in known_issue_trace_ids]

    if len(traces) < min_cluster_size:
        return DiscoveryResult(n_traces=len(traces), n_clusters=0, n_noise=len(traces), clusters=[])

    # Embed
    texts = [t.to_text() for t in traces]
    embeddings = encoder.encode(texts, show_progress_bar=False, batch_size=64)

    # Cluster
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)
    probabilities = clusterer.probabilities_

    # Build cluster summaries
    cluster_ids = set(labels)
    cluster_ids.discard(-1)  # noise

    clusters = []
    for cid in sorted(cluster_ids):
        indices = [i for i, l in enumerate(labels) if l == cid]
        size = len(indices)

        # Get representative traces (highest cluster membership probability)
        idx_probs = [(i, probabilities[i]) for i in indices]
        idx_probs.sort(key=lambda x: x[1], reverse=True)
        top_indices = [ip[0] for ip in idx_probs[:3]]

        clusters.append(
            DiscoveredCluster(
                cluster_id=int(cid),
                size=size,
                percentage=size / len(traces),
                representative_traces=[traces[i].trace_id for i in top_indices],
                representative_texts=[texts[i][:300] for i in top_indices],
                avg_confidence=float(np.mean([probabilities[i] for i in indices])),
            )
        )

    # Sort by size descending
    clusters.sort(key=lambda c: c.size, reverse=True)

    n_noise = sum(1 for l in labels if l == -1)

    return DiscoveryResult(
        n_traces=len(traces),
        n_clusters=len(clusters),
        n_noise=n_noise,
        clusters=clusters,
    )


async def label_discovered_clusters(
    clusters: list[DiscoveredCluster],
) -> list[DiscoveredCluster]:
    """
    Use LLM to generate suggested labels for discovered clusters.
    This gives the user a starting point: "Cluster 3 looks like it might be
    'agent stuck in tool loop' — want to track it?"
    """
    import json

    from openai import AsyncOpenAI

    from agentwatch.config import get_settings

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    for cluster in clusters:
        sample_text = "\n---\n".join(cluster.representative_texts[:3])
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=100,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You analyze AI agent traces. Given sample traces from a cluster, "
                            "suggest a short label describing the common pattern or issue. "
                            'Respond with JSON: {"label": "short description", "severity": "low/medium/high"}'
                        ),
                    },
                    {"role": "user", "content": f"Sample traces from cluster:\n{sample_text}"},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(raw)
            cluster.suggested_label = parsed.get("label", f"Cluster {cluster.cluster_id}")
        except Exception:
            cluster.suggested_label = f"Unnamed pattern (cluster {cluster.cluster_id})"

    return clusters

