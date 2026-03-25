"""Unknown unknowns discovery routes."""

from fastapi import APIRouter, HTTPException

from agentwatch.api.routes.issues import detector, trace_store
from agentwatch.api.schemas import DiscoveryResponse
from agentwatch.discovery.cluster import discover_unknown_issues, label_discovered_clusters

router = APIRouter()


@router.post("/run", response_model=DiscoveryResponse)
async def run_discovery(min_cluster_size: int = 5):
    """
    Run unsupervised discovery to find unknown failure patterns.
    Clusters traces and suggests labels for each cluster.
    """
    if not trace_store:
        raise HTTPException(400, "No traces loaded.")

    # Collect trace IDs already flagged by known issues
    known_ids = set()
    for issue in detector.list_issues():
        for tid, _ in issue.flagged_traces:
            known_ids.add(tid)

    result = discover_unknown_issues(
        trace_store,
        known_issue_trace_ids=known_ids,
        min_cluster_size=min_cluster_size,
    )

    # Auto-label clusters
    if result.clusters:
        result.clusters = await label_discovered_clusters(result.clusters)

    return DiscoveryResponse(
        n_traces=result.n_traces,
        n_clusters=result.n_clusters,
        n_noise=result.n_noise,
        clusters=[
            {
                "cluster_id": c.cluster_id,
                "size": c.size,
                "percentage": round(c.percentage, 4),
                "suggested_label": c.suggested_label,
                "avg_confidence": round(c.avg_confidence, 3),
                "representative_texts": c.representative_texts,
            }
            for c in result.clusters
        ],
    )

