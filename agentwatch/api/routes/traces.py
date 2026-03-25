"""Trace ingestion and query routes."""

from fastapi import APIRouter, HTTPException, Query

from agentwatch.api.routes.issues import trace_store
from agentwatch.api.schemas import TraceIngestRequest, TraceResponse

router = APIRouter()


@router.get("/", response_model=list[TraceResponse])
async def list_traces(
    limit: int = Query(50, le=200),
    offset: int = 0,
    user_id: str | None = None,
    failure_type: str | None = None,
):
    """List traces with optional filters."""
    filtered = trace_store
    if user_id:
        filtered = [t for t in filtered if t.user_id == user_id]
    if failure_type:
        filtered = [t for t in filtered if t.metadata.get("failure_type") == failure_type]

    page = filtered[offset : offset + limit]
    return [
        TraceResponse(
            trace_id=t.trace_id,
            user_id=t.user_id,
            user_query=t.user_query,
            agent_response=t.agent_response,
            total_tokens=t.total_tokens,
            total_duration_ms=t.total_duration_ms,
            tools_called=t.tools_called,
            spans=[s.model_dump() for s in t.spans],
            metadata=t.metadata,
            timestamp=t.timestamp,
        )
        for t in page
    ]


@router.get("/stats")
async def trace_stats():
    """Summary stats for all traces."""
    total = len(trace_store)
    failure_counts = {}
    user_set = set()
    for t in trace_store:
        ft = t.metadata.get("failure_type", "unknown")
        failure_counts[ft] = failure_counts.get(ft, 0) + 1
        if t.user_id:
            user_set.add(t.user_id)

    return {
        "total_traces": total,
        "unique_users": len(user_set),
        "failure_distribution": failure_counts,
        "avg_tokens": sum(t.total_tokens for t in trace_store) / max(total, 1),
        "avg_duration_ms": sum(t.total_duration_ms for t in trace_store) / max(total, 1),
    }


@router.get("/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str):
    """Get a single trace by ID with full span details."""
    for t in trace_store:
        if t.trace_id == trace_id:
            return TraceResponse(
                trace_id=t.trace_id,
                user_id=t.user_id,
                user_query=t.user_query,
                agent_response=t.agent_response,
                total_tokens=t.total_tokens,
                total_duration_ms=t.total_duration_ms,
                tools_called=t.tools_called,
                spans=[s.model_dump() for s in t.spans],
                metadata=t.metadata,
                timestamp=t.timestamp,
            )
    raise HTTPException(404, "Trace not found")

