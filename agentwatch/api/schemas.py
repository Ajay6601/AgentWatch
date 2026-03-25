"""Pydantic schemas for API request/response models."""

from datetime import datetime

from pydantic import BaseModel


class CreateIssueRequest(BaseModel):
    description: str
    approach: str = "all"  # "setfit", "embedding_lr", "prototype", "all"
    labeling_sample_size: int = 80


class IssueResponse(BaseModel):
    issue_id: str
    description: str
    status: str
    best_approach: str
    n_positive: int
    n_negative: int
    total_classified: int
    n_flagged: int
    frequency: float
    n_affected_users: int
    metrics: dict
    created_at: datetime | None = None


class IssueFlagResponse(BaseModel):
    trace_id: str
    confidence: float
    user_query: str = ""
    agent_response: str = ""
    tools_called: list[str] = []
    failure_type: str = ""


class TraceResponse(BaseModel):
    trace_id: str
    user_id: str | None
    user_query: str
    agent_response: str
    total_tokens: int
    total_duration_ms: float
    tools_called: list[str]
    spans: list[dict]
    metadata: dict
    timestamp: datetime | None = None


class TraceIngestRequest(BaseModel):
    trace_id: str
    session_id: str | None = None
    user_id: str | None = None
    agent_id: str = "default"
    user_query: str
    agent_response: str
    total_tokens: int = 0
    total_duration_ms: float = 0
    tools_called: list[str] = []
    spans: list[dict] = []
    metadata: dict = {}


class DiscoveryResponse(BaseModel):
    n_traces: int
    n_clusters: int
    n_noise: int
    clusters: list[dict]


class PipelineProgressResponse(BaseModel):
    stage: str
    message: str
    progress: float


class BenchmarkResponse(BaseModel):
    issue_id: str
    results: list[dict]
    comparison_table: str

