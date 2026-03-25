"""
OpenTelemetry-compatible trace data models.

Follows OTEL span conventions so traces from real agent frameworks
(Vercel AI SDK, LangChain, etc.) can be ingested directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SpanKind(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    AGENT_STEP = "agent_step"
    USER_MESSAGE = "user_message"
    AGENT_RESPONSE = "agent_response"


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class Span(BaseModel):
    """Single span in a trace — maps to an OTEL span."""

    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: str | None = None
    name: str
    kind: SpanKind
    status: SpanStatus = SpanStatus.OK
    start_time: datetime
    end_time: datetime
    attributes: dict = Field(default_factory=dict)
    # LLM-specific
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    # Tool-specific
    tool_name: str | None = None
    tool_input: str | None = None
    tool_output: str | None = None
    # Content
    content: str | None = None

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time).total_seconds() * 1000


class Trace(BaseModel):
    """
    Complete agent trace — one user request through to final response.
    Compatible with OTEL trace format: trace_id + ordered spans.
    """

    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:32])
    session_id: str | None = None
    user_id: str | None = None
    agent_id: str = "support-agent-v1"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    spans: list[Span] = Field(default_factory=list)
    # Computed fields (set after trace completes)
    user_query: str = ""
    agent_response: str = ""
    total_tokens: int = 0
    total_duration_ms: float = 0
    tools_called: list[str] = Field(default_factory=list)
    error: str | None = None
    # Metadata
    metadata: dict = Field(default_factory=dict)

    def finalize(self) -> None:
        """Compute summary fields from spans."""
        for s in self.spans:
            if s.kind == SpanKind.USER_MESSAGE:
                self.user_query = s.content or ""
            if s.kind == SpanKind.AGENT_RESPONSE:
                self.agent_response = s.content or ""
            if s.kind == SpanKind.TOOL_CALL and s.tool_name:
                self.tools_called.append(s.tool_name)
            if s.input_tokens:
                self.total_tokens += s.input_tokens
            if s.output_tokens:
                self.total_tokens += s.output_tokens
        if self.spans:
            self.total_duration_ms = (self.spans[-1].end_time - self.spans[0].start_time).total_seconds() * 1000

    def to_text(self) -> str:
        """Serialize trace to text for embedding/classification."""
        parts = [f"Query: {self.user_query}"]
        for s in self.spans:
            if s.kind == SpanKind.TOOL_CALL:
                parts.append(f"Tool: {s.tool_name} → {(s.tool_output or '')[:200]}")
            elif s.kind == SpanKind.LLM_CALL:
                parts.append(f"LLM ({s.model}): {(s.content or '')[:200]}")
        parts.append(f"Response: {self.agent_response[:300]}")
        return "\n".join(parts)


class TraceWithLabel(BaseModel):
    """Trace with an assigned issue label and confidence score."""

    trace: Trace
    issue_id: str
    label: bool  # True = issue present
    confidence: float
    source: str = "llm_judge"  # or "classifier"

