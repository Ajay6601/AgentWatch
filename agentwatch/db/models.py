from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func

from agentwatch.db.database import Base


class TraceRecord(Base):
    __tablename__ = "traces"

    trace_id = Column(String(64), primary_key=True)
    session_id = Column(String(64), nullable=True, index=True)
    user_id = Column(String(64), nullable=True, index=True)
    agent_id = Column(String(64), default="support-agent-v1", index=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)
    user_query = Column(Text, default="")
    agent_response = Column(Text, default="")
    total_tokens = Column(Integer, default=0)
    total_duration_ms = Column(Float, default=0)
    tools_called = Column(JSON, default=list)
    spans = Column(JSON, default=list)  # Full span data as JSONB
    metadata_ = Column("metadata", JSON, default=dict)
    trace_text = Column(Text, default="")  # Serialized for embedding
    embedding = Column(JSON, nullable=True)  # Stored as list[float]
    created_at = Column(DateTime, server_default=func.now())


class IssueRecord(Base):
    __tablename__ = "issues"

    issue_id = Column(String(32), primary_key=True)
    description = Column(Text, nullable=False)
    status = Column(String(32), default="pending")
    best_approach = Column(String(32), default="")
    n_positive = Column(Integer, default=0)
    n_negative = Column(Integer, default=0)
    total_classified = Column(Integer, default=0)
    n_flagged = Column(Integer, default=0)
    frequency = Column(Float, default=0.0)
    n_affected_users = Column(Integer, default=0)
    metrics = Column(JSON, default=dict)  # TrainingResult as dict
    created_at = Column(DateTime, server_default=func.now())


class IssueFlagRecord(Base):
    __tablename__ = "issue_flags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    issue_id = Column(String(32), index=True)
    trace_id = Column(String(64), index=True)
    confidence = Column(Float, default=0.0)
    source = Column(String(32), default="classifier")  # "llm_judge" or "classifier"

