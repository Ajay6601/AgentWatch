"""
Issue Detector: orchestrates the full pipeline.

Natural language description → LLM judge labels → distilled classifier → bulk classification

This is the "one click" pipeline that maps to Raindrop's Deep Search:
user types "agent is being lazy", system returns frequency, affected users, and traces.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from agentwatch.classifier.few_shot_trainer import (
    TrainingResult,
    TrainedClassifier,
    train_all_approaches,
    train_embedding_lr,
    train_prototype,
    train_setfit,
)
from agentwatch.classifier.llm_judge import JudgeResult, compute_judge_cost, judge_batch
from agentwatch.traces.models import Trace


@dataclass
class Issue:
    """A tracked issue — the materialized view for a natural language query."""

    issue_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending → labeling → training → active
    # Labeling results
    judge_results: list[JudgeResult] = field(default_factory=list)
    n_positive: int = 0
    n_negative: int = 0
    # Trained classifiers
    classifiers: dict[str, TrainedClassifier] = field(default_factory=dict)
    best_approach: str = ""
    # Classification results on full dataset
    flagged_traces: list[tuple[str, float]] = field(default_factory=list)  # (trace_id, confidence)
    total_classified: int = 0
    frequency: float = 0.0  # fraction of traces with this issue
    affected_users: set[str] = field(default_factory=set)


@dataclass
class PipelineProgress:
    """Progress updates during pipeline execution."""

    stage: str
    message: str
    progress: float  # 0.0-1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class IssueDetector:
    """
    Orchestrates the full detection pipeline:
    1. Accept natural language issue description
    2. Sample traces and send to LLM judge for labeling
    3. Train few-shot classifiers on the labels
    4. Run the best classifier on all traces
    5. Return frequency, affected users, and flagged traces
    """

    def __init__(self):
        self.issues: dict[str, Issue] = {}

    async def create_issue(
        self,
        description: str,
        all_traces: list[Trace],
        labeling_sample_size: int = 80,
        approach: str = "all",  # "setfit", "embedding_lr", "prototype", or "all"
        on_progress: callable | None = None,
    ) -> Issue:
        """
        Full pipeline: description → labels → classifier → bulk classification.
        """
        issue = Issue(description=description)
        self.issues[issue.issue_id] = issue

        # --- Stage 1: Sample traces for labeling ---
        issue.status = "labeling"
        if on_progress:
            on_progress(PipelineProgress("labeling", "Sending traces to LLM judge...", 0.1))

        # Sample a balanced subset for labeling
        sample_size = min(labeling_sample_size, len(all_traces))
        import random

        sample = random.sample(all_traces, sample_size)

        # Run LLM judge
        issue.judge_results = await judge_batch(sample, description)

        # Filter out low-confidence labels
        confident_results = [r for r in issue.judge_results if r.confidence >= 0.6]
        issue.n_positive = sum(1 for r in confident_results if r.label)
        issue.n_negative = sum(1 for r in confident_results if not r.label)

        if on_progress:
            on_progress(
                PipelineProgress(
                    "labeling",
                    f"Labeled {len(confident_results)} traces: "
                    f"{issue.n_positive} positive, {issue.n_negative} negative",
                    0.4,
                )
            )

        # Need minimum examples for training
        if issue.n_positive < 3 or issue.n_negative < 3:
            issue.status = "insufficient_data"
            return issue

        # --- Stage 2: Train classifiers ---
        issue.status = "training"
        if on_progress:
            on_progress(PipelineProgress("training", "Training few-shot classifiers...", 0.5))

        # Prepare training data
        trace_map = {t.trace_id: t for t in sample}
        texts, labels = [], []
        for r in confident_results:
            if r.trace_id in trace_map:
                texts.append(trace_map[r.trace_id].to_text())
                labels.append(r.label)

        if approach == "all":
            issue.classifiers = train_all_approaches(texts, labels, issue.issue_id)
            # Pick best by F1
            best = max(issue.classifiers.values(), key=lambda c: c.metrics.f1 if c.metrics else 0)
            issue.best_approach = best.approach
        else:
            fn_map = {"setfit": train_setfit, "embedding_lr": train_embedding_lr, "prototype": train_prototype}
            clf = fn_map[approach](texts, labels, issue.issue_id)
            issue.classifiers[approach] = clf
            issue.best_approach = approach

        if on_progress:
            best_clf = issue.classifiers[issue.best_approach]
            m = best_clf.metrics
            on_progress(
                PipelineProgress(
                    "training",
                    f"Best approach: {issue.best_approach} " f"(F1={m.f1:.2f}, accuracy={m.accuracy:.2f})",
                    0.7,
                )
            )

        # --- Stage 3: Classify all traces ---
        issue.status = "classifying"
        if on_progress:
            on_progress(PipelineProgress("classifying", f"Running classifier on {len(all_traces)} traces...", 0.8))

        best_clf = issue.classifiers[issue.best_approach]
        all_texts = [t.to_text() for t in all_traces]

        # Batch classification
        t0 = time.time()
        predictions = best_clf.predict(all_texts)
        classify_time = (time.time() - t0) * 1000

        for trace, (label, conf) in zip(all_traces, predictions):
            if label:
                issue.flagged_traces.append((trace.trace_id, conf))
                if trace.user_id:
                    issue.affected_users.add(trace.user_id)

        issue.total_classified = len(all_traces)
        issue.frequency = len(issue.flagged_traces) / max(len(all_traces), 1)

        issue.status = "active"
        if on_progress:
            on_progress(
                PipelineProgress(
                    "complete",
                    f"Found {len(issue.flagged_traces)}/{len(all_traces)} traces "
                    f"({issue.frequency:.1%}) affecting {len(issue.affected_users)} users. "
                    f"Classification took {classify_time:.0f}ms total "
                    f"({classify_time/max(len(all_traces),1):.2f}ms/trace).",
                    1.0,
                )
            )

        return issue

    def get_issue(self, issue_id: str) -> Issue | None:
        return self.issues.get(issue_id)

    def list_issues(self) -> list[Issue]:
        return list(self.issues.values())

