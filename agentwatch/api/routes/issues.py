"""Issue creation, listing, and classification routes."""

from fastapi import APIRouter, HTTPException

from agentwatch.api.schemas import BenchmarkResponse, CreateIssueRequest, IssueFlagResponse, IssueResponse
from agentwatch.classifier.benchmarks import benchmark_classifier, compare_with_llm_judge
from agentwatch.classifier.issue_detector import IssueDetector
from agentwatch.traces.models import Trace

router = APIRouter()

# In-memory store (replace with DB in production)
detector = IssueDetector()
trace_store: list[Trace] = []


def set_trace_store(traces: list[Trace]):
    """Set the trace store — called during startup/data generation."""
    global trace_store
    trace_store = traces


@router.post("/", response_model=IssueResponse)
async def create_issue(req: CreateIssueRequest):
    """
    Create a new tracked issue from a natural language description.
    Runs the full pipeline: LLM judge → train classifier → classify all traces.
    """
    if not trace_store:
        raise HTTPException(400, "No traces loaded. Run trace generation first.")

    progress_log = []

    def on_progress(p):
        progress_log.append({"stage": p.stage, "message": p.message, "progress": p.progress})
        print(f"  [{p.stage}] {p.message}")

    issue = await detector.create_issue(
        description=req.description,
        all_traces=trace_store,
        labeling_sample_size=req.labeling_sample_size,
        approach=req.approach,
        on_progress=on_progress,
    )

    best_clf = issue.classifiers.get(issue.best_approach)
    metrics = {}
    if best_clf and best_clf.metrics:
        m = best_clf.metrics
        metrics = {
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "training_time_ms": m.training_time_ms,
            "inference_time_per_trace_ms": m.inference_time_per_trace_ms,
            "device": m.device,
        }

    return IssueResponse(
        issue_id=issue.issue_id,
        description=issue.description,
        status=issue.status,
        best_approach=issue.best_approach,
        n_positive=issue.n_positive,
        n_negative=issue.n_negative,
        total_classified=issue.total_classified,
        n_flagged=len(issue.flagged_traces),
        frequency=issue.frequency,
        n_affected_users=len(issue.affected_users),
        metrics=metrics,
        created_at=issue.created_at,
    )


@router.get("/", response_model=list[IssueResponse])
async def list_issues():
    """List all tracked issues."""
    results = []
    for issue in detector.list_issues():
        best_clf = issue.classifiers.get(issue.best_approach)
        metrics = {}
        if best_clf and best_clf.metrics:
            m = best_clf.metrics
            metrics = {
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
            }
        results.append(
            IssueResponse(
                issue_id=issue.issue_id,
                description=issue.description,
                status=issue.status,
                best_approach=issue.best_approach,
                n_positive=issue.n_positive,
                n_negative=issue.n_negative,
                total_classified=issue.total_classified,
                n_flagged=len(issue.flagged_traces),
                frequency=issue.frequency,
                n_affected_users=len(issue.affected_users),
                metrics=metrics,
                created_at=issue.created_at,
            )
        )
    return results


@router.get("/{issue_id}", response_model=IssueResponse)
async def get_issue(issue_id: str):
    issue = detector.get_issue(issue_id)
    if not issue:
        raise HTTPException(404, "Issue not found")
    best_clf = issue.classifiers.get(issue.best_approach)
    metrics = {}
    if best_clf and best_clf.metrics:
        m = best_clf.metrics
        metrics = {"accuracy": m.accuracy, "precision": m.precision, "recall": m.recall, "f1": m.f1}
    return IssueResponse(
        issue_id=issue.issue_id,
        description=issue.description,
        status=issue.status,
        best_approach=issue.best_approach,
        n_positive=issue.n_positive,
        n_negative=issue.n_negative,
        total_classified=issue.total_classified,
        n_flagged=len(issue.flagged_traces),
        frequency=issue.frequency,
        n_affected_users=len(issue.affected_users),
        metrics=metrics,
        created_at=issue.created_at,
    )


@router.get("/{issue_id}/flags", response_model=list[IssueFlagResponse])
async def get_issue_flags(issue_id: str, limit: int = 50):
    """Get traces flagged for a specific issue."""
    issue = detector.get_issue(issue_id)
    if not issue:
        raise HTTPException(404, "Issue not found")

    trace_map = {t.trace_id: t for t in trace_store}
    flags = []
    for trace_id, conf in issue.flagged_traces[:limit]:
        t = trace_map.get(trace_id)
        if t:
            flags.append(
                IssueFlagResponse(
                    trace_id=trace_id,
                    confidence=conf,
                    user_query=t.user_query,
                    agent_response=t.agent_response,
                    tools_called=t.tools_called,
                    failure_type=t.metadata.get("failure_type", "unknown"),
                )
            )
    return flags


@router.get("/{issue_id}/benchmark", response_model=BenchmarkResponse)
async def benchmark_issue(issue_id: str):
    """Run full benchmark on all trained classifiers for an issue."""
    issue = detector.get_issue(issue_id)
    if not issue:
        raise HTTPException(404, "Issue not found")
    if not issue.classifiers:
        raise HTTPException(400, "No trained classifiers for this issue")

    # Use judge results as ground truth for benchmarking
    trace_map = {t.trace_id: t for t in trace_store}
    test_texts, test_labels = [], []
    for jr in issue.judge_results:
        if jr.confidence >= 0.6 and jr.trace_id in trace_map:
            test_texts.append(trace_map[jr.trace_id].to_text())
            test_labels.append(jr.label)

    bench_results = []
    for name, clf in issue.classifiers.items():
        b = benchmark_classifier(clf, test_texts, test_labels)
        bench_results.append(b)

    comparison = compare_with_llm_judge(bench_results)

    return BenchmarkResponse(
        issue_id=issue_id,
        results=[
            {
                "approach": b.approach,
                "f1": b.f1,
                "accuracy": b.accuracy,
                "ms_per_trace": b.avg_inference_ms_per_trace,
                "throughput": b.throughput_traces_per_sec,
                "cost_per_1m": b.cost_per_1m_traces_usd,
            }
            for b in bench_results
        ],
        comparison_table=comparison,
    )

