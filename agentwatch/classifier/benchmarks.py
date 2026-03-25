"""
Benchmarks: prove the distillation is worth it.

Compares LLM judge (expensive, slow) vs distilled classifier (cheap, fast)
across cost, latency, and accuracy. This is the money table for the demo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from agentwatch.classifier.few_shot_trainer import TrainedClassifier
from agentwatch.classifier.llm_judge import compute_judge_cost


@dataclass
class BenchmarkResult:
    approach: str
    # Accuracy (agreement with LLM judge)
    accuracy: float
    precision: float
    recall: float
    f1: float
    # Latency
    avg_inference_ms_per_trace: float
    throughput_traces_per_sec: float
    # Cost projections
    cost_per_1m_traces_usd: float
    # Scale projections
    traces_per_day_single_gpu: float
    gpus_needed_for_100m_per_day: float
    # Device
    device: str


def benchmark_classifier(
    classifier: TrainedClassifier,
    test_texts: list[str],
    test_labels: list[bool],
    n_warmup: int = 5,
    n_runs: int = 3,
) -> BenchmarkResult:
    """Run full benchmark on a trained classifier."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Warmup
    for _ in range(n_warmup):
        classifier.predict(test_texts[:10])

    # Latency measurement
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        preds = classifier.predict(test_texts)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    avg_total_ms = sum(latencies) / len(latencies)
    avg_per_trace_ms = avg_total_ms / max(len(test_texts), 1)
    throughput = 1000 / avg_per_trace_ms if avg_per_trace_ms > 0 else 0

    # Accuracy
    pred_labels = [p[0] for p in preds]
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    acc = accuracy_score(test_labels, pred_labels)
    prec = precision_score(test_labels, pred_labels, zero_division=0)
    rec = recall_score(test_labels, pred_labels, zero_division=0)
    f1 = f1_score(test_labels, pred_labels, zero_division=0)

    # Cost projection
    # Classifier cost = compute only (GPU time)
    # Assume $0.50/hr for a T4 GPU, $2.50/hr for an A100
    gpu_cost_per_hour = 0.50 if device == "cpu" else 2.50
    traces_per_hour = throughput * 3600
    cost_per_1m = (1_000_000 / max(traces_per_hour, 1)) * gpu_cost_per_hour

    # Scale projections
    traces_per_day = throughput * 86400
    gpus_for_100m = 100_000_000 / max(traces_per_day, 1)

    return BenchmarkResult(
        approach=classifier.approach,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        avg_inference_ms_per_trace=avg_per_trace_ms,
        throughput_traces_per_sec=throughput,
        cost_per_1m_traces_usd=cost_per_1m,
        traces_per_day_single_gpu=traces_per_day,
        gpus_needed_for_100m_per_day=gpus_for_100m,
        device=device,
    )


def compare_with_llm_judge(
    classifier_benchmarks: list[BenchmarkResult],
) -> str:
    """Generate a comparison table: distilled classifiers vs LLM judge."""
    judge_cost = compute_judge_cost(1_000_000)

    lines = [
        "=" * 80,
        "DISTILLATION BENCHMARK: LLM Judge vs Few-Shot Classifiers",
        "=" * 80,
        "",
        f"{'Approach':<20} {'F1':>6} {'Acc':>6} {'ms/trace':>10} "
        f"{'traces/sec':>12} {'$/1M traces':>12} {'Speedup':>10}",
        "-" * 80,
    ]

    # LLM Judge baseline
    judge_ms = 500  # ~500ms per API call
    judge_throughput = 1000 / judge_ms
    lines.append(
        f"{'LLM Judge':<20} {'1.00':>6} {'1.00':>6} {judge_ms:>10.1f} "
        f"{judge_throughput:>12.1f} {judge_cost['cost_per_1m_traces_usd']:>12.2f} {'1.0x':>10}"
    )

    for b in classifier_benchmarks:
        speedup = judge_ms / max(b.avg_inference_ms_per_trace, 0.001)
        cost_ratio = judge_cost["cost_per_1m_traces_usd"] / max(b.cost_per_1m_traces_usd, 0.001)
        lines.append(
            f"{b.approach:<20} {b.f1:>6.2f} {b.accuracy:>6.2f} "
            f"{b.avg_inference_ms_per_trace:>10.2f} {b.throughput_traces_per_sec:>12.1f} "
            f"{b.cost_per_1m_traces_usd:>12.2f} {speedup:>9.0f}x"
        )

    lines.extend(
        [
            "",
            "-" * 80,
            "SCALE PROJECTIONS (100M traces/day — Raindrop production scale):",
            "-" * 80,
        ]
    )

    for b in classifier_benchmarks:
        lines.append(
            f"  {b.approach}: {b.gpus_needed_for_100m_per_day:.1f} GPUs needed "
            f"| ${b.cost_per_1m_traces_usd * 100:.0f}/day"
        )

    llm_daily_cost = judge_cost["cost_per_1m_traces_usd"] * 100
    lines.append(
        f"  LLM Judge: ~{100_000_000 / (judge_throughput * 86400):.0f} API workers | ${llm_daily_cost:,.0f}/day"
    )
    lines.extend(
        [
            "",
            "Cost reduction vs LLM Judge at 100M/day:",
        ]
    )
    for b in classifier_benchmarks:
        savings = llm_daily_cost - (b.cost_per_1m_traces_usd * 100)
        lines.append(f"  {b.approach}: saves ${savings:,.0f}/day ({savings/max(llm_daily_cost,1)*100:.0f}%)")

    return "\n".join(lines)

