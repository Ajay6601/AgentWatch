#!/usr/bin/env python3
"""
AgentWatch Demo — Full Pipeline Walkthrough

Run this to see the complete pipeline:
1. Generate 500 agent traces with injected failures
2. Track an issue using natural language ("agent is being lazy")
3. Watch the LLM judge → few-shot distillation → bulk classification
4. See benchmarks comparing distilled model vs LLM judge
5. Discover unknown failure patterns

Usage:
    python scripts/demo.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from agentwatch.agent.failure_injector import generate_trace_batch
from agentwatch.classifier.benchmarks import benchmark_classifier, compare_with_llm_judge
from agentwatch.classifier.issue_detector import IssueDetector
from agentwatch.classifier.llm_judge import compute_judge_cost
from agentwatch.discovery.cluster import discover_unknown_issues, label_discovered_clusters

console = Console()


async def main():
    console.print(
        Panel.fit(
            "[bold]AgentWatch[/bold] — Few-shot silent failure detection for AI agents\n"
            "Materialized views for natural language over production traces",
            border_style="blue",
        )
    )

    # ---- Step 1: Generate traces ----
    console.print("\n[bold cyan]Step 1:[/bold cyan] Generating agent traces with injected failures...")
    traces = generate_trace_batch(
        n=500,
        failure_distribution={
            "none": 0.55,
            "lazy": 0.15,
            "forgetting": 0.10,
            "hallucination": 0.10,
            "tool_loop": 0.10,
        },
    )

    # Show distribution
    dist = {}
    for t in traces:
        ft = t.metadata.get("failure_type", "unknown")
        dist[ft] = dist.get(ft, 0) + 1

    table = Table(title="Trace Distribution")
    table.add_column("Failure Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")
    for ft, count in sorted(dist.items()):
        table.add_row(ft, str(count), f"{count/len(traces):.1%}")
    console.print(table)

    # ---- Step 2: Track "laziness" issue ----
    console.print(
        "\n[bold cyan]Step 2:[/bold cyan] Creating issue: [italic]'Agent is being lazy — "
        "giving short generic responses without using available tools'[/italic]\n"
    )

    detector = IssueDetector()

    def on_progress(p):
        console.print(f"  [{p.stage}] {p.message}")

    t0 = time.time()
    issue = await detector.create_issue(
        description=(
            "Agent is being lazy — giving short, generic responses like 'check the docs' "
            "or 'visit our website' without actually searching the knowledge base or "
            "using any of its available tools to help the user."
        ),
        all_traces=traces,
        labeling_sample_size=80,
        approach="all",
        on_progress=on_progress,
    )
    pipeline_time = time.time() - t0

    console.print(f"\n  Pipeline completed in [bold]{pipeline_time:.1f}s[/bold]")

    # ---- Step 3: Show results ----
    console.print("\n[bold cyan]Step 3:[/bold cyan] Results\n")

    result_table = Table(title=f"Issue: {issue.description[:60]}...")
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", justify="right")
    result_table.add_row("Status", issue.status)
    result_table.add_row("Best approach", issue.best_approach)
    result_table.add_row("Traces classified", str(issue.total_classified))
    result_table.add_row("Traces flagged", str(len(issue.flagged_traces)))
    result_table.add_row("Frequency", f"{issue.frequency:.1%}")
    result_table.add_row("Affected users", str(len(issue.affected_users)))
    result_table.add_row("LLM judge: positive", str(issue.n_positive))
    result_table.add_row("LLM judge: negative", str(issue.n_negative))
    console.print(result_table)

    # Show classifier metrics for all approaches
    if issue.classifiers:
        metrics_table = Table(title="Classifier Comparison")
        metrics_table.add_column("Approach", style="cyan")
        metrics_table.add_column("Accuracy", justify="right")
        metrics_table.add_column("Precision", justify="right")
        metrics_table.add_column("Recall", justify="right")
        metrics_table.add_column("F1", justify="right")
        metrics_table.add_column("Train Time", justify="right")
        metrics_table.add_column("ms/trace", justify="right")

        for name, clf in issue.classifiers.items():
            if clf.metrics:
                m = clf.metrics
                style = "bold green" if name == issue.best_approach else ""
                metrics_table.add_row(
                    f"{'→ ' if name == issue.best_approach else '  '}{name}",
                    f"{m.accuracy:.3f}",
                    f"{m.precision:.3f}",
                    f"{m.recall:.3f}",
                    f"{m.f1:.3f}",
                    f"{m.training_time_ms:.0f}ms",
                    f"{m.inference_time_per_trace_ms:.2f}",
                    style=style,
                )
        console.print(metrics_table)

    # ---- Step 4: Benchmark vs LLM judge ----
    console.print("\n[bold cyan]Step 4:[/bold cyan] Benchmark — Distilled Classifiers vs LLM Judge\n")

    # Prepare test data from judge results
    trace_map = {t.trace_id: t for t in traces}
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
    console.print(comparison)

    # ---- Step 5: Discover unknown issues ----
    console.print("\n[bold cyan]Step 5:[/bold cyan] Discovering unknown failure patterns...\n")

    known_ids = {tid for tid, _ in issue.flagged_traces}
    discovery = discover_unknown_issues(traces, known_issue_trace_ids=known_ids, min_cluster_size=5)

    if discovery.clusters:
        discovery.clusters = await label_discovered_clusters(discovery.clusters)

        disc_table = Table(title="Discovered Patterns (Unknown Unknowns)")
        disc_table.add_column("Cluster", style="cyan")
        disc_table.add_column("Label")
        disc_table.add_column("Size", justify="right")
        disc_table.add_column("% of traces", justify="right")
        disc_table.add_column("Confidence", justify="right")

        for c in discovery.clusters[:10]:
            disc_table.add_row(
                str(c.cluster_id),
                c.suggested_label,
                str(c.size),
                f"{c.percentage:.1%}",
                f"{c.avg_confidence:.2f}",
            )
        console.print(disc_table)
    else:
        console.print("  No distinct clusters found (try more traces)")

    # ---- Step 6: Show sample flagged traces ----
    console.print("\n[bold cyan]Step 6:[/bold cyan] Sample flagged traces\n")

    for trace_id, conf in issue.flagged_traces[:5]:
        t = trace_map.get(trace_id)
        if t:
            console.print(
                Panel(
                    f"[dim]Trace: {trace_id}[/dim]\n"
                    f"[bold]Query:[/bold] {t.user_query[:100]}\n"
                    f"[bold]Response:[/bold] {t.agent_response[:150]}\n"
                    f"[bold]Tools used:[/bold] {', '.join(t.tools_called) or 'None'}\n"
                    f"[bold]Ground truth:[/bold] {t.metadata.get('failure_type', 'unknown')}\n"
                    f"[bold]Confidence:[/bold] {conf:.2f}",
                    title=f"Flagged trace (confidence: {conf:.2f})",
                    border_style="red" if conf > 0.8 else "yellow",
                )
            )

    # ---- Summary ----
    console.print(
        Panel.fit(
            f"[bold green]Pipeline Summary[/bold green]\n\n"
            f"• Generated {len(traces)} traces with {len(dist)-1} failure types\n"
            f"• LLM judge labeled {len(issue.judge_results)} traces (cost: "
            f"${compute_judge_cost(len(issue.judge_results))['estimated_cost_usd']:.4f})\n"
            f"• Trained 3 classifiers in {pipeline_time:.1f}s\n"
            f"• Best: {issue.best_approach} (F1={issue.classifiers[issue.best_approach].metrics.f1:.3f})\n"
            f"• Classified all {len(traces)} traces at "
            f"{issue.classifiers[issue.best_approach].metrics.inference_time_per_trace_ms:.2f}ms/trace\n"
            f"• Found {len(issue.flagged_traces)} lazy traces ({issue.frequency:.1%}) "
            f"affecting {len(issue.affected_users)} users\n"
            f"• Discovered {discovery.n_clusters} unknown failure patterns",
            border_style="green",
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

