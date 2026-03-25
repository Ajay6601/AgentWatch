"""
LLM-as-judge: the "strong system" in the bootstrap pipeline.

Takes a natural language issue description + a batch of traces,
and uses a large LLM to label each trace as positive/negative.
This is the expensive step that we then distill away.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from openai import AsyncOpenAI

from agentwatch.config import get_settings
from agentwatch.traces.models import Trace


@dataclass
class JudgeResult:
    trace_id: str
    label: bool  # True = issue present
    confidence: float  # 0.0-1.0
    reasoning: str


JUDGE_SYSTEM_PROMPT = """You are an expert AI quality evaluator. You analyze agent interaction traces 
to determine whether a specific issue is present.

You will receive:
1. An ISSUE DESCRIPTION — a natural language description of the behavior to detect
2. A TRACE — the full agent execution trace including user query, tool calls, and response

Your job: determine if this trace exhibits the described issue.

Respond ONLY with this JSON (no markdown, no backticks):
{"label": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""


JUDGE_USER_TEMPLATE = """ISSUE DESCRIPTION:
{issue_description}

TRACE:
{trace_text}

Does this trace exhibit the described issue? Respond with JSON only."""


async def judge_single_trace(
    client: AsyncOpenAI,
    trace: Trace,
    issue_description: str,
    model: str = "gpt-4o-mini",
) -> JudgeResult:
    """Label a single trace using the LLM judge."""
    try:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=200,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": JUDGE_USER_TEMPLATE.format(
                        issue_description=issue_description,
                        trace_text=trace.to_text(),
                    ),
                },
            ],
        )
        raw = resp.choices[0].message.content.strip()
        # Clean potential markdown wrapping
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)

        return JudgeResult(
            trace_id=trace.trace_id,
            label=bool(parsed["label"]),
            confidence=float(parsed.get("confidence", 0.8)),
            reasoning=parsed.get("reasoning", ""),
        )
    except Exception as e:
        # On parse failure, return low-confidence negative
        return JudgeResult(
            trace_id=trace.trace_id,
            label=False,
            confidence=0.0,
            reasoning=f"Judge error: {str(e)[:100]}",
        )


async def judge_batch(
    traces: list[Trace],
    issue_description: str,
    max_concurrent: int | None = None,
) -> list[JudgeResult]:
    """
    Label a batch of traces using the LLM judge.
    This is the expensive step — typically $0.01-0.05 per trace with gpt-4o-mini.
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    sem = asyncio.Semaphore(max_concurrent or settings.judge_max_concurrent)

    async def bounded_judge(trace: Trace) -> JudgeResult:
        async with sem:
            return await judge_single_trace(client, trace, issue_description, settings.judge_model)

    results = await asyncio.gather(*[bounded_judge(t) for t in traces])
    return list(results)


def compute_judge_cost(n_traces: int, avg_input_tokens: int = 400, avg_output_tokens: int = 60) -> dict:
    """Estimate the cost of running the LLM judge on N traces."""
    # gpt-4o-mini pricing (as of 2024)
    input_cost_per_1k = 0.00015
    output_cost_per_1k = 0.0006
    total_input = n_traces * avg_input_tokens / 1000 * input_cost_per_1k
    total_output = n_traces * avg_output_tokens / 1000 * output_cost_per_1k
    return {
        "n_traces": n_traces,
        "model": "gpt-4o-mini",
        "estimated_cost_usd": round(total_input + total_output, 4),
        "cost_per_trace_usd": round((total_input + total_output) / max(n_traces, 1), 6),
        "cost_per_1m_traces_usd": round((total_input + total_output) / max(n_traces, 1) * 1_000_000, 2),
    }

