"""
Failure injector — generates traces with realistic, silent failure modes.
These are the kinds of failures Raindrop detects: no exception thrown,
agent confidently returns a bad response.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta

from agentwatch.traces.models import Span, SpanKind, SpanStatus, Trace

# Diverse queries covering different support domains
QUERIES = [
    "How do I reset my password?",
    "I was charged twice this month, can you help?",
    "What are your API rate limits?",
    "I need to set up the Slack integration",
    "How do I export my data in GDPR format?",
    "My webhook deliveries are failing",
    "Can you check if the API is down? I'm getting timeouts",
    "I need to change my team member permissions",
    "Our enterprise SSO isn't working after the update",
    "I want to cancel my subscription, the product is too slow",
    "How do I migrate from the free tier to pro?",
    "The search feature isn't returning recent results",
    "I need to add a custom domain to my account",
    "Can you explain the difference between Editor and Admin roles?",
    "Our data pipeline integration keeps disconnecting",
    "I'm locked out of my account after too many failed attempts",
    "How do I set up automated billing for my team?",
    "The dashboard is showing stale data from yesterday",
    "I need an invoice for tax purposes",
    "Can you help me debug a 403 error on the API?",
]


def _ts(base: datetime, offset_ms: int) -> datetime:
    return base + timedelta(milliseconds=offset_ms)


def generate_normal_trace(query: str | None = None) -> Trace:
    """Generate a trace where the agent behaves correctly."""
    q = query or random.choice(QUERIES)
    t0 = datetime.utcnow()
    trace = Trace(
        trace_id=uuid.uuid4().hex[:32],
        user_id=f"user_{random.randint(100, 999)}",
        timestamp=t0,
    )

    # User message
    trace.spans.append(
        Span(
            name="user_message",
            kind=SpanKind.USER_MESSAGE,
            start_time=t0,
            end_time=t0,
            content=q,
        )
    )

    # Agent searches KB
    trace.spans.append(
        Span(
            name="search_knowledge_base",
            kind=SpanKind.TOOL_CALL,
            start_time=_ts(t0, 100),
            end_time=_ts(t0, 350),
            tool_name="search_knowledge_base",
            tool_input=q,
            tool_output="Found relevant article: detailed step-by-step instructions...",
        )
    )

    # Agent may look up customer
    if random.random() > 0.4:
        cid = f"C00{random.randint(1, 5)}"
        trace.spans.append(
            Span(
                name="lookup_customer",
                kind=SpanKind.TOOL_CALL,
                start_time=_ts(t0, 400),
                end_time=_ts(t0, 500),
                tool_name="lookup_customer",
                tool_input=cid,
                tool_output="Customer: Acme Corp | Plan: Enterprise | Status: active",
            )
        )

    # LLM generates response
    response = (
        f"I found the relevant information for your question about "
        f"'{q[:50]}'. Based on our knowledge base, here are the detailed steps: "
        f"First, navigate to the appropriate settings page. Then follow the "
        f"step-by-step instructions provided. If you encounter any issues, "
        f"I've also checked your account status and everything looks good. "
        f"Let me know if you need any further assistance."
    )
    trace.spans.append(
        Span(
            name="llm_generate",
            kind=SpanKind.LLM_CALL,
            start_time=_ts(t0, 550),
            end_time=_ts(t0, 1200),
            model="gpt-4o-mini",
            content=response,
            input_tokens=random.randint(300, 600),
            output_tokens=random.randint(150, 400),
        )
    )

    trace.spans.append(
        Span(
            name="agent_response",
            kind=SpanKind.AGENT_RESPONSE,
            start_time=_ts(t0, 1200),
            end_time=_ts(t0, 1250),
            content=response,
        )
    )

    trace.finalize()
    trace.metadata["failure_type"] = "none"
    return trace


def generate_lazy_trace(query: str | None = None) -> Trace:
    """
    LAZINESS: Agent gives a generic, short response without using tools.
    This is Raindrop's preset #1 — the agent CBA to actually help.
    """
    q = query or random.choice(QUERIES)
    t0 = datetime.utcnow()
    trace = Trace(
        trace_id=uuid.uuid4().hex[:32],
        user_id=f"user_{random.randint(100, 999)}",
        timestamp=t0,
    )

    trace.spans.append(
        Span(
            name="user_message",
            kind=SpanKind.USER_MESSAGE,
            start_time=t0,
            end_time=t0,
            content=q,
        )
    )

    # No tool calls — agent skips research entirely
    lazy_responses = [
        "You can find that information in our help center.",
        "Please check the documentation for details on this.",
        "I'd recommend reaching out to our support team for this.",
        "That should be in your account settings somewhere.",
        "You can try the settings page for that.",
        "I'm not sure about the specifics, but our docs should help.",
        "Please visit our website for more information.",
    ]
    response = random.choice(lazy_responses)

    trace.spans.append(
        Span(
            name="llm_generate",
            kind=SpanKind.LLM_CALL,
            start_time=_ts(t0, 50),
            end_time=_ts(t0, 200),
            model="gpt-4o-mini",
            content=response,
            input_tokens=random.randint(100, 200),
            output_tokens=random.randint(10, 30),
        )
    )

    trace.spans.append(
        Span(
            name="agent_response",
            kind=SpanKind.AGENT_RESPONSE,
            start_time=_ts(t0, 200),
            end_time=_ts(t0, 210),
            content=response,
        )
    )

    trace.finalize()
    trace.metadata["failure_type"] = "lazy"
    return trace


def generate_forgetting_trace(query: str | None = None) -> Trace:
    """
    FORGETTING: Agent ignores context from the conversation,
    asks for info already provided, or contradicts prior messages.
    """
    q = query or random.choice(QUERIES)
    t0 = datetime.utcnow()
    trace = Trace(
        trace_id=uuid.uuid4().hex[:32],
        user_id=f"user_{random.randint(100, 999)}",
        session_id=f"sess_{random.randint(1000, 9999)}",
        timestamp=t0,
        metadata={
            "prior_context": "Customer already said they are on Enterprise plan and mentioned customer ID C001"
        },
    )

    trace.spans.append(
        Span(
            name="user_message",
            kind=SpanKind.USER_MESSAGE,
            start_time=t0,
            end_time=t0,
            content=f"{q} (I already told you I'm on the Enterprise plan, customer C001)",
        )
    )

    # Agent searches but ignores the context provided
    trace.spans.append(
        Span(
            name="search_knowledge_base",
            kind=SpanKind.TOOL_CALL,
            start_time=_ts(t0, 100),
            end_time=_ts(t0, 300),
            tool_name="search_knowledge_base",
            tool_input=q,
            tool_output="Found relevant article with general instructions...",
        )
    )

    # Agent asks for info already provided
    forgetting_responses = [
        "I'd be happy to help with that! Could you first tell me what plan you're on?",
        "Sure, I can look into this. What's your customer ID?",
        "I can help with that. Are you on our free, pro, or enterprise plan?",
        "Let me check that for you. Could you remind me of your account details?",
    ]
    response = random.choice(forgetting_responses)

    trace.spans.append(
        Span(
            name="llm_generate",
            kind=SpanKind.LLM_CALL,
            start_time=_ts(t0, 350),
            end_time=_ts(t0, 700),
            model="gpt-4o-mini",
            content=response,
            input_tokens=random.randint(400, 700),
            output_tokens=random.randint(30, 80),
        )
    )

    trace.spans.append(
        Span(
            name="agent_response",
            kind=SpanKind.AGENT_RESPONSE,
            start_time=_ts(t0, 700),
            end_time=_ts(t0, 710),
            content=response,
        )
    )

    trace.finalize()
    trace.metadata["failure_type"] = "forgetting"
    return trace


def generate_hallucination_trace(query: str | None = None) -> Trace:
    """
    HALLUCINATION: Agent gives a confident, specific but incorrect answer.
    Cites non-existent features or makes up policy details.
    """
    q = query or random.choice(QUERIES)
    t0 = datetime.utcnow()
    trace = Trace(
        trace_id=uuid.uuid4().hex[:32],
        user_id=f"user_{random.randint(100, 999)}",
        timestamp=t0,
    )

    trace.spans.append(
        Span(
            name="user_message",
            kind=SpanKind.USER_MESSAGE,
            start_time=t0,
            end_time=t0,
            content=q,
        )
    )

    # Agent searches but ignores/misinterprets results
    trace.spans.append(
        Span(
            name="search_knowledge_base",
            kind=SpanKind.TOOL_CALL,
            start_time=_ts(t0, 100),
            end_time=_ts(t0, 300),
            tool_name="search_knowledge_base",
            tool_input=q,
            tool_output="Found relevant article about account settings...",
        )
    )

    hallucination_responses = [
        "You can access the Advanced AI Dashboard by going to Settings > AI > Dashboard Pro. "
        "This feature lets you create custom ML models directly in the browser. "
        "It was launched last month and is available on all plans.",
        "Our refund policy allows full refunds within 90 days, no questions asked. "
        "Just go to Billing > Refund Request and your money will be back in 24 hours. "
        "We also offer 200% credit if you prefer that option.",
        "You can use our built-in SSH tunnel feature at Settings > Network > SSH. "
        "This gives you direct database access with root privileges. "
        "Just paste your public key and you'll have full access immediately.",
        "The API supports unlimited requests on all plans — there are no rate limits. "
        "You just need to add the X-Unlimited-Access: true header to your requests. "
        "This was announced in our v5.0 release last week.",
    ]
    response = random.choice(hallucination_responses)

    trace.spans.append(
        Span(
            name="llm_generate",
            kind=SpanKind.LLM_CALL,
            start_time=_ts(t0, 350),
            end_time=_ts(t0, 900),
            model="gpt-4o-mini",
            content=response,
            input_tokens=random.randint(300, 600),
            output_tokens=random.randint(100, 250),
        )
    )

    trace.spans.append(
        Span(
            name="agent_response",
            kind=SpanKind.AGENT_RESPONSE,
            start_time=_ts(t0, 900),
            end_time=_ts(t0, 910),
            content=response,
        )
    )

    trace.finalize()
    trace.metadata["failure_type"] = "hallucination"
    return trace


def generate_loop_trace(query: str | None = None) -> Trace:
    """
    TOOL LOOP: Agent calls the same tool repeatedly without making progress.
    Common failure mode in production agents.
    """
    q = query or random.choice(QUERIES)
    t0 = datetime.utcnow()
    trace = Trace(
        trace_id=uuid.uuid4().hex[:32],
        user_id=f"user_{random.randint(100, 999)}",
        timestamp=t0,
    )

    trace.spans.append(
        Span(
            name="user_message",
            kind=SpanKind.USER_MESSAGE,
            start_time=t0,
            end_time=t0,
            content=q,
        )
    )

    # Agent calls the same tool 3-5 times with minor variations
    loop_count = random.randint(3, 5)
    offset = 100
    for i in range(loop_count):
        trace.spans.append(
            Span(
                name=f"search_knowledge_base_{i}",
                kind=SpanKind.TOOL_CALL,
                start_time=_ts(t0, offset),
                end_time=_ts(t0, offset + 200),
                tool_name="search_knowledge_base",
                tool_input=q if i == 0 else f"{q} more details",
                tool_output="No additional articles found." if i > 0 else "Found an article...",
            )
        )
        offset += 300

    response = (
        "I've searched our knowledge base multiple times but couldn't find a "
        "definitive answer. Let me search one more time... Actually, I think "
        "the information might be in a different section. Let me try again."
    )
    trace.spans.append(
        Span(
            name="llm_generate",
            kind=SpanKind.LLM_CALL,
            start_time=_ts(t0, offset),
            end_time=_ts(t0, offset + 400),
            model="gpt-4o-mini",
            content=response,
            input_tokens=random.randint(800, 1200),
            output_tokens=random.randint(50, 100),
        )
    )

    trace.spans.append(
        Span(
            name="agent_response",
            kind=SpanKind.AGENT_RESPONSE,
            start_time=_ts(t0, offset + 400),
            end_time=_ts(t0, offset + 410),
            content=response,
        )
    )

    trace.finalize()
    trace.metadata["failure_type"] = "tool_loop"
    return trace


# Registry of failure generators with default probabilities
FAILURE_GENERATORS = {
    "none": (generate_normal_trace, 0.55),
    "lazy": (generate_lazy_trace, 0.15),
    "forgetting": (generate_forgetting_trace, 0.10),
    "hallucination": (generate_hallucination_trace, 0.10),
    "tool_loop": (generate_loop_trace, 0.10),
}


def generate_trace_batch(
    n: int = 500,
    failure_distribution: dict[str, float] | None = None,
) -> list[Trace]:
    """Generate a batch of traces with controlled failure distribution."""
    dist = failure_distribution or {k: v[1] for k, v in FAILURE_GENERATORS.items()}
    types = list(dist.keys())
    weights = [dist[t] for t in types]

    traces = []
    for _ in range(n):
        ftype = random.choices(types, weights=weights, k=1)[0]
        gen_fn = FAILURE_GENERATORS[ftype][0]
        traces.append(gen_fn())

    random.shuffle(traces)
    return traces

