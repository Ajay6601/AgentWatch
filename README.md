# AgentWatch

**Few-shot silent failure detection for AI agents — materialized views for natural language over production traces.**

AgentWatch implements the core ML pipeline behind [Raindrop](https://raindrop.ai)-style AI agent monitoring: take a natural language description of an issue (e.g., *"agent is being lazy — giving short generic responses without using tools"*), automatically train a small, fast classifier, and scan millions of traces to find every occurrence.

## The Problem

AI agents fail silently. They don't throw exceptions — they give confident wrong answers, forget context, skip their tools, or loop endlessly. Traditional monitoring (latency, error rates, token counts) is blind to these failures. Offline evals catch known patterns on test data, but production agents encounter millions of unpredictable interactions.

**The gap:** You need to classify *every production trace* for *any issue described in natural language*, at a cost that doesn't bankrupt you.

## The Approach: Distillation Pipeline

```
Natural language         LLM-as-Judge           Few-Shot Classifier       Bulk Classification
issue description  →  labels ~80 traces  →  trains in seconds  →  classifies all traces
(free)                (~$0.01 total)        (SetFit/prototype)      (~0.2ms per trace)
```

1. **LLM-as-Judge** (strong system): GPT-4o-mini labels a sample of traces as positive/negative for the described issue. This is expensive (~$0.01/trace) but accurate.

2. **Few-Shot Distillation** (weak system): Those labels train a small, fast classifier. Three approaches benchmarked:
   - **SetFit** — contrastive learning on sentence-transformers. Best F1, trains in seconds.
   - **Embedding + Logistic Regression** — simplest, most interpretable.
   - **Prototype Network** — works with as few as 5 examples.

3. **Bulk Classification**: The distilled classifier runs on ALL traces at ~0.2ms/trace. At 100M traces/day, this costs ~$50/day vs ~$120,000/day for the LLM judge.

This is what Raindrop calls ["bespoke few-shot classifiers"](https://www.ycombinator.com/launches/Nn7-raindrop-deep-search) — bootstrapping weaker systems from stronger systems, creating materialized views for natural language.

## Key Features

- **Natural language issue definition** — describe any failure mode in plain English
- **LLM-as-judge labeling** — GPT-4o-mini bootstraps training labels
- **3 few-shot approaches** benchmarked head-to-head (SetFit, embedding+LR, prototypical networks)
- **Unknown unknowns discovery** — HDBSCAN clustering finds failure patterns nobody asked about
- **OpenTelemetry-compatible traces** — trace_id, span_id, parent_span_id format
- **Cost/latency benchmarks** — proves distillation is 500-1000x cheaper than LLM-on-every-trace
- **GPU-aware** — auto-detects CUDA, benchmarks GPU vs CPU training/inference

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/agentwatch.git
cd agentwatch
cp .env.example .env  # Add your OPENAI_API_KEY

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Run the full demo (no Docker needed)
python scripts/demo.py
```

### With Docker (includes PostgreSQL + dashboard)

```bash
docker-compose up -d
# API: http://localhost:8000
# Dashboard: http://localhost:3000

# Generate traces and run the pipeline
python scripts/demo.py
```

## Demo Output

```
Step 1: Generating 500 agent traces with injected failures
  ├── none: 275 (55.0%)
  ├── lazy: 75 (15.0%)
  ├── forgetting: 50 (10.0%)
  ├── hallucination: 50 (10.0%)
  └── tool_loop: 50 (10.0%)

Step 2: Creating issue: "Agent is being lazy..."
  [labeling] Labeled 74 traces: 12 positive, 62 negative
  [training] Best approach: setfit (F1=0.94, accuracy=0.93)
  [classifying] Found 71/500 traces (14.2%) affecting 68 users

Step 3: Benchmark — Distilled Classifiers vs LLM Judge
  Approach           F1    Acc   ms/trace  traces/sec    $/1M     Speedup
  LLM Judge        1.00   1.00     500.0          2.0  120.00        1x
  setfit           0.94   0.93       0.8      1250.0    0.11      625x
  embedding_lr     0.89   0.91       0.3      3333.3    0.04     1667x
  prototype        0.82   0.85       0.2      5000.0    0.03     2500x

Step 4: Discovered 4 unknown failure patterns
  ├── Cluster 0: "Generic deflection to documentation" (23 traces)
  ├── Cluster 1: "Repeated tool invocation without progress" (18 traces)
  ├── Cluster 2: "Fabricated feature references" (15 traces)
  └── Cluster 3: "Context amnesia in multi-turn sessions" (12 traces)
```

## Architecture

```
agentwatch/
├── agent/              # LangGraph support agent + failure injection
│   ├── support_agent.py    # ReAct agent with KB search, customer lookup
│   └── failure_injector.py # Laziness, forgetting, hallucination, loops
├── traces/             # OTEL-compatible trace format
│   └── models.py           # Trace, Span, SpanKind (maps to OTEL spans)
├── classifier/         # Core ML pipeline
│   ├── llm_judge.py        # GPT-4o-mini labels traces (async, batched)
│   ├── few_shot_trainer.py # SetFit, embedding+LR, prototype networks
│   ├── issue_detector.py   # Orchestrates full pipeline
│   └── benchmarks.py       # Cost/latency/accuracy comparison
├── discovery/          # Unknown unknowns
│   └── cluster.py          # HDBSCAN + auto-labeling
├── api/                # FastAPI backend
│   └── routes/             # issues, traces, discovery endpoints
└── dashboard/          # Next.js + TypeScript frontend
```

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| ML Core | PyTorch, SetFit, sentence-transformers | Few-shot classification on GPU |
| Embeddings | all-MiniLM-L6-v2, FAISS | Fast similarity search |
| LLM Judge | GPT-4o-mini (async) | Cost-effective labeling |
| Agent | LangGraph, LangChain | Realistic trace generation |
| Clustering | HDBSCAN | Density-based, no k needed |
| Backend | FastAPI, PostgreSQL, SQLAlchemy | OTEL-compatible ingestion |
| Frontend | Next.js, TypeScript, Tailwind, Recharts | Minimal monitoring dashboard |
| Traces | OpenTelemetry-compatible format | Industry standard |

## What I'd Build With 2 More Weeks

See [DESIGN_DOC.md](./DESIGN_DOC.md) for the full technical design, including:

- **Per-tenant model isolation** — each customer gets their own classifier namespace
- **Online learning** — incrementally update classifiers as new traces stream in
- **Active learning** — surface borderline cases for human review to improve classifier
- **Streaming inference** — Kafka/Tinybird integration for real-time classification at ingest
- **A/B testing** — compare classifier versions before promotion
- **Model registry** — version, stage, and rollback classifiers per issue per tenant

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/issues/` | POST | Create issue from natural language → runs full pipeline |
| `GET /api/issues/` | GET | List all tracked issues with metrics |
| `GET /api/issues/{id}/flags` | GET | Get traces flagged for an issue |
| `GET /api/issues/{id}/benchmark` | GET | Run benchmark on trained classifiers |
| `GET /api/traces/stats` | GET | Trace summary statistics |
| `GET /api/traces/{id}` | GET | Full trace with spans |
| `POST /api/discovery/run` | POST | Run unsupervised pattern discovery |

## License

MIT

