# AgentWatch

**Few-shot silent failure detection for AI agents — materialized views for natural language over production traces.**

AgentWatch implements the core classification engine behind AI agent observability platforms like [Raindrop](https://raindrop.ai). An engineer types a natural language description of a problem — *"agent is being lazy"* — and the system automatically trains a small, fast classifier that can scan every production trace to find where that problem occurs, how often, and who's affected.

## The Problem

AI agents don't crash — they fail silently. A support agent that gives a confident wrong answer, forgets what the user said, or skips its tools entirely never throws an exception. The engineering team has zero visibility.

The naive solution is to run every trace through GPT-4 and ask "does this trace have a problem?" But at 100M traces/day, that costs ~$10,000/day per issue. You can't afford it.

The real solution is **distillation**: use the expensive model once on a small sample to teach a cheap model, then run the cheap model on everything.

## How It Works

```
"agent is being lazy"        GPT-4o-mini labels         SetFit classifier         All 500 traces
(natural language)     →     100 sample traces     →    trains in 18 min    →     classified in 11s
                             ($0.01 total)               (F1 = 0.95)              (12.2% flagged)
```

### Step 1 — LLM-as-Judge (strong teacher)

GPT-4o-mini evaluates a stratified sample of ~100 traces against the issue description. Each trace gets a binary label (issue present or not) with a confidence score. Cost: ~$0.01 total.

### Step 2 — Few-Shot Distillation (fast student)

Those labels train a small classifier. Three approaches, benchmarked head-to-head:

| Approach | How it works | F1 | Train time |
|----------|-------------|-----|-----------|
| **SetFit** | Contrastive learning on sentence-transformers, then classification head | 0.957 | ~18 min (CPU) |
| **Embedding + LR** | Encode traces with MiniLM, fit logistic regression | 0.857 | ~2.4s |
| **Prototype Network** | Compute class centroids, classify by distance | 0.800 | ~1.4s |

### Step 3 — Bulk Classification

The best classifier runs on ALL traces. Result: "61 out of 500 traces are lazy (12.2%), affecting 59 users" — with a link to each trace.

### Step 4 — Unknown Unknowns Discovery

HDBSCAN clusters all traces in embedding space and surfaces patterns nobody asked about. The system found 35 clusters including "Repeated Incorrect Response," "Repetitive Request for Customer ID," and "Customer Lookup Failure" — real issues discovered automatically.

## Demo Results

```
Pipeline Summary
────────────────
• Generated 500 real agent traces (GPT-4o-mini with prompted failure modes)
• LLM judge labeled 100 traces (cost: $0.01)
• Trained 3 classifiers — best: SetFit (F1=0.957, accuracy=0.929)  
• Classified all 500 traces at ~15ms/trace
• Found 61 lazy traces (12.2%) affecting 59 users
• Discovered 35 unknown failure patterns
• 96% cost reduction vs running LLM judge on every trace

Distillation Benchmark
──────────────────────
Approach          F1     Acc    $/1M traces    vs LLM Judge
LLM Judge        1.00   1.00   $96.00         baseline
SetFit           0.96   0.93   $3.79          96% cheaper
Embedding+LR     0.86   0.79   $3.63          96% cheaper
Prototype        0.80   0.71   $3.54          96% cheaper

At Raindrop scale (100M traces/day):
  LLM Judge: ~$9,600/day
  Distilled classifier: ~$370/day
  Savings: $9,230/day (96%)
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/agentwatch.git
cd agentwatch
cp .env.example .env   # Add your OPENAI_API_KEY

# Install
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Run the full pipeline demo (~20 min on CPU, ~$0.15 OpenAI cost)
python scripts/demo.py

# Or start the API + dashboard
python scripts/run_server.py --n 500   # Terminal 1
cd dashboard && npm install && npm run dev   # Terminal 2
# Open http://localhost:3000
```

## Architecture

```
agentwatch/
├── agent/                  # Trace generation
│   ├── real_trace_generator.py  # Runs LLM with failure-inducing prompts
│   ├── failure_injector.py      # Fast synthetic traces for testing
│   └── tools.py                 # KB search, customer lookup, status check
│
├── classifier/             # Core ML pipeline (Layer 2)
│   ├── llm_judge.py             # GPT-4o-mini labels sample traces
│   ├── few_shot_trainer.py      # SetFit, embedding+LR, prototype networks
│   ├── issue_detector.py        # Orchestrates: judge → train → classify
│   └── benchmarks.py            # Cost/latency/accuracy comparisons
│
├── discovery/              # Unknown unknowns
│   └── cluster.py               # HDBSCAN clustering + auto-labeling
│
├── traces/                 # Data models
│   └── models.py                # OTEL-compatible Trace/Span format
│
├── api/                    # FastAPI backend
│   └── routes/                  # Issues, traces, discovery endpoints
│
└── dashboard/              # Next.js + TypeScript frontend
    └── app/components/          # Issue creator, trace viewer, metrics
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Classification | PyTorch, SetFit, sentence-transformers | Few-shot model training |
| Embeddings | all-MiniLM-L6-v2, FAISS | Trace encoding and similarity search |
| LLM Judge | GPT-4o-mini (async, batched) | Bootstrap training labels |
| Agent | LangGraph, LangChain, OpenAI | Real agent trace generation |
| Clustering | HDBSCAN | Density-based pattern discovery |
| Backend | FastAPI, Pydantic | REST API with OTEL-compatible ingestion |
| Frontend | Next.js, TypeScript, Tailwind, Recharts | Monitoring dashboard |
| Traces | OpenTelemetry-compatible format | Industry-standard trace_id/span_id |

## API

| Endpoint | Description |
|----------|-------------|
| `POST /api/issues/` | Create issue from natural language → runs full pipeline |
| `GET /api/issues/` | List tracked issues with metrics |
| `GET /api/issues/{id}/flags` | Traces flagged for an issue |
| `GET /api/issues/{id}/benchmark` | Benchmark classifiers for an issue |
| `GET /api/traces/stats` | Trace summary statistics |
| `GET /api/traces/{id}` | Full trace with span details |
| `POST /api/discovery/run` | Discover unknown failure patterns |

## What I'd Build Next

See [DESIGN_DOC.md](./DESIGN_DOC.md) for the full production scaling plan:

- **Per-tenant model isolation** — each customer's classifiers in their own namespace
- **Online learning** — retrain classifiers as agent behavior drifts
- **Active learning** — route borderline traces to the LLM judge, not all traces
- **Streaming inference** — classify at ingest via Kafka/Tinybird, not batch
- **A/B testing** — compare classifier versions before promoting to production
- **Model registry** — version, stage, and rollback per issue per tenant

## License

MIT
