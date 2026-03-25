# AgentWatch — Technical Design Document

## Overview

AgentWatch is a prototype implementation of the core ML pipeline for AI agent observability: turning natural language issue descriptions into production-grade classifiers that can scan millions of agent traces.

This document covers the current architecture, design decisions, and what a production system would look like.

## Core Insight

The key tension in AI observability is:

- **LLMs** can classify any issue described in natural language, but cost ~$0.01/trace and take ~500ms. At 100M traces/day, that's $120K/day.
- **Small classifiers** run at ~0.2ms/trace and cost fractions of a cent per million, but need labeled data to train.

The solution is **distillation**: use the LLM to label a small sample (50-100 traces), then train a small classifier on those labels. The classifier agrees with the LLM 90-95% of the time at 1/1000th the cost.

This is what the Raindrop team calls "bespoke few-shot classifiers" — and it's the right framing. Each issue gets its own tiny, specialized model. The models are disposable: if the issue definition changes, retrain in seconds.

## Architecture Decisions

### Why SetFit over standard fine-tuning?

SetFit (Sentence Transformers Fine-Tuning) uses contrastive learning on a sentence-transformer, then fits a lightweight classification head. Advantages for this use case:

1. **Few examples needed** — works well with 8-64 labeled samples
2. **Fast training** — seconds, not hours
3. **No prompt engineering** — the model learns from examples, not instructions
4. **Small models** — ~22MB for MiniLM-based, fits in CPU memory
5. **No GPU required for inference** — though GPU helps for training

Standard LoRA/PEFT fine-tuning would be overkill: we're training binary classifiers on 50 examples, not adapting a foundation model.

### Why three approaches?

Different use cases benefit from different tradeoffs:

| Approach | Best when | Tradeoff |
|----------|-----------|----------|
| SetFit | 30+ labeled examples | Best accuracy, slower training (~10s) |
| Embedding + LR | 50+ examples, need interpretability | Fast, inspectable decision boundary |
| Prototype Network | 5-10 examples only | Fastest training, lower accuracy |

In production, you'd auto-select based on how many confident LLM labels you get.

### Why HDBSCAN for discovery?

The "unknown unknowns" problem requires clustering without knowing k (the number of failure types). HDBSCAN is density-based: it finds clusters of varying shapes and marks sparse points as noise. This matches the real data distribution — most traces are normal (noise), and failure modes form dense subclusters.

K-means would force you to pick k, which defeats the purpose of discovery.

### Why OpenTelemetry-compatible format?

Raindrop's SDK integrates via OTEL trace exporters. Using the same format (trace_id, span_id, parent_span_id, attributes) means traces from real agents using Vercel AI SDK, LangChain, etc. can be ingested directly. This isn't hypothetical — it's the actual integration path.

## What Production Looks Like

### Per-Tenant Model Management

Each customer gets isolated classifiers:

```
tenant_001/
├── issues/
│   ├── laziness/
│   │   ├── v1/ (setfit model)
│   │   ├── v2/ (retrained with more labels)
│   │   └── active → v2
│   └── hallucination/
│       └── v1/
tenant_002/
└── issues/
    └── custom_issue_1/
```

This maps to Raindrop's description: "we gradually train small, custom models, private to each company."

### Online Learning Pipeline

```
New trace arrives
    → Run through all active classifiers (batch, ~0.2ms each)
    → Store predictions
    → Every N traces, check if classifier drift exceeds threshold
    → If drifted: sample new traces → LLM judge → retrain
    → A/B test new classifier vs old
    → Promote if better
```

### Streaming Architecture (What I'd Build at Scale)

Current: PostgreSQL + batch classification (works to ~100K traces/day)

Production at Raindrop scale (100M+ traces/day):

```
Trace ingestion (OTEL) → Kafka/Kinesis → Tinybird (analytics store)
                                       → Classification worker pool
                                           → FAISS index (for similarity search)
                                           → Alert engine (Slack notifications)
```

Raindrop already uses Tinybird for analytics (confirmed in their case study). The classification workers would run the distilled models, writing predictions back to Tinybird as materialized columns.

### Active Learning

The most impactful improvement for classifier quality:

1. Run classifier on new traces
2. Flag traces with confidence between 0.4-0.6 (the uncertain zone)
3. Send those to the LLM judge for labeling
4. Add labels to training set, retrain

This creates a flywheel: the classifier gets better over time by focusing LLM spend on the examples it's least sure about. Cost grows sublinearly with trace volume.

### GPU Utilization

Current: sentence-transformer inference on CPU (~0.3ms/trace) or GPU (~0.1ms/trace).

At scale: batch multiple tenants' classifiers on the same GPU. Since each model shares the same sentence-transformer encoder, you encode once and run multiple classification heads — the marginal cost of an additional issue classifier per tenant is nearly zero.

## Limitations & Honest Gaps

1. **Synthetic traces** — the current failure injector produces stylized failures. Real production failures are messier and more varied. The pipeline architecture is sound, but real-world F1 scores would be lower.

2. **Single-tenant** — no model isolation, no multi-tenant API keys. This is the biggest gap vs production.

3. **No streaming** — batch classification only. Production needs real-time classification at ingest.

4. **No model versioning** — classifiers are in-memory. Production needs a model registry with rollback.

5. **FAISS index is local** — production needs a distributed vector store for similarity search across tenants.

## Metrics That Matter

For an AI observability product, the metrics that determine adoption:

- **Time to first insight**: How fast from "describe issue" to "here are the affected traces"? Current: ~15 seconds (labeling + training + classification).
- **Classifier agreement with LLM judge**: 90-95% is the target. Below 85% and users won't trust it.
- **Cost per million traces classified**: Must be <$1 for the business model to work at scale.
- **False positive rate**: For Slack alerts, FP rate must be <5% or engineers will ignore them.

