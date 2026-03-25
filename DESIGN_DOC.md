## Design doc (high level)

### Goals

- Generate **synthetic** agent runs that emit **trace spans** in an OpenTelemetry-compatible shape.
- Run a lightweight **issue detector** pipeline:
  - label traces via an LLM-as-judge (optional stub)
  - train a small text model (SetFit / sentence-transformer) (optional stub)
  - serve classification results via API
- Detect **unknown unknowns** by clustering unlabeled/low-confidence issues.

### Components

- **Agent trace generator** (`agentwatch/agent/`, `agentwatch/traces/`)
- **Classifier pipeline** (`agentwatch/classifier/`)
- **Discovery** (`agentwatch/discovery/`)
- **FastAPI backend** (`agentwatch/api/`)
- **Dashboard** (`dashboard/`)

### Storage

- Postgres via SQLAlchemy.
- Traces stored as JSON blobs plus extracted fields for filtering.

