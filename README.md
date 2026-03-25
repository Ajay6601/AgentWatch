## AgentWatch

AgentWatch is a small end-to-end reference project for generating synthetic agent traces (OpenTelemetry-ish spans), classifying issues from traces, and exploring unknown unknowns.

### Repository layout

- **`agentwatch/`**: Python package (FastAPI backend, trace models, classifier, discovery)
- **`dashboard/`**: Minimal Next.js frontend
- **`scripts/`**: CLI utilities (trace generation, benchmarks, demo)
- **`notebooks/`**: Interactive walkthrough
- **`tests/`**: Unit/integration tests

### Quickstart (local)

1) Create an env file.

```bash
copy .env.example .env
```

2) Install and run the API.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python -m agentwatch.api.main
```

3) Run the dashboard.

```bash
cd dashboard
npm install
npm run dev
```

### Docker

```bash
docker compose up --build
```

