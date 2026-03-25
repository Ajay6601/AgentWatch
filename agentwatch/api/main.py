"""
FastAPI backend for AgentWatch.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentwatch.api.routes import discovery, issues, traces

app = FastAPI(
    title="AgentWatch",
    description="Few-shot silent failure detection for AI agents",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(traces.router, prefix="/api/traces", tags=["traces"])
app.include_router(issues.router, prefix="/api/issues", tags=["issues"])
app.include_router(discovery.router, prefix="/api/discovery", tags=["discovery"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agentwatch"}

