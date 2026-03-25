#!/usr/bin/env python3
"""
Start the AgentWatch API server with pre-generated traces loaded.
This is the script to run for the dashboard demo.

Usage:
    python scripts/run_server.py [--n 500] [--port 8000]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentwatch.agent.failure_injector import generate_trace_batch
from agentwatch.api.routes.issues import set_trace_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Generating {args.n} traces...")
    traces = generate_trace_batch(n=args.n)

    dist = {}
    for t in traces:
        ft = t.metadata.get("failure_type", "unknown")
        dist[ft] = dist.get(ft, 0) + 1
    for ft, count in sorted(dist.items()):
        print(f"  {ft}: {count} ({count/len(traces):.1%})")

    set_trace_store(traces)
    print(f"Loaded {len(traces)} traces. Starting API on port {args.port}...")

    import uvicorn

    uvicorn.run(
        "agentwatch.api.main:app",
        host="0.0.0.0",
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()

