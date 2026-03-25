#!/usr/bin/env python3
"""
Generate synthetic agent traces and load them into the API.
Run this before using the dashboard.

Usage:
    python scripts/generate_traces.py [--n 500]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentwatch.agent.failure_injector import generate_trace_batch
from agentwatch.api.routes.issues import set_trace_store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Number of traces to generate")
    parser.add_argument("--output", type=str, default=None, help="Save traces to JSON file")
    args = parser.parse_args()

    print(f"Generating {args.n} traces...")
    traces = generate_trace_batch(n=args.n)

    # Distribution
    dist = {}
    for t in traces:
        ft = t.metadata.get("failure_type", "unknown")
        dist[ft] = dist.get(ft, 0) + 1

    print("\nTrace distribution:")
    for ft, count in sorted(dist.items()):
        print(f"  {ft}: {count} ({count/len(traces):.1%})")

    # Load into API store
    set_trace_store(traces)
    print(f"\nLoaded {len(traces)} traces into API memory store.")

    # Optionally save to file
    if args.output:
        data = [t.model_dump(mode="json") for t in traces]
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved to {args.output}")

    return traces


if __name__ == "__main__":
    main()

