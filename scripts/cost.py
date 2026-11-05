"""Sum spend from the call cache (results/cache/*.json) per model: tokens and USD at list prices."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# USD per million tokens (input, output), list prices as of Nov 2026
PRICE = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}  # Haiku 5.5: prompts <= 100K tokens


def main(cache=ROOT / "results" / "cache"):
    agg = defaultdict(lambda: [0, 0, 0])  # calls, in, out
    for f in cache.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        m = d.get("model", "unknown")
        a = agg[m]
        a[0] += 1; a[1] += d.get("in", 0); a[2] += d.get("out", 0)
    total = 0.0
    print(f"{'model':22s} {'calls':>7s} {'in_tok':>11s} {'out_tok':>11s} {'USD':>8s}")
    for m, (n, i, o) in sorted(agg.items()):
        pi, po = PRICE.get(m, (0, 0))
        usd = i / 1e6 * pi + o / 1e6 * po
        total += usd
        print(f"{m:22s} {n:7d} {i:11,d} {o:11,d} {usd:8.2f}")
    print(f"{'TOTAL':22s} {sum(a[0] for a in agg.values()):7d} {'':>11s} {'':>11s} {total:8.2f}")
    return total


if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
