"""Auto-report for the curve-momentum (grinder) forward paper-test. Run on a timer
(cm-report.timer). Reads cm_trades.jsonl, computes forward stats + a STATISTICAL VERDICT,
appends a timestamped line to cm_forward_report.log, and prints to stdout (journal).

The verdict gates on power so it never over-reads a tiny sample:
  PENDING     n < MIN_N            — too few trades to conclude
  HOLDS       CI-low  > 0          — forward edge positive -> regime generalises
  COLLAPSED   CI-high < 0          — forward edge negative -> regime-dependent
  INCONCLUSIVE                     — enough trades but the CI straddles 0

Benchmark = grinder, BASE frictions (what the live strategy uses): mean ~+11%, win ~51%.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

LOG = Path("cm_trades.jsonl")
OUT = Path("cm_forward_report.log")
MIN_N = 150          # minimum closed trades for a first read
DECISIVE_N = 300     # past this, a HOLDS/COLLAPSED verdict is "you can stop"
BENCH_MEAN, BENCH_WIN = 0.12, 0.63  # floor=38 + scale-out 50%@1.3x (backtest+stress)


def verdict(r: np.ndarray) -> tuple[str, str]:
    n = len(r)
    if n < MIN_N:
        return "PENDING", f"underpowered ({n}/{MIN_N} trades)"
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(r, n).mean() for _ in range(3000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    tag = " [DECISIVE]" if n >= DECISIVE_N else ""
    if lo > 0:
        return "HOLDS", f"forward edge POSITIVE, CI-low {lo:+.1%} > 0 — regime generalises{tag}"
    if hi < 0:
        return "COLLAPSED", f"forward edge NEGATIVE, CI-high {hi:+.1%} < 0 — regime-dependent{tag}"
    return "INCONCLUSIVE", f"CI [{lo:+.1%}, {hi:+.1%}] straddles 0 — keep collecting"


def main() -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if not LOG.exists() or not LOG.read_text().strip():
        line = f"[{stamp}] PENDING — no closed CM trades yet"
        print(line)
        with OUT.open("a") as f:
            f.write(line + "\n")
        return
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    r = np.array([x["ret"] for x in rows])
    n = len(r)
    win = (r > 0).mean()
    mean = r.mean()
    by_reason = {k: sum(x["reason"] == k for x in rows) for k in sorted({x["reason"] for x in rows})}
    state, why = verdict(r)
    line = (f"[{stamp}] {state}: n={n} win={win:.0%} mean={mean:+.1%} "
            f"(bench ~{BENCH_MEAN:+.0%}/{BENCH_WIN:.0%}) exits={by_reason} — {why}")
    print(line)
    with OUT.open("a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
