"""Forward paper-test report — read cm_trades.jsonl (written live by the monitor's
CurveMomentumPaper) and compare the FORWARD, out-of-regime results to the in-window
benchmark. This IS the regime test: if forward stats track the benchmark, the edge
generalizes; if they collapse, it was regime-specific.

The live strategy uses BASE frictions (MEV 1%, no haircut, exit at the trigger swap),
so the apples-to-apples benchmark is the base-friction in-window number, not the
moderate-friction +24%.

  Benchmark (in-window OOS late slice, base frictions, from the end-to-end replay):
    mean ~+62%, win ~28%, TP-dominated.   (moderate-friction equivalent: ~+24%/CI+10.6%)

Run: uv run python cm_forward_report.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cm_gap_filter import load_clean_trades

LOG = Path("cm_trades.jsonl")
BENCH_MEAN, BENCH_WIN = 0.12, 0.63  # floor=38 + scale-out 50%@1.3x


def main() -> None:
    if not LOG.exists():
        print("no cm_trades.jsonl yet — the forward test hasn't closed a trade.")
        return
    rows, dropped = load_clean_trades(LOG)
    if not rows:
        print("cm_trades.jsonl is empty.")
        return
    r = np.array([x["ret"] for x in rows])
    n = len(r)
    rng = np.random.default_rng(0)
    ci_low = np.percentile([rng.choice(r, n).mean() for _ in range(3000)], 2.5) if n > 1 else float("nan")
    span_h = (max(x["exit_ts"] for x in rows) - min(x["entry_ts"] for x in rows)) / 3600.0
    by_reason = {k: sum(x["reason"] == k for x in rows) for k in sorted({x["reason"] for x in rows})}

    print(f"FORWARD paper-test (cm_trades.jsonl) — {n} closed trades over {span_h:.1f}h"
          f" ({dropped} blackout-spanning dropped)\n")
    print(f"  mean return   : {r.mean():+.1%}   (in-window benchmark ~{BENCH_MEAN:+.0%})")
    print(f"  bootstrap CI-low: {ci_low:+.1%}   (must be > 0 for a live-positive edge)")
    print(f"  median        : {np.median(r):+.1%}")
    print(f"  win rate      : {(r>0).mean():.1%}   (benchmark ~{BENCH_WIN:.0%})")
    print(f"  total PnL     : {r.sum():+.1f} SOL (1 SOL/trade)")
    print(f"  exits         : {by_reason}")
    print()
    if n < 100:
        print(f"  [UNDERPOWERED] {n} trades — need a few hundred for a stable regime read.")
    else:
        verdict = ("HOLDS — edge generalizes to the new regime" if (ci_low > 0 and r.mean() > 0.10)
                   else "WEAKER/COLLAPSED vs benchmark — likely regime-dependent")
        print(f"  REGIME READ: {verdict}")


if __name__ == "__main__":
    main()
