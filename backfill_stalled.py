"""Backfill full-population `stalled` labels — close the stalled-labelling
selection bias (Charter I3).

Problem: historically, `stalled` was written ONLY by the paper broker when an
*entered* position timed out (live_monitor.py:606). So the resolved negative
class was the paper-entered subset — a selected sample, not the population. Most
real stalls sat in `outcome='unknown'`, inflating the apparent graduation rate
and biasing every model fit.

This script labels the settled, never-graduated, traded tokens that the monitor
would now label `stalled` going forward (the live `_gc` fix). A token qualifies
when:
  * outcome is still 'unknown' (non-clobbering: graduated/already-resolved win),
  * it has >= 1 swap (zero-swap create-only tokens never traded — left 'unknown'
    as a distinct category; they aren't a trading outcome and aren't featurizable),
  * its last swap is older than STALE_INACTIVITY_SEC before the latest swap in
    the DB (outcome settled — matches the live eviction threshold), and
  * it never crossed the graduation vSol (defensive; unknowns shouldn't have).

outcome_ts is set to last_swap_ts + STALE_INACTIVITY_SEC — the earliest moment
the stall is confirmable — for reproducible, no-lookahead creator-history joins.

Reversible: backfilled stalls have NO paper_trades row, so
  UPDATE tokens SET outcome='unknown', outcome_ts=NULL
  WHERE outcome='stalled' AND mint NOT IN (SELECT mint FROM paper_trades);
undoes exactly this set. An audit JSONL of affected mints is also written.

Run (preview):  uv run python backfill_stalled.py
Run (apply):    uv run python backfill_stalled.py --apply
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

# Inactivity window after which a non-graduated token is considered stalled.
# Mirrors live_monitor.STALE_AFTER_SEC (900s / 15 min) — kept as a local constant
# to avoid importing live_monitor (which pulls websockets/asyncio). If you change
# STALE_AFTER_SEC there, change it here too.
STALE_INACTIVITY_SEC = 900.0

DB = "pumpfun.db"

# Tokens qualifying for a full-population stall label. Single source of truth for
# both the preview count and the apply step so they can't drift.
_CANDIDATE_SQL = """
WITH mx AS (SELECT MAX(ts) AS m FROM swaps),
     last_swap AS (SELECT mint, MAX(ts) AS lt, MAX(vsol_after) AS peak_vsol
                   FROM swaps GROUP BY mint)
SELECT t.mint, ls.lt AS last_ts
FROM tokens t
JOIN last_swap ls ON ls.mint = t.mint
CROSS JOIN mx
WHERE t.outcome = 'unknown'
  AND ls.lt <= mx.m - :stale
  AND ls.peak_vsol < :grad_vsol
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--apply", action="store_true",
                    help="write the labels (default: preview only)")
    args = ap.parse_args()

    # Graduation vSol from the live constants (falls back to historical default).
    from config import get_cached_constants
    grad_vsol = get_cached_constants().v_sol_graduation

    # isolation_level=None => autocommit; we control transactions with explicit
    # BEGIN/COMMIT per batch. timeout=30 sets busy_timeout so we wait for, rather
    # than error on, the running monitor's write lock.
    conn = sqlite3.connect(args.db, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row

    params = {"stale": STALE_INACTIVITY_SEC, "grad_vsol": grad_vsol}
    rows = conn.execute(_CANDIDATE_SQL, params).fetchall()
    n = len(rows)

    before = dict(conn.execute(
        "SELECT outcome, COUNT(*) c FROM tokens GROUP BY outcome").fetchall())
    print(f"DB: {args.db}   graduation_vsol={grad_vsol:.2f}   "
          f"stale_inactivity={STALE_INACTIVITY_SEC:.0f}s")
    print(f"current outcomes: {before}")
    print(f"candidates to label stalled (settled, traded, never graduated, "
          f"currently unknown): {n}")
    resolved_before = before.get("graduated", 0) + before.get("stalled", 0)
    resolved_after = resolved_before + n
    if resolved_before:
        print(f"resolved set: {resolved_before} -> {resolved_after}   "
              f"grad rate: {before.get('graduated',0)/resolved_before:.1%} -> "
              f"{before.get('graduated',0)/resolved_after:.1%}  "
              f"(de-biased toward the true base rate)")

    if not args.apply:
        print("\n[dry-run] no changes written. Re-run with --apply to commit.")
        conn.close()
        return

    # Audit log for reversibility.
    ts = int(time.time())
    audit = Path(f"backfill_stalled_{ts}.jsonl")
    now = time.time()
    # Commit in small batches so each write-lock hold is brief — the live monitor
    # is a concurrent writer; short transactions let its writes interleave (and
    # its busy_timeout absorbs the rest). One big 12k-row txn would hold the lock
    # for seconds.
    BATCH = 500
    written = 0
    with audit.open("w") as f:
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            conn.execute("BEGIN")
            for r in chunk:
                outcome_ts = float(r["last_ts"]) + STALE_INACTIVITY_SEC
                # only_if_unknown semantics, inline (script owns the connection):
                conn.execute(
                    "UPDATE tokens SET outcome='stalled', outcome_ts=?, updated_at=? "
                    "WHERE mint=? AND outcome='unknown'",
                    (outcome_ts, now, r["mint"]),
                )
                f.write(json.dumps({"mint": r["mint"], "outcome_ts": outcome_ts}) + "\n")
            conn.execute("COMMIT")
            written += len(chunk)
        print(f"  ...labelled {written}/{len(rows)}")

    after = dict(conn.execute(
        "SELECT outcome, COUNT(*) c FROM tokens GROUP BY outcome").fetchall())
    print(f"\napplied. outcomes now: {after}")
    print(f"audit log (for reversal): {audit}")
    conn.close()


if __name__ == "__main__":
    main()
