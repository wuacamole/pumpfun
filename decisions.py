"""
decisions.py
============
Governance machinery for the charter (§9.2, §8). Two concerns:

  * **Decision log** — append entries to ``DECISIONS.md``. Every gate decision,
    threshold change, material modeling choice, and kill criterion gets a
    timestamped, append-only entry. If it isn't in the log, it didn't happen.
  * **Backtest-run counter** — every distinct backtest configuration is
    appended here. The count is reported at the §5.4 gate; a high count raises
    the effective bar (multiple-comparisons honesty, charter §8 G3).

This module is deliberately I/O-only; it never touches the model or the
backtest logic. It is the audit trail, not the experiment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

DECISIONS_PATH = Path(__file__).parent / "DECISIONS.md"
RUNS_PATH = Path(__file__).parent / "backtest_runs.jsonl"


def append_decision(
    kind: str,
    author: str,
    metrics: dict[str, object],
    decision: str,
    rationale: str,
    data_hash: str = "—",
) -> None:
    """Append a decision-log entry to DECISIONS.md.

    ``kind``: gate | threshold | model | kill | run | drift.
    ``decision``: pass | fail | defer | kill | note.
    """
    runs = count_backtest_runs()
    entry = (
        f"[{time.strftime('%Y-%m-%d')}] [{kind}] author={author}\n"
        f"data_hash={data_hash}   backtest_runs_to_date={runs}\n"
        f"metrics={json.dumps(metrics, default=float)}\n"
        f"decision={decision}   rationale={rationale}\n"
    )
    with DECISIONS_PATH.open("a") as f:
        f.write(f"\n{entry}")


def record_backtest_run(config: dict[str, object]) -> int:
    """Record a distinct backtest configuration. Returns the new run count.

    Append a JSONL line with the config + a timestamp. The count is the
    researcher-degrees-of-freedom counter (charter §8 G3): the more configs
    tried, the more the final §5.4 result is suspect. The count is reported
    at the gate.
    """
    line = {"ts": time.time(), "config": config}
    with RUNS_PATH.open("a") as f:
        f.write(json.dumps(line, default=str) + "\n")
    return count_backtest_runs()


def count_backtest_runs() -> int:
    """Return the number of distinct backtest runs recorded to date."""
    if not RUNS_PATH.exists():
        return 0
    n = 0
    with RUNS_PATH.open("r") as f:
        for _ in f:
            n += 1
    return n


def reset_backtest_runs() -> None:
    """Clear the run counter. Use ONLY when starting a fresh research era
    (e.g. after a §9.3 amendment that burns the holdout). Recording this
    action in the decision log is the caller's responsibility.
    """
    if RUNS_PATH.exists():
        RUNS_PATH.unlink()