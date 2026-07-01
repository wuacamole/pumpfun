"""Shared gap filter for forward-test trade records (cm_trades.jsonl).

Under interval (duty-cycled) firehose capture the feed is only connected for
ON_SEC of every PERIOD; a trade whose lifetime spans an OFF blackout has a
path-blind, unreliable exit (we never saw the price action that closed it) and
must be excluded from forward-edge statistics. A trade is CLEAN iff:

  * it closed before interval mode began (continuous-capture era), OR
  * it is fully contained in one logged ON window (interval_windows.jsonl,
    written by the monitor's interval runner at the end of each window), OR
  * pre-window-log fallback: it closed in the interval era outside any logged
    window AND its hold fit inside the 90s ON window in effect before the
    boundary log existed.

Used by _chk.py, cm_forward_report.py and cm_auto_report.py so the headline
numbers can never silently include blackout-spanning exits.
"""
from __future__ import annotations

import json
from pathlib import Path

INTERVAL_START_TS = 1782768720.0   # interval-mode cutover (2026-06-29 21:32 UTC)
PRELOG_ON_SEC = 90.0               # ON_SEC in effect before window logging existed
TRADES_PATH = Path("cm_trades.jsonl")
WINDOWS_PATH = Path("interval_windows.jsonl")


def _load_windows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_clean(trade: dict, windows: list[dict]) -> bool:
    t_in, t_out = trade["entry_ts"], trade["exit_ts"]
    if t_out < INTERVAL_START_TS:
        return True                                    # continuous-capture era
    for w in windows:                                  # fully inside one ON window
        if w["start"] <= t_in and t_out <= w["end"]:
            return True
    if not any(w["start"] <= t_out <= w["end"] for w in windows):
        return (t_out - t_in) <= PRELOG_ON_SEC         # pre-log interval fallback
    return False                                       # spanned a blackout -> drop


def load_clean_trades(trades_path: Path = TRADES_PATH,
                      windows_path: Path = WINDOWS_PATH) -> tuple[list[dict], int]:
    """Read the forward-trade log and apply the gap filter.

    Returns (clean_rows, n_dropped). Missing/empty log -> ([], 0)."""
    if not trades_path.exists():
        return [], 0
    rows = [json.loads(line) for line in trades_path.read_text().splitlines() if line.strip()]
    windows = _load_windows(windows_path)
    clean = [t for t in rows if is_clean(t, windows)]
    return clean, len(rows) - len(clean)
