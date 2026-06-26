"""
base_rate.py
============
Charter §8 drift detection: rolling population graduation rate + gap-rate ceiling.

  * **Base-rate monitor** — the rolling fraction of tokens that graduate. On a
    regime shift (the rate moves outside a band around its trailing mean), flag
    all fitted models as *stale* and require re-fit before further reliance
    (charter I7: "Edge decays. A pass is never permanent").

  * **Gap-rate ceiling** — if the logs-feed gap rate exceeds the pre-registered
    ceiling, the affected window is *low-integrity* and must be excluded from
    training/eval. This prevents a silent hole in the training set from being
    mistaken for a regime shift or a model failure.

Both are pure queries over the SQLite store + the PumpLogsFeed.gaps list. They
never touch the model or the live feed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseRateSnapshot:
    """One observation of the rolling population graduation rate."""

    window_start_ts: float
    window_end_ts: float
    n_tokens: int
    n_graduated: int
    base_rate: float           # n_graduated / n_tokens
    is_stale: bool              # outside the band -> models need re-fit


class BaseRateMonitor:
    """Rolling population graduation rate monitor.

    Computes the graduation rate over a trailing window of the ``tokens``
    table. When the rate moves outside ``stale_band`` around the long-run mean,
    flags models as stale (charter §8: "on a regime shift, flag all fitted
    models as stale and require re-fit").
    """

    def __init__(
        self,
        window_sec: float = 86400.0 * 7,   # 1-week rolling window
        stale_band: float = 0.005,         # ±0.5% around the long-run mean
        min_tokens: int = 50,              # need this many for a stable estimate
    ) -> None:
        self.window_sec = window_sec
        self.stale_band = stale_band
        self.min_tokens = min_tokens
        self.history: deque[float] = deque(maxlen=200)

    def snapshot(self, store_db: str | Path, now: float) -> BaseRateSnapshot:
        """Compute the current base rate over the trailing window."""
        from storage import Store
        s = Store(store_db)
        try:
            cutoff = now - self.window_sec
            rows = s.conn.execute(
                """SELECT outcome FROM tokens
                   WHERE first_seen_ts >= ? AND outcome != 'unknown'""",
                (cutoff,),
            ).fetchall()
            n_tokens = len(rows)
            n_grad = sum(1 for r in rows if r["outcome"] == "graduated")
            rate = n_grad / n_tokens if n_tokens else float("nan")
            self.history.append(rate)
            # Stale if the current rate is outside the band around the long-run
            # mean (and we have enough tokens for a stable estimate).
            is_stale = False
            if len(self.history) >= 3 and n_tokens >= self.min_tokens:
                mean = sum(self.history) / len(self.history)
                if abs(rate - mean) > self.stale_band:
                    is_stale = True
            return BaseRateSnapshot(
                window_start_ts=cutoff,
                window_end_ts=now,
                n_tokens=n_tokens,
                n_graduated=n_grad,
                base_rate=rate,
                is_stale=is_stale,
            )
        finally:
            s.close()


# --- Gap-rate ceiling ------------------------------------------------------
# Pre-registered ceiling for the logs-feed gap rate. If the fraction of slots
# with gaps exceeds this, the affected window is low-integrity and excluded
# from training/eval (charter §8: "mark the affected window as low-integrity
# and exclude it from training/eval"). Pre-register before the §5.4 gate.
GAP_RATE_CEILING: float = 0.05  # 5% of slots may have gaps


def gap_rate(gaps: list[tuple[int, int]], slots_observed: int) -> float:
    """Fraction of slots that had a gap. ``gaps`` is the PumpLogsFeed.gaps list
    of ``(last_slot, reconnect_slot)`` tuples; ``slots_observed`` is the total
    slot span the feed has watched. Returns 0.0 if no slots were observed."""
    if slots_observed <= 0:
        return 0.0
    gapped_slots = sum(b - a for a, b in gaps)
    return min(gapped_slots / slots_observed, 1.0)


def is_low_integrity_window(
    gaps: list[tuple[int, int]],
    slots_observed: int,
    ceiling: float = GAP_RATE_CEILING,
) -> bool:
    """True if the gap rate exceeds the pre-registered ceiling (charter §8)."""
    return gap_rate(gaps, slots_observed) > ceiling