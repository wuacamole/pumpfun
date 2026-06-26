"""
monitor.py
=========
Phase 3.2 — monitoring and alerting. A live dashboard + alert stream:

  * flagged tokens, decisions made
  * running paper PnL (fed by :mod:`paper_trade`)
  * model calibration drift over time (Brier score over a rolling window)
  * edge-decay early-warning (rolling hit rate vs the historical mean)

Two surfaces:
  * :class:`Dashboard`   — a live ``rich`` dashboard (refreshes the terminal).
  * :class:`AlertStream`  — a simple alert sink: prints / writes to a file
    anything that crosses a threshold (decision ENTER, big drawdown,
    calibration drift over X sigma).

Both are designed to run in the same process as the live monitor. They take
events via method calls (no IPC) — the monitor pushes events in. This keeps
the wiring trivial and the cost zero.

If you want a separate-process dashboard later, wrap these in a tiny
websocket/JSON-RPC bridge; the API is already event-shaped.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class DecisionEvent:
    mint: str
    decision: str           # ENTER | WATCH | SKIP
    score: float
    pct_to_graduation: float
    bot_share: float
    sol_per_human: float
    edge: float
    ts: float = field(default_factory=time.time)


@dataclass
class PnlEvent:
    """One closed paper-trade position."""
    mint: str
    status: str              # graduated | stopped | timed_out
    realized_return: float
    bankroll_after: float
    ts: float = field(default_factory=time.time)


@dataclass
class OpenPositionView:
    mint: str
    entry_ts: float
    entry_sol: float
    entry_vsol: float
    current_vsol: float
    age_sec: float
    unrealized_return: float  # rough, based on vSol ratio


class AlertStream:
    """Threshold-based alert sink. Writes human-readable alerts to a file.

    When ``quiet`` is True (dashboard mode), alerts are *only* written to the
    log file — not stdout — so they don't fight the live ``rich`` panel.

    Alerts fired:
      * ENTER decision (always)
      * closed-position PnL on every exit
      * drawdown alert if running bankroll < drawdown_threshold * peak
      * calibration drift alert if rolling Brier > mean + drift_sigma * std

    Charter §8 requires that on calibration drift, "if capital is ever live,
    auto-reduce size to zero pending human review." Wire a ``drift_action``
    callback (e.g. ``broker.halt_trading``) to enforce this automatically.
    """

    def __init__(
        self,
        path: str = "alerts.log",
        drawdown_threshold: float = 0.8,
        drift_sigma: float = 2.0,
        quiet: bool = False,
        drift_action: Callable[[], None] | None = None,
    ) -> None:
        self.path = path
        self.drawdown_threshold = drawdown_threshold
        self.drift_sigma = drift_sigma
        self.quiet = quiet
        self.peak_bankroll = 0.0
        self.brier_history: deque[float] = deque(maxlen=200)
        # Optional callback invoked when calibration drift is detected.
        # Charter §8: "auto-reduce size to zero pending human review."
        self.drift_action = drift_action
        self.drift_fired = False  # latch: fire once until cleared

    def on_decision(self, e: DecisionEvent) -> None:
        if e.decision == "ENTER":
            self._emit(
                f"[ENTER] {e.mint[:8]}.. score={e.score:.1f} "
                f"edge={e.edge:+.1%} bots={e.bot_share:.0%} "
                f"SOL/human={e.sol_per_human:+.3f} "
                f"({e.pct_to_graduation:.1f}% bonded)"
            )

    def on_pnl(self, e: PnlEvent) -> None:
        self._emit(
            f"[CLOSE:{e.status}] {e.mint[:8]}.. return={e.realized_return:+.1%} "
            f"bankroll={e.bankroll_after:.3f}"
        )
        if e.bankroll_after > self.peak_bankroll:
            self.peak_bankroll = e.bankroll_after
        if self.peak_bankroll > 0 and e.bankroll_after < self.peak_bankroll * self.drawdown_threshold:
            self._emit(
                f"[DRAWDOWN] bankroll {e.bankroll_after:.3f} < "
                f"{self.drawdown_threshold:.0%} of peak {self.peak_bankroll:.3f}"
            )

    def on_calibration(self, brier: float) -> None:
        if math.isnan(brier):
            return
        self.brier_history.append(brier)
        if len(self.brier_history) >= 30:
            arr = list(self.brier_history)
            mean = sum(arr) / len(arr)
            var = sum((x - mean) ** 2 for x in arr) / len(arr)
            std = math.sqrt(var)
            if std > 0 and brier > mean + self.drift_sigma * std:
                self._emit(
                    f"[DRIFT] calibration Brier={brier:.4f} > "
                    f"mean+{self.drift_sigma}σ ({mean:.4f}+{self.drift_sigma*std:.4f})"
                )
                # Charter §8: auto-reduce size to zero pending human review.
                # Fire the drift_action callback once (latched) so a sustained
                # drift doesn't spam; clear it via ``clear_drift`` after review.
                if not self.drift_fired and self.drift_action is not None:
                    self.drift_fired = True
                    try:
                        self.drift_action()
                    except Exception as e:
                        self._emit(f"[DRIFT] auto-de-risk callback failed: {e}")

    def clear_drift(self) -> None:
        """Clear the drift latch after human review. Call this when the model
        has been re-fit and calibration is back within the band."""
        self.drift_fired = False

    def _emit(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        if not self.quiet:
            print(line)
        try:
            with open(self.path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


class Dashboard:
    """A live ``rich`` dashboard. Construct, then ``.live()`` context-manages it.

    Push events to update the displayed tables. The panel owns the terminal
    (``screen=True``) so it doesn't compete with ``print()``.

    To keep the panel informative even with no ENTERs, the monitor pushes:
      * :meth:`set_bankroll`     — current broker bankroll, every loop
      * :meth:`set_feed_stats`   — tokens seen / swaps/sec, every loop
      * :meth:`update_watch`     — per-token live score snapshot, every rescore
      * :meth:`set_open_positions` — the broker's current open positions
      * :meth:`on_decision` / :meth:`on_pnl` — event-driven
    """

    def __init__(self, max_decisions: int = 20, max_pnl: int = 20, max_watch: int = 15) -> None:
        self.decisions: deque[DecisionEvent] = deque(maxlen=max_decisions)
        self.pnl: deque[PnlEvent] = deque(maxlen=max_pnl)
        self.watch: dict[str, DecisionEvent] = {}  # mint -> latest event
        self._watch_order: deque[str] = deque(maxlen=max_watch * 4)
        self.open_positions: list[OpenPositionView] = []
        self.bankroll = 0.0
        self.n_open = 0
        self.n_tokens_seen = 0
        self.n_swaps_total = 0
        self.swaps_per_sec = 0.0
        self.start_ts = time.time()
        self.win_rate = float("nan")
        self.mean_return = float("nan")
        self.last_calibration = float("nan")
        self.last_edge = float("nan")
        self._live: Live | None = None

    # ---- event push API ----------------------------------------------------
    def on_decision(self, e: DecisionEvent) -> None:
        self.decisions.append(e)
        self.watch[e.mint] = e
        self._watch_order.append(e.mint)
        self.last_edge = e.edge
        self._refresh()

    def on_pnl(self, e: PnlEvent) -> None:
        self.pnl.append(e)
        self.bankroll = e.bankroll_after
        rets = [p.realized_return for p in self.pnl]
        self.win_rate = sum(1 for r in rets if r > 0) / len(rets) if rets else float("nan")
        self.mean_return = sum(rets) / len(rets) if rets else float("nan")
        self._refresh()

    def set_bankroll(self, b: float) -> None:
        self.bankroll = b
        self._refresh()

    def set_open(self, n: int) -> None:
        self.n_open = n
        self._refresh()

    def set_open_positions(self, positions: list[OpenPositionView]) -> None:
        self.open_positions = positions
        self._refresh()

    def set_feed_stats(self, n_tokens_seen: int, n_swaps_total: int, swaps_per_sec: float) -> None:
        self.n_tokens_seen = n_tokens_seen
        self.n_swaps_total = n_swaps_total
        self.swaps_per_sec = swaps_per_sec
        self._refresh()

    def on_calibration(self, brier: float) -> None:
        self.last_calibration = brier
        self._refresh()

    def tick(self) -> None:
        """Force a re-render. Call this from a timer even with no events."""
        self._refresh()

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    # ---- rendering --------------------------------------------------------
    def _render(self) -> Panel:
        uptime = max(time.time() - self.start_ts, 1e-3)
        stats = Table(show_header=False, box=None, padding=(0, 2))
        stats.add_row("bankroll", f"{self.bankroll:.3f} SOL")
        stats.add_row("open positions", str(self.n_open))
        stats.add_row("win rate", f"{self.win_rate:.1%}" if not math.isnan(self.win_rate) else "—")
        stats.add_row("mean return", f"{self.mean_return:+.1%}" if not math.isnan(self.mean_return) else "—")
        stats.add_row("last edge", f"{self.last_edge:+.1%}" if not math.isnan(self.last_edge) else "—")
        stats.add_row("calibration Brier", f"{self.last_calibration:.4f}" if not math.isnan(self.last_calibration) else "—")

        feed = Table(show_header=False, box=None, padding=(0, 2))
        feed.add_row("uptime", f"{uptime/60:.1f} min")
        feed.add_row("tokens seen", str(self.n_tokens_seen))
        feed.add_row("swaps total", str(self.n_swaps_total))
        feed.add_row("swaps/sec", f"{self.swaps_per_sec:.1f}")

        top = Table.grid(expand=True)
        top.add_column()
        top.add_column()
        top.add_row(stats, feed)

        # Watch list: tokens currently being scored (most recent first, cap 15).
        watch = Table(title="Watching", header_style="bold cyan")
        watch.add_column("mint", width=10)
        watch.add_column("dec", width=5)
        watch.add_column("score", width=6, justify="right")
        watch.add_column("%grad", width=6, justify="right")
        watch.add_column("bots", width=5, justify="right")
        watch.add_column("edge", width=7, justify="right")
        watch_mints = list(self._watch_order)[::-1]
        seen_mints: set[str] = set()
        for mint in watch_mints:
            if mint in seen_mints:
                continue
            seen_mints.add(mint)
            if len(seen_mints) > 15:
                break
            d = self.watch.get(mint)
            if d is None:
                continue
            color = "green" if d.decision == "ENTER" else ("yellow" if d.decision == "WATCH" else "dim")
            watch.add_row(
                f"{d.mint[:8]}..",
                Text(d.decision, style=color),
                f"{d.score:.1f}",
                f"{d.pct_to_graduation:.1f}",
                f"{d.bot_share:.0%}",
                f"{d.edge:+.1%}",
            )

        # Open positions.
        pos = Table(title="Open positions", header_style="bold cyan")
        pos.add_column("mint", width=10)
        pos.add_column("age", width=7, justify="right")
        pos.add_column("entry", width=7, justify="right")
        pos.add_column("vSol", width=6, justify="right")
        pos.add_column("unrealized", width=10, justify="right")
        for op in self.open_positions[:15]:
            color = "green" if op.unrealized_return >= 0 else "red"
            pos.add_row(
                f"{op.mint[:8]}..",
                f"{op.age_sec/60:.1f}m",
                f"{op.entry_sol:.2f}",
                f"{op.current_vsol:.0f}",
                Text(f"{op.unrealized_return:+.1%}", style=color),
            )

        # Recent decisions.
        dec = Table(title="Recent decisions", header_style="bold cyan")
        dec.add_column("time", width=8)
        dec.add_column("mint", width=10)
        dec.add_column("dec", width=5)
        dec.add_column("score", width=6, justify="right")
        dec.add_column("edge", width=7, justify="right")
        dec.add_column("bots", width=5, justify="right")
        for d in list(self.decisions)[::-1]:
            dec.add_row(
                time.strftime("%H:%M:%S", time.localtime(d.ts)),
                f"{d.mint[:8]}..",
                d.decision,
                f"{d.score:.1f}",
                f"{d.edge:+.1%}",
                f"{d.bot_share:.0%}",
            )

        # Recent closes.
        pnl = Table(title="Recent closes", header_style="bold cyan")
        pnl.add_column("time", width=8)
        pnl.add_column("mint", width=10)
        pnl.add_column("status", width=10)
        pnl.add_column("return", width=8, justify="right")
        pnl.add_column("bankroll", width=9, justify="right")
        for cl in list(self.pnl)[::-1]:
            color = "green" if cl.realized_return > 0 else "red"
            pnl.add_row(
                time.strftime("%H:%M:%S", time.localtime(cl.ts)),
                f"{cl.mint[:8]}..",
                cl.status,
                Text(f"{cl.realized_return:+.1%}", style=color),
                f"{cl.bankroll_after:.3f}",
            )

        body = Table.grid(expand=True)
        body.add_row(top)
        watch_pos = Table.grid(expand=True)
        watch_pos.add_row(watch, pos)
        dec_pnl = Table.grid(expand=True)
        dec_pnl.add_row(dec, pnl)
        body.add_row(watch_pos)
        body.add_row(dec_pnl)
        title = "pump.fun conviction — paper trading" if self.n_open or self.pnl else "pump.fun conviction — live monitor"
        return Panel(body, title=title, border_style="cyan")

    def live(self) -> Live:
        # screen=True owns the terminal — prints can't scroll the panel away.
        self._live = Live(self._render(), refresh_per_second=2, screen=True, transient=False)
        return self._live


def write_metrics_snapshot(path: str, dashboard: Dashboard, alert: AlertStream) -> None:
    """Append a metrics line to a JSONL file for later plotting / postmortem."""
    snap = {
        "ts": time.time(),
        "bankroll": dashboard.bankroll,
        "n_open": dashboard.n_open,
        "win_rate": dashboard.win_rate,
        "mean_return": dashboard.mean_return,
        "last_edge": dashboard.last_edge,
        "last_calibration": dashboard.last_calibration,
        "n_brier_samples": len(alert.brier_history),
    }
    with open(path, "a") as f:
        f.write(json.dumps(snap, default=float) + "\n")