"""
paper_trade.py
==============
Phase 3.1 — paper-trading mode. Run the live system end-to-end, "entering" and
"exiting" positions on paper, tracking simulated PnL with realistic frictions:
  * fees        — 1.25% per swap (matches ``conviction_engine.FEE_RATE``)
  * slippage    — configurable, default 2%
  * MEV penalty — sandwich penalty on entry, default 1% (pessimistic)

Why this exists
---------------
A backtest can hide execution reality. Paper trading on *live* data for weeks
reveals whether the edge survives contact with fees, slippage, and the bots
that will trade against you. If it doesn't survive paper trading, it won't
survive real money.

How to use
----------
This module is a *simulated broker*. The live monitor calls :meth:`PaperBroker.on_enter`
when the engine says ENTER, and :meth:`PaperBroker.on_exit` when the token
graduates (auto-sell) or hits a stop / time stop. The broker records every fill
with realistic frictions and keeps a running PnL + position log.

It is deliberately NOT wired to the websocket loop in ``live_monitor.py`` —
that's a one-line hook. The point is to have a separate, inspectable object you
can also drive from a replay of stored swaps (so you can paper-trade historical
tokens you didn't catch live).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from config import fee_at, get_cached_constants
from conviction_engine import Swap

if TYPE_CHECKING:
    from storage import Store


@dataclass
class Fill:
    mint: str
    side: str               # "buy" | "sell"
    ts: float
    sol_amount: float       # SOL notional of the fill (pre-cost)
    effective_price: float  # price-per-unit implied, post slippage
    fee_sol: float
    slippage_sol: float
    mev_sol: float          # only on entry
    net_sol: float          # what actually left/entered the wallet


@dataclass
class Position:
    mint: str
    entry_ts: float
    entry_sol: float        # gross SOL deployed
    entry_cost_sol: float   # fees + slippage + MEV paid at entry
    units_held: float       # token units bought
    entry_vsol: float
    trade_id: int | None = None  # SQLite paper_trades.id (None if not persisted)
    exit_ts: float | None = None
    exit_sol: float | None = None
    exit_cost_sol: float | None = None
    realized_return: float | None = None
    status: str = "open"   # open | graduated | stopped | timed_out
    fills: list[Fill] = field(default_factory=list)


@dataclass
class PaperBroker:
    """Simulated broker that applies fees + slippage + MEV penalty to fills.

    Position sizing is fractional: each entry risks ``size_fraction`` of the
    current bankroll (default 2% — conservative for a thin, negative-sum market).

    Persistence
    -----------
    If a ``Store`` is attached, every entry/exit/fill is written to the
    ``paper_trades`` / ``paper_fills`` tables (the durable historical record)
    and the in-flight state (bankroll + open positions) is snapshotted to the
    ``paper_state`` KV bag on every change. On restart, :func:`reload_state`
    restores bankroll and reattaches to open positions by their trade ids, so
    a crash or restart never wipes your paper PnL or loses in-flight trades.
    """

    bankroll: float = 10.0
    size_fraction: float = 0.02
    # Fee/slippage/MEV defaults are only used when a fill's ts is None or when
    # fee_at(ts) can't be evaluated. Real fills call fee_at(ts) so the schedule
    # that was live at the trade's timestamp is applied — a backtest across a
    # fee-regime change stays honest. See config.FeeSchedule.
    fee_rate: float = 0.0125   # fallback only; overridden by fee_at(ts) per fill
    slippage: float = 0.02     # fallback only
    mev_penalty: float = 0.01  # fallback only
    time_stop_sec: float = 1800.0     # exit if not graduated in 30 min
    stop_loss_fraction: float = 0.5  # exit if the bonding curve loses 50% of vSol
    # Default graduation payoff multiple (used when no logged PumpSwap price is
    # available — the pre-data-collection state). Conservative: the paper
    # implies more, but a flat 5x is the honest placeholder until real
    # post-graduation prices are logged. Charter invariant I2 (post-cost
    # realized PnL of the actual strategy) requires the logged price path.
    default_graduation_multiple: float = 5.0
    # Callable: mint -> (price_sol_per_unit, payoff_multiple) | None. When set
    # (by the live monitor or backtest), on_graduated uses the real PumpSwap
    # first-trade price instead of the flat default. Returns None if no logged
    # price exists for this mint yet.
    graduation_price_lookup: Any = None
    positions: dict[str, Position] = field(default_factory=dict)
    closed: list[Position] = field(default_factory=list)
    fills_log: list[Fill] = field(default_factory=list)
    store: Store | None = None  # attach to enable persistence

    def _schedule(self, ts: float) -> tuple[float, float, float]:
        """Return (fee_rate, slippage, mev_penalty) in effect at ts."""
        s = fee_at(ts)
        return s.curve_fee_rate, s.slippage, s.mev_penalty

    def _graduation_vsol(self) -> float:
        """The graduation vSol threshold (live, from the on-chain cache)."""
        return get_cached_constants().v_sol_graduation

    # ---- persistence ------------------------------------------------------
    def attach_store(self, store: Store, reload: bool = True) -> None:
        """Attach a Store for persistence. If ``reload``, restore bankroll +
        open positions from a previous run."""
        self.store = store
        if reload:
            self.reload_state()

    def halt_trading(self) -> None:
        """Charter §8 auto-de-risk: set size_fraction to zero so no new
        positions are opened. Existing open positions are still managed
        (graduation auto-sell, time/loss stops) — capital is not stranded.
        Requires a human to call :meth:`resume_trading` after review."""
        self.size_fraction = 0.0

    def resume_trading(self, size_fraction: float) -> None:
        """Resume trading after a human clears a drift alert. Pass the new
        size_fraction (may be smaller than the pre-halt value)."""
        self.size_fraction = size_fraction

    def reload_state(self) -> None:
        """Restore bankroll + open positions from the Store. Called on startup
        so a restart picks up where the previous run left off."""
        if self.store is None:
            return
        bankroll = self.store.get_paper_state("bankroll")
        if isinstance(bankroll, (int, float)):
            self.bankroll = float(bankroll)
        # Reattach open positions by their trade_ids.
        ids = self.store.open_paper_trade_ids()
        for mint, trade_id in ids.items():
            # We only persisted the entry-side economics on open; the exit fields
            # are None until the position closes. Reconstruct a minimal Position
            # so check_exits can still manage it.
            row = self.store.conn.execute(
                "SELECT entry_ts, entry_sol, entry_cost_sol, units_held, entry_vsol "
                "FROM paper_trades WHERE id = ?", (trade_id,)
            ).fetchone()
            if row is None:
                continue
            self.positions[mint] = Position(
                mint=mint,
                entry_ts=row["entry_ts"],
                entry_sol=row["entry_sol"],
                entry_cost_sol=row["entry_cost_sol"],
                units_held=row["units_held"],
                entry_vsol=row["entry_vsol"],
                trade_id=trade_id,
            )

    def _snapshot_state(self) -> None:
        """Persist the in-flight state (bankroll + open positions) to the KV bag.
        Called after every entry/exit so a restart never loses track."""
        if self.store is None:
            return
        self.store.set_paper_state("bankroll", self.bankroll)
        self.store.set_paper_state("open_positions", [
            {"mint": p.mint, "trade_id": p.trade_id} for p in self.positions.values()
        ])

    def _persist_fill(self, pos: Position, fill: Fill) -> None:
        if self.store is None or pos.trade_id is None:
            return
        self.store.log_paper_fill(
            trade_id=pos.trade_id, mint=fill.mint, side=fill.side, ts=fill.ts,
            sol_amount=fill.sol_amount, effective_price=fill.effective_price,
            fee_sol=fill.fee_sol, slippage_sol=fill.slippage_sol,
            mev_sol=fill.mev_sol, net_sol=fill.net_sol,
        )

    # ---- entry -------------------------------------------------------------
    def on_enter(self, mint: str, ts: float, vsol: float, swap: Swap | None = None) -> Position | None:
        """Open a paper position. Returns the Position or None if rejected.

        Rejection reasons: already in a position for this mint, bankroll
        below the minimum notional, or already graduated.
        """
        if mint in self.positions:
            return None
        grad_vsol = self._graduation_vsol()
        if vsol >= grad_vsol:
            return None  # too late — already graduated
        notional = max(self.bankroll * self.size_fraction, 0.0)
        if notional <= 0 or notional > self.bankroll:
            return None

        # Apply entry frictions at the schedule in effect at this trade's ts.
        fee_rate, slip_rate, mev_rate = self._schedule(ts)
        fee = notional * fee_rate
        slip = notional * slip_rate
        mev = notional * mev_rate
        net = notional - fee - slip - mev
        if net <= 0:
            return None

        # Token units bought — we don't know the curve price exactly without
        # the bonding-curve state, so we approximate "units" as a notional
        # fraction of vSol. This is only used for the exit accounting; the
        # realized return is computed from the bonding-curve payoff model.
        units = net / max(vsol, 1e-6)
        fill = Fill(
            mint=mint, side="buy", ts=ts, sol_amount=notional,
            effective_price=notional / max(units, 1e-9),
            fee_sol=fee, slippage_sol=slip, mev_sol=mev, net_sol=-net,
        )
        self.fills_log.append(fill)
        self.bankroll -= notional  # capital deployed; costs already netted
        pos = Position(
            mint=mint, entry_ts=ts, entry_sol=notional,
            entry_cost_sol=fee + slip + mev, units_held=units, entry_vsol=vsol,
            fills=[fill],
        )
        # Persist: open the trade row + log the entry fill + snapshot state.
        if self.store is not None:
            pos.trade_id = self.store.open_paper_trade(
                mint=mint, entry_ts=ts, entry_sol=notional,
                entry_cost_sol=fee + slip + mev, units_held=units, entry_vsol=vsol,
            )
            self._persist_fill(pos, fill)
            self._snapshot_state()
        self.positions[mint] = pos
        return pos

    # ---- exit --------------------------------------------------------------
    def on_graduated(self, mint: str, ts: float) -> Position | None:
        """Auto-sell on graduation. Payoff is the real PumpSwap first-trade
        price when a logged graduation price is available (charter I2: post-cost
        realized PnL of the actual strategy); otherwise falls back to the
        conservative flat ``default_graduation_multiple`` (5x), the
        pre-data-collection placeholder.

        Exit fee is the PumpSwap fee (post-graduation the token trades on
        PumpSwap, not the bonding curve), looked up at the exit timestamp.
        """
        pos = self.positions.pop(mint, None)
        if pos is None:
            return None
        # Resolve the exit multiple: logged PumpSwap price > flat default.
        multiple = self.default_graduation_multiple
        if self.graduation_price_lookup is not None:
            try:
                gp = self.graduation_price_lookup(mint)
                if gp is not None:
                    price, logged_multiple = gp
                    if logged_multiple is not None and logged_multiple > 0:
                        multiple = logged_multiple
            except Exception:
                pass  # fall back to default — never block the exit on a lookup error
        gross_exit = pos.entry_sol * multiple
        s = fee_at(ts)
        fee_rate = s.pumpswap_fee_rate   # graduated token trades on PumpSwap
        slip_rate = s.slippage
        fee = gross_exit * fee_rate
        slip = gross_exit * slip_rate
        net = gross_exit - fee - slip
        fill = Fill(
            mint=mint, side="sell", ts=ts, sol_amount=gross_exit,
            effective_price=gross_exit / max(pos.units_held, 1e-9),
            fee_sol=fee, slippage_sol=slip, mev_sol=0.0, net_sol=net,
        )
        self.fills_log.append(fill)
        self.bankroll += net
        pos.exit_ts = ts
        pos.exit_sol = net
        pos.exit_cost_sol = fee + slip
        pos.realized_return = (net - pos.entry_sol) / pos.entry_sol
        pos.status = "graduated"
        pos.fills.append(fill)
        self.closed.append(pos)
        self._persist_close(pos, fill)
        return pos

    def on_stop(self, mint: str, ts: float, current_vsol: float) -> Position | None:
        """Time / loss stop. Exit at the bonding-curve's current implied value.

        Exit fee is the bonding-curve fee (the token is still on the curve,
        not yet graduated to PumpSwap), looked up at the exit timestamp.
        """
        pos = self.positions.pop(mint, None)
        if pos is None:
            return None
        # Implied current value of the position from vSol ratio.
        ratio = current_vsol / max(pos.entry_vsol, 1e-6)
        ratio = max(ratio, 0.05)   # floor at 5% of entry value (rug-ish)
        gross_exit = pos.entry_sol * ratio
        s = fee_at(ts)
        fee = gross_exit * s.curve_fee_rate
        slip = gross_exit * s.slippage
        net = gross_exit - fee - slip
        fill = Fill(
            mint=mint, side="sell", ts=ts, sol_amount=gross_exit,
            effective_price=gross_exit / max(pos.units_held, 1e-9),
            fee_sol=fee, slippage_sol=slip, mev_sol=0.0, net_sol=net,
        )
        self.fills_log.append(fill)
        self.bankroll += net
        pos.exit_ts = ts
        pos.exit_sol = net
        pos.exit_cost_sol = fee + slip
        pos.realized_return = (net - pos.entry_sol) / pos.entry_sol
        if ts - pos.entry_ts >= self.time_stop_sec:
            pos.status = "timed_out"
        else:
            pos.status = "stopped"
        pos.fills.append(fill)
        self.closed.append(pos)
        self._persist_close(pos, fill)
        return pos

    def _persist_close(self, pos: Position, fill: Fill) -> None:
        """Persist the close + exit fill + snapshot in-flight state."""
        if self.store is None or pos.trade_id is None:
            return
        self.store.close_paper_trade(
            trade_id=pos.trade_id, exit_ts=pos.exit_ts or 0.0,
            exit_sol=pos.exit_sol or 0.0, exit_cost_sol=pos.exit_cost_sol or 0.0,
            realized_return=pos.realized_return or 0.0, status=pos.status,
        )
        self._persist_fill(pos, fill)
        self._snapshot_state()

    # ---- periodic check ---------------------------------------------------
    def check_exits(self, now: float, vsol_by_mint: dict[str, float]) -> list[Position]:
        """Call once per loop: apply time / loss stops to open positions.

        ``vsol_by_mint`` is the latest vSol for each open-position mint. Returns
        the positions that were closed by this call.
        """
        closed_now: list[Position] = []
        grad_vsol = self._graduation_vsol()
        for mint in list(self.positions.keys()):
            pos = self.positions[mint]
            vsol = vsol_by_mint.get(mint, pos.entry_vsol)
            # Graduated — handled by on_graduated normally, but guard.
            if vsol >= grad_vsol:
                c = self.on_graduated(mint, now)
                if c:
                    closed_now.append(c)
                continue
            # Time stop.
            if now - pos.entry_ts >= self.time_stop_sec:
                c = self.on_stop(mint, now, vsol)
                if c:
                    closed_now.append(c)
                continue
            # Loss stop: bonding curve lost > stop_loss_fraction of its vSol.
            if vsol < pos.entry_vsol * (1.0 - self.stop_loss_fraction):
                c = self.on_stop(mint, now, vsol)
                if c:
                    closed_now.append(c)
        return closed_now

    # ---- summary ----------------------------------------------------------
    def summary(self) -> dict[str, object]:
        n_open = len(self.positions)
        n_closed = len(self.closed)
        n_grad = sum(1 for p in self.closed if p.status == "graduated")
        n_stop = sum(1 for p in self.closed if p.status == "stopped")
        n_to = sum(1 for p in self.closed if p.status == "timed_out")
        rets = [p.realized_return for p in self.closed if p.realized_return is not None]
        mean_ret = sum(rets) / len(rets) if rets else float("nan")
        win_rate = sum(1 for r in rets if r > 0) / len(rets) if rets else float("nan")
        return {
            "bankroll": round(self.bankroll, 4),
            "n_open": n_open,
            "n_closed": n_closed,
            "n_graduated": n_grad,
            "n_stopped": n_stop,
            "n_timed_out": n_to,
            "mean_return": mean_ret,
            "win_rate": win_rate,
            "total_fees_slippage_mev": round(sum(
                f.fee_sol + f.slippage_sol + f.mev_sol for f in self.fills_log
            ), 4),
        }