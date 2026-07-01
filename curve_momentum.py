"""Curve-momentum paper strategy — the only edge candidate that survived the validation
gauntlet (DECISIONS [model] curve-momentum / -oos / -execstress / -drawdown / -modelsel).

STRATEGY (all params are validated/OOS-selected; scale-out added 2026-06-29):
  * Enter a token the FIRST time it reaches ENTRY_VSOL, IFF it was caught clean
    (first observed vSol < CATCH_VSOL) and the graduation model ranks it in the top
    ~50% (P(grad) >= MODEL_SCORE_MIN, the historical entry-zone median).
  * PARTIAL SCALE-OUT exit (raises win 51%->63%, friction-robust to 5% haircut, OOS-stable
    across time-thirds; see stress_scaleout.py / refine_exit_entry.py):
      - Bank FIRST_FRAC of the position the first time price reaches FIRST_MULT*entry.
      - The remaining RUNNER then exits at the FIRST of: take-profit (TP_MULT*entry),
        BREAKEVEN stop (price falls back to entry — the runner can no longer lose),
        graduation (force-exit on the curve), or the safety time-stop.
      - If price hits the hard SL (SL_MULT*entry) BEFORE the first tranche banks, the
        WHOLE position stops there.
    Why win-rate jumps: the trade is green the instant the first tranche banks, and the
    runner is breakeven-protected. Exit is ON THE BONDING CURVE (liquid — sidesteps the
    [kill] liquidity wall).
  * Realized return uses EXACT constant-product entry+exit impact per tranche (slippage
    scales with size — capacity ~2 SOL/position) plus curve fee, MEV, adverse haircut.

PAPER ONLY, separate object from PaperBroker (killed graduation strategy). No capital.
The dominant UNVALIDATED risk is regime generalization — this exists to forward paper-test
it. Entry velocity (vSol/s caught->entry) is LOGGED per trade but NOT gated (a candidate
further lever held out to avoid overfitting; evaluate on forward data before committing).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path

from config import fee_at, get_cached_constants
from model import Swap, curve_price_per_token

ENTRY_VSOL = 38.0          # enter the first time vSol reaches this (raised 36->38
                           # 2026-06-29: floor-36 forward test converged inconclusive-at-
                           # full-power (n=303, CI-low -2.8%); floor=38 trims weak low-
                           # momentum entries. Validated on 26k-tok backtest (CI-low
                           # +7.5%->+9.2%) and the floor-36 forward sample (CI-low -2.8%->
                           # +1.1%, win 51%->55%). See cm_trades_floor36.jsonl for prior run.
CATCH_VSOL = 32.0          # only tokens caught before the entry zone (genuine early history)
# Scale-out geometry (validated 2026-06-29): bank 50% at 1.3x, runner -> 1.7x / breakeven.
FIRST_MULT = 1.3           # bank the first tranche when price reaches this x entry
FIRST_FRAC = 0.5           # fraction of the position banked at FIRST_MULT
TP_MULT = 1.7              # runner take-profit at 1.7x spot
SL_MULT = 0.2              # hard stop-loss (only active before the first tranche banks)
MODEL_SCORE_MIN = 0.115    # tightened 0.070 -> 0.115 on 2026-07-02: the pre-committed
                           # trigger ("tighten to ~0.115 only if the forward test confirms
                           # the live lift") FIRED. Forward true-OOS n=940: the marginal
                           # band [0.070,0.115) is dead weight (win 55.1%, mean +1.4%,
                           # CI-low -7.7%) while >=0.115 is win 59.9% / +10.7% / CI-low
                           # +5.0%. Replay agrees (time-ordered test slice: win 62.0% vs
                           # 60.1%, CI-low +11.6% vs +9.2%; 6-window regime check equal-or-
                           # better in 5/6, win never lower). Keeps ~70% of entries.
                           # History: 0.070 was the de-skewed model's top-50% cut set
                           # 2026-06-29 (see git for the full recalibration note).
MEV_RATE = 0.01            # sandwich/priority assumption on entry
ADVERSE_HAIRCUT = 0.0      # extra per-side fill loss; 0 = base (stress separately)
MAX_HOLD_SEC = 3600.0      # safety time-stop: exit at current price if held this long
SMART_SELL_EXIT = 2        # DEPRECATED override threshold, retained for the unit test of
                           # the (now log-only) smart-sell counter. The force-exit was
                           # DEMOTED to log-only 2026-06-30: regime_test.py (walk-forward)
                           # showed it was a mild net DRAG, not a hedge (e.g. W2
                           # +11.5%->+10.0%), and it fired ~0x live. We still count distinct
                           # smart sellers per trade (smart_sells) but no longer exit on them.
                           # Smart list = smart_wallets.json (smart_wallets.py; OOS win-lift
                           # +24.8%, persistence-checked). Empty set -> feature inert.
SMART_WALLETS_PATH = "smart_wallets.json"


def load_smart_wallets(path: str = SMART_WALLETS_PATH) -> set[str]:
    """Load the ranked smart-wallet set built by smart_wallets.py. Returns an empty
    set (feature inert) if the file is absent, so the strategy never hard-depends on it."""
    p = Path(path)
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()).get("wallets", []))
    except Exception:
        return set()


@dataclass
class CMPosition:
    mint: str
    entry_ts: float
    entry_vsol: float
    size_sol: float
    model_score: float
    first_reserve: float       # bank the first tranche at/above this vSol
    tp_reserve: float          # runner take-profit
    sl_reserve: float          # hard stop (pre-bank only)
    last_pregrad_vsol: float
    velocity: float            # vSol/s from caught swap to entry (logged, not gated)
    smart_buys: int = 0        # distinct smart wallets that bought before entry (confidence)
    banked: bool = False       # has the first tranche been sold?
    banked_ret: float = 0.0    # realized return of the banked tranche (at its own size)
    smart_sellers: set = field(default_factory=set)   # distinct smart wallets that have sold


@dataclass
class CMTrade:
    mint: str
    entry_ts: float
    exit_ts: float
    entry_vsol: float
    exit_vsol: float           # the RUNNER's exit price (or full-size stop if pre-bank)
    size_sol: float
    ret: float                 # blended position return
    reason: str                # "tp" | "be" | "sl" | "grad" | "time" | "evict" | "smart_exit"
    model_score: float
    velocity: float
    smart_buys: int            # distinct smart wallets that bought before entry
    smart_sells: int           # distinct smart wallets that sold during the hold


class CurveMomentumPaper:
    def __init__(self, size_sol: float = 1.0, score_min: float = MODEL_SCORE_MIN,
                 tp_mult: float = TP_MULT, sl_mult: float = SL_MULT,
                 first_mult: float = FIRST_MULT, first_frac: float = FIRST_FRAC,
                 mev: float = MEV_RATE, haircut: float = ADVERSE_HAIRCUT, pc=None,
                 smart_wallets: set[str] | None = None):
        self.pc = pc or get_cached_constants()
        self.smart = smart_wallets if smart_wallets is not None else set()
        self.smart_buyers: dict[str, set] = {}   # per-mint distinct smart buyers seen
        self.size_sol = size_sol
        self.score_min = score_min
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.first_mult = first_mult
        self.first_frac = first_frac
        self.mev = mev
        self.haircut = haircut
        self.grad_vsol = self.pc.v_sol_graduation
        self.k = 100.0 * 100.0 / curve_price_per_token(100.0, self.pc)  # constant-product k
        self.first_vsol: dict[str, float] = {}
        self.first_ts: dict[str, float] = {}
        self.positions: dict[str, CMPosition] = {}
        self.traded: set[str] = set()      # one position per mint, ever
        self.closed: list[CMTrade] = []

    # -- exact constant-product round-trip return for a given tranche size ----------
    def _trade_return(self, v_entry: float, v_exit: float, ts: float, size: float) -> float:
        fee = fee_at(ts).curve_fee_rate
        s_in = size * (1.0 - fee - self.mev - self.haircut)
        if s_in <= 0:
            return -1.0
        q = self.k / v_entry - self.k / (v_entry + s_in)      # tokens bought (entry impact)
        if q <= 0:
            return -1.0
        sol_raw = v_exit - self.k / (self.k / v_exit + q)     # SOL out (exit impact)
        sol_net = sol_raw * (1.0 - fee - self.haircut)
        return sol_net / size - 1.0

    # -- main driver: feed every swap for a tracked mint ---------------------------
    def on_swap(self, mint: str, swap: Swap, score_fn=None) -> CMTrade | None:
        """Process one swap. ``score_fn() -> float`` is called lazily ONLY at the
        entry trigger (so the model is scored at most once per token). Returns a
        CMTrade when this swap closes a position, else None."""
        if mint not in self.first_vsol:
            self.first_vsol[mint] = swap.vsol_after
            self.first_ts[mint] = swap.ts
        # track distinct smart wallets buying this mint (confidence signal at entry)
        if self.smart and swap.is_buy and swap.wallet in self.smart:
            self.smart_buyers.setdefault(mint, set()).add(swap.wallet)
        if mint in self.positions:
            return self._maybe_exit(mint, swap)
        if mint in self.traded:
            return None
        # entry trigger: the FIRST swap to reach the entry zone is the single decision
        # point. We mark the mint decided either way, so a sub-threshold score means
        # skip — we don't re-enter later at a higher (worse) vSol.
        if swap.vsol_after >= ENTRY_VSOL and self.first_vsol[mint] < CATCH_VSOL:
            self.traded.add(mint)
            score = score_fn() if score_fn is not None else None
            if score is not None and score >= self.score_min:
                self._open(mint, swap, float(score))
        return None

    def _open(self, mint: str, swap: Swap, score: float) -> None:
        v = swap.vsol_after
        dt = swap.ts - self.first_ts.get(mint, swap.ts)
        vel = (v - self.first_vsol.get(mint, v)) / dt if dt > 0 else 0.0
        self.positions[mint] = CMPosition(
            mint=mint, entry_ts=swap.ts, entry_vsol=v, size_sol=self.size_sol,
            model_score=score, first_reserve=v * sqrt(self.first_mult),
            tp_reserve=v * sqrt(self.tp_mult), sl_reserve=v * sqrt(self.sl_mult),
            last_pregrad_vsol=v, velocity=vel,
            smart_buys=len(self.smart_buyers.get(mint, ())))
        self.traded.add(mint)

    def _maybe_exit(self, mint: str, swap: Swap) -> CMTrade | None:
        pos = self.positions[mint]
        v = swap.vsol_after
        if v >= self.grad_vsol:                               # graduating: exit on curve
            return self._finalize(pos, pos.last_pregrad_vsol, swap.ts, "grad")
        pos.last_pregrad_vsol = v
        # smart-sell tracking: LOG-ONLY. The regime test (regime_test.py, walk-forward)
        # showed the force-exit override was a mild net DRAG, not a hedge (e.g. W2
        # +11.5%->+10.0%), and it fired ~0x live. We keep counting distinct smart sellers
        # so smart_sells is still recorded per trade, but no longer force-exit on it.
        if self.smart and not swap.is_buy and swap.wallet in self.smart:
            pos.smart_sellers.add(swap.wallet)
        if not pos.banked:
            if v <= pos.sl_reserve:                           # hard stop before banking
                return self._finalize(pos, v, swap.ts, "sl")
            if v >= pos.first_reserve:                        # bank first tranche, keep runner
                pos.banked = True
                pos.banked_ret = self._trade_return(
                    pos.entry_vsol, v, swap.ts, pos.size_sol * self.first_frac)
        else:
            if v >= pos.tp_reserve:                           # runner take-profit
                return self._finalize(pos, v, swap.ts, "tp")
            if v <= pos.entry_vsol:                           # runner breakeven stop
                return self._finalize(pos, v, swap.ts, "be")
        if swap.ts - pos.entry_ts > MAX_HOLD_SEC:             # safety time-stop (either phase)
            return self._finalize(pos, v, swap.ts, "time")
        return None

    def force_close(self, mint: str, ts: float, reason: str = "evict") -> CMTrade | None:
        """Close an open position at its last seen pre-graduation price. Called by the
        monitor on stale-token eviction / migration so positions never dangle open."""
        pos = self.positions.get(mint)
        if pos is None:
            return None
        return self._finalize(pos, pos.last_pregrad_vsol, ts, reason)

    def _finalize(self, pos: CMPosition, runner_exit_vsol: float, ts: float, reason: str) -> CMTrade:
        """Close the position, blending the banked tranche (if any) with the runner."""
        if pos.banked:
            runner = self._trade_return(pos.entry_vsol, runner_exit_vsol, ts,
                                        pos.size_sol * (1.0 - self.first_frac))
            ret = self.first_frac * pos.banked_ret + (1.0 - self.first_frac) * runner
        else:
            ret = self._trade_return(pos.entry_vsol, runner_exit_vsol, ts, pos.size_sol)
        t = CMTrade(mint=pos.mint, entry_ts=pos.entry_ts, exit_ts=ts,
                    entry_vsol=pos.entry_vsol, exit_vsol=runner_exit_vsol, size_sol=pos.size_sol,
                    ret=ret, reason=reason, model_score=pos.model_score, velocity=pos.velocity,
                    smart_buys=pos.smart_buys, smart_sells=len(pos.smart_sellers))
        self.closed.append(t)
        del self.positions[pos.mint]
        self.smart_buyers.pop(pos.mint, None)
        return t

    def stats(self) -> dict:
        import statistics as st
        r = [t.ret for t in self.closed]
        if not r:
            return {"n": 0, "open": len(self.positions)}
        return {"n": len(r), "open": len(self.positions),
                "mean_ret": sum(r) / len(r), "median_ret": st.median(r),
                "win_rate": sum(x > 0 for x in r) / len(r),
                "total_pnl_sol": sum(x * self.size_sol for x in r),
                "by_reason": {x: sum(t.reason == x for t in self.closed)
                              for x in sorted({t.reason for t in self.closed})}}
