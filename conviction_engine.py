"""
conviction_engine.py
=====================
Pump.fun token scoring engine.

Implements the findings from Marino, Naviglio, Tarantelli & Lillo,
"Predicting the success of new crypto-tokens: the Pump.fun case" (arXiv 2602.14860):

  1. Graduation requires ~85 real SOL bonded (vSol reaches 115 incl. 30 virtual).
  2. Buy-and-hold breakeven:  p(graduate) > (vSol / 115)^2
  3. The dominant predictor of graduation is FAST accumulation via FEW trades
     (conviction), NOT volume or trade-count. High bot-share predicts FAILURE.
  4. isBot = transaction bypassed the pump.fun frontend (direct contract call).

This module is pure logic with no network dependencies, so it can be unit-tested.
Feed it a list of swap events; it returns a score and a trade/skip decision.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from config import ProtocolConstants, fee_at, get_cached_constants

# --- Protocol constants -----------------------------------------------------
# These were hardcoded (30/115/85/0.0125) and are now resolved from the on-chain
# Pump Global account via :mod:`config`. Call :func:`config.load_protocol_constants`
# on startup to populate the cache with the live values; before that, the
# historical defaults (matching the paper) are used so the module is importable
# without a network call. The module-level names below are kept for backward
# compatibility — they read the cache, so they reflect whatever the startup
# load resolved. Don't bind them to a local at import time and expect updates;
# call the functions or read the attributes fresh each time.


def _pc() -> ProtocolConstants:
    """The current protocol constants (from the on-chain cache)."""
    return get_cached_constants()


def __getattr__(name: str) -> object:
    # PEP 562: resolve the legacy constant names against the live cache so a
    # startup on-chain read propagates to every call site without re-import.
    if name == "VIRTUAL_SOL_START":
        return _pc().virtual_sol_start
    if name == "GRADUATION_VSOL":
        return _pc().v_sol_graduation
    if name == "REAL_SOL_TARGET":
        return _pc().real_sol_target
    if name == "FEE_RATE":
        # The bonding-curve fee in effect "now" (live monitor path). Backtests
        # use fee_at(ts) per-trade instead of this module-level constant.
        return fee_at(time.time()).curve_fee_rate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Explicit accessors for the on-chain constants. Internal code should use these
# (not the legacy names) to guarantee the live cache is read at call time.
def _virtual_sol_start() -> float:
    """Initial virtual SOL reserves (30.0 on current mainnet)."""
    return get_cached_constants().initial_virtual_sol_reserves


def _graduation_vsol() -> float:
    """Total vSol at graduation (≈115.0 on current mainnet)."""
    return get_cached_constants().v_sol_graduation


def _real_sol_target() -> float:
    """Real SOL needed to graduate (≈85.0 on current mainnet)."""
    return get_cached_constants().real_sol_target


# Re-export for static analysis / IDE help (values are placeholders; the live
# values come from __getattr__ above). These are also used by the fallback in
# config.get_cached_constants when no on-chain load has run yet.
# NOTE: Do NOT define these as module-level names — that would shadow __getattr__
# and make the on-chain cache read-back dead code. The fallback defaults live
# in config.get_cached_constants(); the __getattr__ below is the single
# dispatch point for live values.


@dataclass
class Swap:
    """One bonding-curve swap event, decoded from the chain."""
    ts: float            # unix seconds
    wallet: str          # trader address
    is_buy: bool         # True = buy, False = sell
    sol_amount: float    # SOL in (buy) or out (sell), pre-fee
    vsol_after: float    # virtual SOL in the bonding curve just after this swap
    via_frontend: bool   # True if routed through pump.fun website; False = direct contract
    # True if the Pump program was invoked directly at the top level (autonomous
    # bot, the paper's isBot), False if routed via another program, None if
    # unknown (no logs: PumpPortal/RPC paths). Parsed from logs at zero RPC cost;
    # a candidate model feature once enough labelled data accumulates. Kept
    # distinct from the size-heuristic ``via_frontend`` so the working model is
    # unaffected until we deliberately retrain.
    is_direct: bool | None = None
    # Raw top-level program that drove the trade (pump itself for a direct call,
    # else the wrapping router/aggregator/telegram-bot). Stored raw so route
    # categorisation stays revisable; None when unknown (no logs). Zero RPC.
    entry_program: str | None = None


@dataclass(frozen=True)
class TokenScore:
    mint: str
    vsol: float
    real_sol_bonded: float
    pct_to_graduation: float
    n_swaps: int
    n_unique_wallets: int
    n_unique_human_wallets: int
    bot_trade_share: float
    sol_per_human_wallet: float     # the conviction metric
    net_sol_per_swap: float         # efficiency: few big trades vs churn
    seconds_since_launch: float
    conviction_score: float         # 0..100 composite
    breakeven_prob: float           # the p you must beat: (vsol/115)^2
    estimated_grad_prob: float      # our conditional estimate
    edge: float                     # estimated_grad_prob - breakeven_prob
    decision: str                   # "ENTER", "WATCH", or "SKIP"
    reasons: list[str] = field(default_factory=list)


def _breakeven_probability(vsol: float) -> float:
    """The minimum graduation probability a buy-and-hold entry needs to be +EV.
    Straight from the paper: p > (vSol / 115)^2."""
    # Floor the progress ratio at 0: bonding progress is physically non-negative
    # (virtual reserves only grow), but the reconstructed-vSol fallback can drift
    # negative on a large sell. A negative ratio would give a spuriously positive
    # breakeven here and a *complex* result under the fractional power in
    # _estimate_graduation_prob — guard at every base-of-a-power use.
    progress = max(vsol / _graduation_vsol(), 0.0)
    return min(progress ** 2, 1.0)


def _estimate_graduation_prob(
    vsol: float,
    bot_share: float,
    sol_per_human_wallet: float,
    net_sol_per_swap: float,
    seconds_since_launch: float,
) -> float:
    """
    Heuristic conditional graduation estimate.

    This is a transparent, hand-built stand-in for the conditional curve the
    paper estimates empirically. Replace the weights with a logistic regression
    fit on your own labelled history once you have data (see fit_model()).

    The structure encodes the paper's qualitative findings:
      - baseline rises with bonding progress (vsol)
      - HIGH bot share pushes the probability DOWN
      - HIGH conviction (sol per human wallet, big net trades) pushes it UP
      - momentum matters: successful tokens move fast (median 4.4 min)
    """
    # Floor at 0 before the fractional power: a negative base under ** 1.6
    # returns a Python complex, which then crashes the final clamp. See
    # _breakeven_probability for why vsol can transiently go negative.
    base = max(vsol / _graduation_vsol(), 0.0) ** 1.6   # below the breakeven parabola on its own

    # Conviction lift: this is what moves the curve ABOVE breakeven.
    conviction = math.tanh(sol_per_human_wallet / 0.5)      # ~0.5 SOL/human wallet saturates
    efficiency = math.tanh(net_sol_per_swap / 0.15)         # big net trades, not churn

    # Bot penalty: high bot share predicts failure beyond early stages.
    bot_penalty = 1.0 - 0.6 * bot_share

    # Momentum: reward early progress; decay if it's stalling.
    if seconds_since_launch <= 0:
        momentum = 1.0
    else:
        sol_per_min = (vsol - _virtual_sol_start()) / max(seconds_since_launch / 60.0, 1e-6)
        momentum = math.tanh(sol_per_min / 5.0)             # ~5 real SOL/min is strong

    raw = base * (0.45 + 0.55 * conviction) * (0.5 + 0.5 * efficiency)
    raw = raw * bot_penalty * (0.4 + 0.6 * momentum)
    return float(max(0.0, min(raw, 1.0)))


def score_token(mint: str, swaps: list[Swap]) -> TokenScore:
    """Compute the full conviction score for a token from its swap history so far."""
    if not swaps:
        raise ValueError("no swaps provided")

    swaps = sorted(swaps, key=lambda s: s.ts)
    t0 = swaps[0].ts
    latest = swaps[-1]
    vsol = latest.vsol_after
    real_sol_bonded = max(0.0, vsol - _virtual_sol_start())
    seconds = max(latest.ts - t0, 1e-6)

    # Bot share, by trade count (the paper's isBot = direct contract call).
    bot_trades = sum(1 for s in swaps if not s.via_frontend)
    bot_share = bot_trades / len(swaps)

    # Net SOL per swap (efficiency): signed buys minus sells.
    net_sol = sum((s.sol_amount if s.is_buy else -s.sol_amount) for s in swaps)
    net_sol_per_swap = net_sol / len(swaps)

    # Conviction: net SOL committed per unique HUMAN wallet.
    human_wallets = {s.wallet for s in swaps if s.via_frontend}
    all_wallets = {s.wallet for s in swaps}
    n_human = len(human_wallets)
    human_net = sum(
        (s.sol_amount if s.is_buy else -s.sol_amount)
        for s in swaps if s.via_frontend
    )
    sol_per_human = (human_net / n_human) if n_human else 0.0

    grad_prob = _estimate_graduation_prob(
        vsol, bot_share, sol_per_human, net_sol_per_swap, seconds
    )
    breakeven = _breakeven_probability(vsol)
    edge = grad_prob - breakeven

    # Composite 0..100 conviction score (for ranking the feed).
    conviction_score = 100 * max(0.0, min(1.0,
        0.45 * math.tanh(sol_per_human / 0.5)
      + 0.25 * math.tanh(net_sol_per_swap / 0.15)
      + 0.20 * (1.0 - bot_share)
      + 0.10 * (real_sol_bonded / _real_sol_target())
    ))

    reasons = []
    if bot_share > 0.7:
        reasons.append(f"bot share {bot_share:.0%} is high -> predicts failure")
    if sol_per_human < 0.05:
        reasons.append("low SOL per human wallet -> churn, not conviction")
    if net_sol_per_swap < 0:
        reasons.append("net SOL outflow -> sellers winning")
    if seconds > 600 and real_sol_bonded < 20:
        reasons.append("stalled: >10 min and <20 real SOL bonded")
    if edge > 0:
        reasons.append(f"edge +{edge:.1%}: estimated odds beat breakeven")

    # Decision gate: only enter when our estimate genuinely clears breakeven
    # AND conviction is real. Otherwise watch or skip.
    if edge > 0.03 and conviction_score >= 55 and bot_share < 0.6:
        decision = "ENTER"
    elif edge > 0 and conviction_score >= 40:
        decision = "WATCH"
    else:
        decision = "SKIP"

    return TokenScore(
        mint=mint, vsol=round(vsol, 3), real_sol_bonded=round(real_sol_bonded, 3),
        pct_to_graduation=round(100 * real_sol_bonded / _real_sol_target(), 1),
        n_swaps=len(swaps), n_unique_wallets=len(all_wallets),
        n_unique_human_wallets=n_human, bot_trade_share=round(bot_share, 3),
        sol_per_human_wallet=round(sol_per_human, 4),
        net_sol_per_swap=round(net_sol_per_swap, 4),
        seconds_since_launch=round(seconds, 1),
        conviction_score=round(conviction_score, 1),
        breakeven_prob=round(breakeven, 4),
        estimated_grad_prob=round(grad_prob, 4),
        edge=round(edge, 4), decision=decision, reasons=reasons,
    )
