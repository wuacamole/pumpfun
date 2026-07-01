"""DRAWDOWN / BANKROLL ANALYSIS — can you survive the curve-momentum lottery's losing
streaks long enough to collect its +mean?

22-25% win rate, tail-carried: the mean is positive but the path is brutal. A +14.7%/
trade edge is untradeable if the drawdown bankrupts you first. This builds the time-
ordered equity curve (flat 1 SOL stake) on the OOS slice under MODERATE-realistic
frictions and measures:
  * max peak-to-trough drawdown (SOL and as a multiple of mean trade PnL),
  * longest consecutive losing streak,
  * peak CONCURRENT open positions (capital actually tied up — many tokens are live
    in the entry zone at once),
  * a rough bankroll to survive, and return on deployed capital.

Free; no Helius. Frictions: entry+exit latency 1 swap, MEV 3%, adverse haircut 2%/side
(the +14.7% mean scenario — a competent, not perfect, bot).
"""
from __future__ import annotations

import numpy as np
from config import get_cached_constants
from curve_slippage_replay import k_whole, load
from exec_stress import trade_return
from peak_exit_replay import ENTRY_VSOL

SIZE_SOL = 1.0
TP, SL = 5.0, 0.3
SPLIT = 0.6
ENTRY_LAT, EXIT_LAT, MEV, HAIRCUT = 1, 1, 0.03, 0.02


def trades(toks, pc, k, grad_v):
    tpf, slf = np.sqrt(TP), np.sqrt(SL)
    recs = []
    for mint, graduated, swaps in toks:
        i0 = next((i for i, s in enumerate(swaps) if s.vsol_after >= ENTRY_VSOL), None)
        if i0 is None or i0 < 2:
            continue
        ei = min(i0 + ENTRY_LAT, len(swaps) - 1)
        V_e = swaps[ei].vsol_after
        if V_e <= 0 or V_e >= grad_v:
            continue
        fwd = [s for s in swaps[ei + 1:] if s.vsol_after < grad_v]
        if not fwd:
            continue
        tp_r, sl_r = V_e * tpf, V_e * slf
        jtrig = next((j for j, s in enumerate(fwd)
                      if s.vsol_after >= tp_r or s.vsol_after <= sl_r), None)
        xi = (len(fwd) - 1) if jtrig is None else min(jtrig + EXIT_LAT, len(fwd) - 1)
        ret = trade_return(V_e, fwd[xi].vsol_after, SIZE_SOL, k, MEV, HAIRCUT)
        recs.append((swaps[ei].ts, fwd[xi].ts, ret))
    return recs


def main():
    pc = get_cached_constants(); k = k_whole(pc); grad_v = pc.v_sol_graduation
    toks = load(); toks.sort(key=lambda t: t[2][0].ts)
    recs = trades(toks[int(len(toks) * SPLIT):], pc, k, grad_v)
    recs.sort(key=lambda r: r[1])                      # by exit time
    rets = np.array([r[2] for r in recs])
    pnl = rets * SIZE_SOL
    n = len(pnl)

    equity = np.cumsum(pnl)
    running_max = np.maximum.accumulate(equity)
    dd = running_max - equity
    max_dd = dd.max()
    # longest losing streak (by realized PnL, exit order)
    streak = best = 0
    for x in rets:
        streak = streak + 1 if x < 0 else 0
        best = max(best, streak)
    # peak concurrent open positions
    ev = sorted([(r[0], 1) for r in recs] + [(r[1], -1) for r in recs])
    cur = peak = 0
    for _, d in ev:
        cur += d; peak = max(peak, cur)
    span_hr = (max(r[1] for r in recs) - min(r[0] for r in recs)) / 3600.0

    print(f"OOS slice, 1 SOL/trade, moderate frictions (lat1/1, MEV3%, haircut2%)\n")
    print(f"  trades                 : {n} over {span_hr:.1f}h ({n/span_hr:.0f}/h)")
    print(f"  win rate               : {(rets>0).mean():.1%}")
    print(f"  mean PnL/trade         : {pnl.mean():+.3f} SOL ({rets.mean():+.1%})")
    print(f"  total PnL (window)      : {equity[-1]:+.1f} SOL")
    print(f"  worst single trade      : {pnl.min():+.3f} SOL")
    print(f"  longest losing streak   : {best} trades")
    print(f"  max drawdown            : {max_dd:.1f} SOL "
          f"({max_dd/abs(pnl.mean()):.0f}x mean-trade; "
          f"{max_dd/equity[-1]*100:.0f}% of total window PnL)")
    print(f"  peak concurrent positions: {peak}  (=> ~{peak} SOL deployed at once)")
    bankroll = peak * SIZE_SOL + max_dd
    print(f"  rough bankroll to survive: ~{bankroll:.0f} SOL "
          f"(peak exposure {peak} + max DD {max_dd:.0f})")
    print(f"  window return on bankroll: {equity[-1]/bankroll*100:+.0f}% over {span_hr:.1f}h")
    # variance sanity: per-trade Sharpe-like
    print(f"  per-trade mean/std       : {rets.mean()/rets.std():.3f}  "
          f"(std {rets.std():.1%})")


if __name__ == "__main__":
    main()
