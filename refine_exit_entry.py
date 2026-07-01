"""Refinement search for the grinder at floor=38 — can we raise win-rate WITHOUT
killing EV? Two honest levers:
  (A) EXIT geometry: trailing stop (ride winners, give back X% of peak) and partial
      scale-out (bank half early, breakeven-stop the rest). These convert would-be
      losers into small winners -> win-rate up.
  (B) ENTRY velocity: how fast vsol climbed CATCH->ENTRY (fast climbers = stronger
      momentum). Gate to the top half.

Every variant scored on the backtest store with EXACT constant-product slippage @1 SOL.
Reported: n, win, mean, median, CI-low, AND both-halves (time-ordered) so we can see if
a lift survives out of the fitting window. NOTE: this searches several knobs on one
dataset -> multiple-comparison/overfit risk; only a lift that (a) has a mechanism and
(b) holds in BOTH halves is worth shipping. The winner then goes to the slippage gauntlet.
"""
from __future__ import annotations
import numpy as np
from peak_exit_replay import load
from curve_slippage_replay import trade_return, ci_low, k_whole
from config import get_cached_constants

pc = get_cached_constants(); K = k_whole(pc); GRAD_V = pc.v_sol_graduation
FLOOR, CATCH = 38.0, 32.0
TP, SL = 1.7, 0.2


def entry_idx(swaps):
    i0 = next((i for i, s in enumerate(swaps) if s.vsol_after >= FLOOR), None)
    if i0 is None or i0 < 2:
        return None
    return i0


def exit_fixed(swaps, i0, tp=TP, sl=SL):
    V_e = swaps[i0].vsol_after
    tp_r, sl_r = V_e * np.sqrt(tp), V_e * np.sqrt(sl)
    for s in swaps[i0 + 1:]:
        if s.vsol_after >= GRAD_V:
            return V_e * np.sqrt(min(tp, (GRAD_V / V_e) ** 2))  # capped at grad
        if s.vsol_after >= tp_r or s.vsol_after <= sl_r:
            return s.vsol_after
    fwd = [s for s in swaps[i0 + 1:] if s.vsol_after < GRAD_V]
    return fwd[-1].vsol_after if fwd else V_e


def exit_trailing(swaps, i0, act, trail, hard_sl=SL):
    """Hard SL at hard_sl; once price reaches act*entry, trail: exit when price falls
    `trail` fraction below the running peak. price ∝ vsol^2."""
    V_e = swaps[i0].vsol_after
    sl_r = V_e * np.sqrt(hard_sl)
    act_r = V_e * np.sqrt(act)
    peak = V_e
    armed = False
    for s in swaps[i0 + 1:]:
        v = s.vsol_after
        if v >= GRAD_V:
            return min(v, GRAD_V)
        if v <= sl_r:
            return sl_r
        peak = max(peak, v)
        if v >= act_r:
            armed = True
        if armed:
            trail_r = peak * np.sqrt(1 - trail)   # give back `trail` of PEAK price
            if v <= trail_r:
                return v
    fwd = [s for s in swaps[i0 + 1:] if s.vsol_after < GRAD_V]
    return fwd[-1].vsol_after if fwd else V_e


def ret_partial(swaps, i0, first_mult, first_frac, runner_tp, S):
    """Scale-out: sell first_frac at first_mult; runner exits at runner_tp or breakeven
    (entry) stop. Blended return."""
    V_e = swaps[i0].vsol_after
    first_r = V_e * np.sqrt(first_mult)
    tp_r = V_e * np.sqrt(runner_tp)
    sl_r = V_e * np.sqrt(SL)
    banked = False
    for s in swaps[i0 + 1:]:
        v = s.vsol_after
        if not banked:
            if v <= sl_r:                                   # stopped before banking
                return trade_return(V_e, sl_r, S, K)
            if v >= first_r or v >= GRAD_V:
                banked = True
                r1 = trade_return(V_e, min(v, GRAD_V), S * first_frac, K)
                be_r = V_e                                  # runner stop -> breakeven
                continue
        else:
            if v >= tp_r or v >= GRAD_V:
                r2 = trade_return(V_e, min(v, GRAD_V), S * (1 - first_frac), K)
                return first_frac * r1 + (1 - first_frac) * r2
            if v <= be_r:
                r2 = trade_return(V_e, be_r, S * (1 - first_frac), K)
                return first_frac * r1 + (1 - first_frac) * r2
    fwd = [s for s in swaps[i0 + 1:] if s.vsol_after < GRAD_V]
    last = fwd[-1].vsol_after if fwd else V_e
    if not banked:
        return trade_return(V_e, last, S, K)
    r2 = trade_return(V_e, last, S * (1 - first_frac), K)
    return first_frac * r1 + (1 - first_frac) * r2


def velocity(swaps, i0):
    """vsol climbed per second from the catch swap to entry (momentum speed)."""
    dt = swaps[i0].ts - swaps[0].ts
    return (swaps[i0].vsol_after - swaps[0].vsol_after) / dt if dt > 0 else np.nan


def summary(name, rets, order=None):
    r = np.array(rets)
    line = f"{name:<34} n={len(r):>4} win={(r>0).mean():>5.1%} mean={r.mean():>+6.1%} med={np.median(r):>+6.1%} CI={ci_low(r):>+6.1%}"
    if order is not None:
        o = np.array(order); h = len(o) // 2
        a, b = r[np.argsort(o)][:h], r[np.argsort(o)][h:]
        line += f"  | halves win {(a>0).mean():.0%}/{(b>0).mean():.0%} mean {a.mean():+.0%}/{b.mean():+.0%}"
    print(line)


def main():
    toks = load()
    entries = []
    for mint, grad, swaps in toks:
        i0 = entry_idx(swaps)
        if i0 is not None:
            entries.append((swaps, i0))
    print(f"floor={FLOOR} entries: {len(entries)}  (exact slippage @1 SOL, backtest store)\n")
    S = 1.0
    ts0 = [sw[i].ts for sw, i in entries]

    # baseline
    summary("BASELINE fixed TP1.7/SL0.2", [trade_return(sw[i].vsol_after, exit_fixed(sw, i), S, K) for sw, i in entries], ts0)

    print("\n-- (A) TRAILING STOP (activation x entry, give back X% of peak) --")
    for act in (1.2, 1.3, 1.5):
        for trail in (0.2, 0.3, 0.4):
            rr = [trade_return(sw[i].vsol_after, exit_trailing(sw, i, act, trail), S, K) for sw, i in entries]
            summary(f"trail act={act} giveback={trail}", rr, ts0)

    print("\n-- (A) PARTIAL SCALE-OUT (bank frac at mult, runner->TP/breakeven) --")
    for fm, ff in ((1.3, 0.5), (1.4, 0.5), (1.3, 0.66)):
        rr = [ret_partial(sw, i, fm, ff, TP, S) for sw, i in entries]
        summary(f"scaleout {ff:.0%}@{fm}x runner->1.7x", rr, ts0)

    print("\n-- (B) ENTRY VELOCITY gate (keep fast climbers, fixed exit) --")
    vel = np.array([velocity(sw, i) for sw, i in entries])
    base = np.array([trade_return(sw[i].vsol_after, exit_fixed(sw, i), S, K) for sw, i in entries])
    for q in (50, 33, 25):
        thr = np.nanpercentile(vel, 100 - q)
        m = vel >= thr
        summary(f"top-{q}% velocity (>= {thr:.3f} vsol/s)", base[m], np.array(ts0)[m])


if __name__ == "__main__":
    main()
