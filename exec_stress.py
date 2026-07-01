"""EXECUTION-FRICTION STRESS TEST — does the curve-momentum edge survive pessimistic,
realistic execution instead of perfect fills?

Applied to the OOS LATE slice with the FROZEN rule (TP=5x/SL=0.3x, selected on early
data), so this stacks on the time-ordered OOS validation. Frictions layered:

  * EXIT latency: you SEE the trigger swap, then submit — you fill at the NEXT swap's
    reserve, not the trigger's. Captures TP mean-reversion AND stop gap-through (if the
    next print is lower, you fill lower).
  * ENTRY latency: you fill one swap after your decision (usually a worse, higher vSol).
  * MEV: sandwiching beyond the 1% baseline.
  * Adverse-fill haircut: extra per-side loss (priority fees, front-running) on top of
    exact curve impact.

If the edge survives the COMBINED pessimistic scenario, it is robust to execution. If it
dies, we stop here and spend no credits. Exact constant-product slippage; size 1 SOL.
"""
from __future__ import annotations

import numpy as np
from config import get_cached_constants
from curve_slippage_replay import ci_low, jackknife_pos, k_whole, load
from peak_exit_replay import ENTRY_VSOL

SIZE_SOL = 1.0
FEE = 0.01
TP, SL = 5.0, 0.3          # frozen rule from OOS selection
SPLIT = 0.6                # evaluate on the LATE 40% (OOS slice)


def trade_return(V_e, V_x, S, k, mev, haircut):
    S_in = S * (1.0 - FEE - mev - haircut)
    if S_in <= 0:
        return -1.0
    Q = k / V_e - k / (V_e + S_in)
    if Q <= 0:
        return -1.0
    sol_raw = V_x - k / (k / V_x + Q)
    sol_net = sol_raw * (1.0 - FEE - haircut)
    return sol_net / S - 1.0


def run(toks, pc, k, grad_v, entry_lat, exit_lat, mev, haircut):
    tp_r_factor, sl_r_factor = np.sqrt(TP), np.sqrt(SL)
    rets = []
    for mint, graduated, swaps in toks:
        i0 = next((i for i, s in enumerate(swaps) if s.vsol_after >= ENTRY_VSOL), None)
        if i0 is None or i0 < 2:
            continue
        ei = min(i0 + entry_lat, len(swaps) - 1)        # entry latency
        V_e = swaps[ei].vsol_after
        if V_e <= 0 or V_e >= grad_v:
            continue
        fwd = [s for s in swaps[ei + 1:] if s.vsol_after < grad_v]
        if not fwd:
            continue
        tp_res, sl_res = V_e * tp_r_factor, V_e * sl_r_factor
        jtrig = None
        for j, s in enumerate(fwd):
            if s.vsol_after >= tp_res or s.vsol_after <= sl_res:
                jtrig = j
                break
        if jtrig is None:
            V_x = fwd[-1].vsol_after                     # forced exit, last pre-grad
        else:
            V_x = fwd[min(jtrig + exit_lat, len(fwd) - 1)].vsol_after  # exit latency
        rets.append(trade_return(V_e, V_x, SIZE_SOL, k, mev, haircut))
    return np.array(rets)


def main():
    pc = get_cached_constants(); k = k_whole(pc); grad_v = pc.v_sol_graduation
    toks = load(); toks.sort(key=lambda t: t[2][0].ts)
    late = toks[int(len(toks) * SPLIT):]
    print(f"OOS late slice; frozen rule TP={TP}x/SL={SL}x; size {SIZE_SOL} SOL\n")
    scenarios = [
        ("baseline (perfect fill, MEV1%)",        0, 0, 0.01, 0.00),
        ("+ exit latency 1 swap",                 0, 1, 0.01, 0.00),
        ("+ entry latency 1 swap",                1, 1, 0.01, 0.00),
        ("+ MEV 3%",                              1, 1, 0.03, 0.00),
        ("+ adverse-fill haircut 2%/side",        1, 1, 0.03, 0.02),
        ("PESSIMISTIC (lat1/1, MEV5%, haircut3%)",1, 1, 0.05, 0.03),
    ]
    print(f"{'scenario':<42} {'n':>5} {'mean':>8} {'CI_low':>8} {'win%':>6} {'jack+':>6}")
    for name, el, xl, mev, hc in scenarios:
        r = run(late, pc, k, grad_v, el, xl, mev, hc)
        print(f"{name:<42} {len(r):>5} {r.mean():>+8.1%} {ci_low(r):>+8.1%} "
              f"{(r>0).mean():>6.1%} {str(jackknife_pos(r)):>6}")


if __name__ == "__main__":
    main()
