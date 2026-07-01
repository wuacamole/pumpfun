"""CURVE-AWARE SLIPPAGE STRESS TEST — does the pre-graduation curve-exit edge survive
realistic, size-dependent slippage instead of a flat 2%?

peak_exit_replay.py found a modest realizable edge (TP3x/SL0.4x mean +7.6%, CI-low +4%)
but used a FLAT 2%/side slippage — almost certainly too optimistic in the tail (winners
near graduation, vSol~100-114, where the curve is steepest and your own sell moves it
most). Here we replace the flat slippage with EXACT constant-product math:

  buy  S SOL at reserve V:  S_in = S*(1-fee-mev); Q = k/V - k/(V+S_in)   (entry impact)
  sell Q at reserve V_x:    SOL = (V_x - k/(k/V_x + Q)) * (1-fee)        (exit impact)
  realized return = SOL/S - 1

Slippage now scales with position size S, so sweeping S maps the strategy's CAPACITY:
the size at which impact eats the edge. TP/SL triggers fire on SPOT price (V=V_entry*sqrt(M)),
but the realized fill includes our own impact — exactly the drag the flat model ignored.

Approximation: we trade our size against the curve at the reserves on the RECORDED
trajectory (price-taker on the path, but pay impact on our own size) — standard backtest
convention, conservative if anything. Free replay; no Helius.
"""
from __future__ import annotations

import numpy as np
from config import fee_at, get_cached_constants
from model import curve_price_per_token
from peak_exit_replay import ENTRY_VSOL, load

# Current-regime curve fee (1%) + MEV on entry (1%); slippage now modelled exactly.
FEE = 0.01
MEV = 0.01


def k_whole(pc):
    return 100.0 * 100.0 / curve_price_per_token(100.0, pc)


def trade_return(V_e, V_x, S, k):
    """Realized return of buying S SOL at reserve V_e, selling at reserve V_x, with
    exact constant-product entry+exit impact and fees."""
    S_in = S * (1.0 - FEE - MEV)
    Q = k / V_e - k / (V_e + S_in)            # tokens bought (entry impact baked in)
    if Q <= 0:
        return -1.0
    vtok_x = k / V_x
    sol_raw = V_x - k / (vtok_x + Q)          # SOL out (exit impact baked in)
    sol_net = sol_raw * (1.0 - FEE)
    return sol_net / S - 1.0


def replay_size(S, tp_mult, sl_mult, toks, pc, k, grad_v):
    rets, grad_flags = [], []
    for mint, graduated, swaps in toks:
        i0 = next((i for i, s in enumerate(swaps) if s.vsol_after >= ENTRY_VSOL), None)
        if i0 is None or i0 < 2:
            continue
        V_e = swaps[i0].vsol_after
        fwd = [s for s in swaps[i0 + 1:] if s.vsol_after < grad_v]
        tp_reserve = V_e * np.sqrt(tp_mult)   # spot hits TP when V = V_e*sqrt(TP)
        sl_reserve = V_e * np.sqrt(sl_mult)
        V_x = None
        for s in fwd:
            if s.vsol_after >= tp_reserve or s.vsol_after <= sl_reserve:
                V_x = s.vsol_after; break
        if V_x is None:
            V_x = fwd[-1].vsol_after if fwd else V_e
        rets.append(trade_return(V_e, V_x, S, k))
        grad_flags.append(graduated)
    return np.array(rets), np.array(grad_flags)


def ci_low(x, nb=3000):
    rng = np.random.default_rng(0)
    return np.percentile([rng.choice(x, len(x)).mean() for _ in range(nb)], 2.5)


def jackknife_pos(x):
    """Still positive-mean after deleting the single best trade?"""
    if len(x) < 2:
        return False
    return (x.sum() - x.max()) / (len(x) - 1) > 0


def main():
    pc = get_cached_constants()
    k = k_whole(pc)
    grad_v = pc.v_sol_graduation
    toks = load()
    print(f"curve-aware slippage stress test  (k_whole={k:.1f}, grad_vSol={grad_v:.1f})")
    print(f"loaded {len(toks)} entry-zone tokens\n")
    for tp, sl in [(99.0, 0.0001), (3.0, 0.4)]:
        print(f"### exit rule TP={tp}x / SL={sl}x  (flat-2% version gave "
              f"{'+8.8%' if tp>50 else '+7.6%'} mean) ###")
        print(f"{'size_SOL':>8} {'n':>5} {'mean':>8} {'median':>8} {'win%':>6} "
              f"{'CI_low':>8} {'jack+':>6}")
        for S in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
            r, g = replay_size(S, tp, sl, toks, pc, k, grad_v)
            print(f"{S:>8.2f} {len(r):>5} {r.mean():>+8.1%} {np.median(r):>+8.1%} "
                  f"{(r>0).mean():>6.1%} {ci_low(r):>+8.1%} {str(jackknife_pos(r)):>6}")
        print()


if __name__ == "__main__":
    main()
