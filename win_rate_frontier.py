"""Dabble: map the win-rate / return frontier of the curve-momentum exit rule, to see
whether a HIGHER-win-rate config exists that stays net-positive (the live forward test
is ~17% win, mostly stop-outs).

Tension: the edge is the heavy tail, so a lower TP banks more (smaller) wins -> higher
win rate but a capped mean. This grids TP x SL with EXACT curve-aware slippage (1 SOL)
on the OOS late slice and reports win% next to mean/CI-low so the tradeoff is visible.
(Picking the best cell here is in-sample; anything promising needs its own OOS.)
Free; no Helius.
"""
from __future__ import annotations

import numpy as np
from config import get_cached_constants
from curve_slippage_replay import ci_low, k_whole, load, replay_size

SPLIT = 0.6
TPS = [1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0, 5.0]
SLS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def main():
    pc = get_cached_constants(); k = k_whole(pc); grad_v = pc.v_sol_graduation
    toks = load(); toks.sort(key=lambda t: t[2][0].ts)
    late = toks[int(len(toks) * SPLIT):]
    print(f"OOS late slice ({len(late)} tokens), 1 SOL, curve-aware slippage\n")
    print(f"{'TP':>4} {'SL':>4} {'n':>5} {'win%':>6} {'mean':>8} {'CI_low':>8} {'med':>7}")
    best = []
    for tp in TPS:
        for sl in SLS:
            r, _ = replay_size(1.0, tp, sl, late, pc, k, grad_v)
            cl = ci_low(r)
            row = (tp, sl, len(r), (r > 0).mean(), r.mean(), cl, np.median(r))
            best.append(row)
    # print sorted by win rate, marking positive-CI rows
    for tp, sl, n, win, mean, cl, med in sorted(best, key=lambda x: -x[3]):
        flag = "  <-- positive CI" if cl > 0 else ""
        print(f"{tp:>4} {sl:>4} {n:>5} {win:>6.1%} {mean:>+8.1%} {cl:>+8.1%} {med:>+7.1%}{flag}")
    pos = [b for b in best if b[5] > 0]
    if pos:
        bw = max(pos, key=lambda x: x[3])
        print(f"\nhighest win-rate config that stays CI-positive: "
              f"TP={bw[0]}/SL={bw[1]}  win {bw[3]:.1%}  mean {bw[4]:+.1%}  CI-low {bw[5]:+.1%}")


if __name__ == "__main__":
    main()
