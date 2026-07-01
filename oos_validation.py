"""TIME-ORDERED OUT-OF-SAMPLE VALIDATION of the pre-graduation curve-momentum rule.

The curve_slippage_replay result (+18% CI-low @0.5 SOL) used a TP/SL tuned IN-SAMPLE
over several configs — the exact researcher-DoF that produced false positives earlier.
This guards against it honestly:

  1. Sort tokens by launch ts; split EARLY (selection) / LATE (test).
  2. Grid-search TP/SL on EARLY ONLY; pick the config with the best EARLY mean return
     (what you'd actually do tuning on history).
  3. Apply that frozen config to LATE (tokens that launch AFTER everything used to
     select it). Report OOS mean/median/CI-low/jackknife.

If the LATE (OOS) edge holds (CI-low>0, jackknife+), it survived selection. If it
collapses, TP=3x/SL=0.4x was overfit. Exact constant-product slippage; size=1 SOL.
Free replay; no Helius.
"""
from __future__ import annotations

import numpy as np
from config import get_cached_constants
from curve_slippage_replay import ci_low, jackknife_pos, k_whole, load, replay_size

SIZE_SOL = 1.0
SPLIT = 0.6  # earliest 60% select, latest 40% test
TPS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 99.0]   # 99 = hold to graduation/end (no TP)
SLS = [0.3, 0.4, 0.5, 0.6]


def stats(toks, tp, sl, pc, k, grad_v):
    r, g = replay_size(SIZE_SOL, tp, sl, toks, pc, k, grad_v)
    return r, r.mean(), ci_low(r), jackknife_pos(r), (r > 0).mean()


def main():
    pc = get_cached_constants(); k = k_whole(pc); grad_v = pc.v_sol_graduation
    toks = load()
    toks.sort(key=lambda t: t[2][0].ts)           # by launch ts
    s = int(len(toks) * SPLIT)
    early, late = toks[:s], toks[s:]
    print(f"n={len(toks)}  EARLY(select)={len(early)}  LATE(test)={len(late)}  "
          f"size={SIZE_SOL} SOL\n")

    # 1) Grid-search on EARLY; pick best early mean.
    grid = []
    for tp in TPS:
        for sl in SLS:
            _, em, ecl, _, _ = stats(early, tp, sl, pc, k, grad_v)
            grid.append((em, ecl, tp, sl))
    grid.sort(reverse=True)
    best_em, best_ecl, btp, bsl = grid[0]
    print("EARLY grid top-5 (by early mean):")
    for em, ecl, tp, sl in grid[:5]:
        print(f"  TP={tp:<4} SL={sl}  early mean {em:+.1%}  CI-low {ecl:+.1%}")
    print(f"\n=> SELECTED on early data: TP={btp} / SL={bsl}\n")

    # 2) Freeze it; evaluate on LATE (true OOS).
    _, lm, lcl, ljk, lwin = stats(late, btp, bsl, pc, k, grad_v)
    print("OUT-OF-SAMPLE (late slice, frozen rule):")
    print(f"  TP={btp}/SL={bsl}  mean {lm:+.1%}  CI-low {lcl:+.1%}  "
          f"jackknife+ {ljk}  win {lwin:.1%}")

    # 3) Context: how the previously-highlighted TP3x/SL0.4x does OOS, and the average
    #    OOS over the whole grid (is the family positive, or did selection cherry-pick?).
    _, m3, cl3, jk3, _ = stats(late, 3.0, 0.4, pc, k, grad_v)
    print(f"\n  reference TP3x/SL0.4x OOS: mean {m3:+.1%}  CI-low {cl3:+.1%}  jackknife+ {jk3}")
    late_means = [stats(late, tp, sl, pc, k, grad_v)[1] for tp in TPS for sl in SLS]
    print(f"  whole-grid OOS mean: avg {np.mean(late_means):+.1%}  "
          f"min {np.min(late_means):+.1%}  max {np.max(late_means):+.1%}  "
          f"frac>0 {np.mean(np.array(late_means) > 0):.0%}")

    # 4) Did early ranking predict late? (selection validity)
    late_for_grid = {(tp, sl): stats(late, tp, sl, pc, k, grad_v)[1]
                     for _, _, tp, sl in grid}
    from scipy.stats import spearmanr
    early_rank = [em for em, _, _, _ in grid]
    late_rank = [late_for_grid[(tp, sl)] for _, _, tp, sl in grid]
    rho, p = spearmanr(early_rank, late_rank)
    print(f"\n  early-vs-late config-ranking Spearman: rho={rho:+.2f} p={p:.3f}  "
          f"(does early tuning transfer?)")


if __name__ == "__main__":
    main()
