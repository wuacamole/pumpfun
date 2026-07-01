"""Verify the retrained latest.pkl: (1) scores SPREAD (no longer bunched at base rate),
(2) still SELECT winners on scale-out returns, (3) find the entry-zone median to
recalibrate MODEL_SCORE_MIN (= the top-~50% cut the live gate intends)."""
from __future__ import annotations
import numpy as np
from peak_exit_replay import load
from model import ModelStore, predict_grad_prob
from curve_slippage_replay import ci_low
from stress_scaleout import scaleout_ret, FLOOR


def entry_idx(swaps):
    i0 = next((i for i, s in enumerate(swaps) if s.vsol_after >= FLOOR), None)
    return i0 if (i0 is not None and i0 >= 2) else None


def main():
    m = ModelStore().load("latest")
    print(f"new latest.pkl: kind={m.kind} brier={m.brier_score:.4f}\n")
    toks = load(); toks.sort(key=lambda t: t[2][0].ts)
    ents = [(sw, i) for _, _, sw in toks if (i := entry_idx(sw)) is not None]
    sc = np.array([predict_grad_prob(m, sw, as_of=sw[i].ts) for sw, i in ents])
    ret = np.array([scaleout_ret(sw, i, 1.0, 0.01) for sw, i in ents])

    print(f"score spread (n={len(sc)}): min={sc.min():.4f} p10={np.percentile(sc,10):.4f} "
          f"p50={np.percentile(sc,50):.4f} p75={np.percentile(sc,75):.4f} "
          f"p90={np.percentile(sc,90):.4f} max={sc.max():.4f} | <=0.07: {(sc<=0.07).mean():.0%}")

    order = np.argsort(-sc)
    print("\nselection on scale-out returns (in-sample, sanity):")
    for name, frac in [("ALL", 1.0), ("top50%", .5), ("top33%", .33), ("top25%", .25)]:
        idx = order[:int(len(sc)*frac)]
        r = ret[idx]
        print(f"  {name:<7} n={len(r):>4} win={(r>0).mean():.1%} mean={r.mean():+.1%} CI={ci_low(r):+.1%}")
    print(f"  corr(score,ret)={np.corrcoef(sc,ret)[0,1]:+.3f}")

    p50 = float(np.percentile(sc, 50)); p67 = float(np.percentile(sc, 67))
    print(f"\nRECALIBRATE MODEL_SCORE_MIN: top-50% cut = {p50:.4f} ; top-33% cut = {p67:.4f}")


if __name__ == "__main__":
    main()
