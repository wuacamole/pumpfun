"""Does the graduation model add SELECTION lift on top of floor=38 + scale-out? The live
gate is degenerate (forward scores bunched at the 0.065 floor, corr~0 with outcome), so
right now we trade the caught-clean basket ~unfiltered. If the model genuinely separates
winners here, restoring a working live gate is real lift; if not, the gate is moot.

No leakage: model REFIT on EARLY tokens, scores LATE (OOS) tokens at their entry snapshot.
Also reports scale-out PARAM ROBUSTNESS (is 50%@1.3x a plateau or a spike?).
"""
from __future__ import annotations
import numpy as np
from peak_exit_replay import load
from curve_slippage_replay import ci_low, k_whole
from config import get_cached_constants
from model import build_training_frame, fit, predict_grad_prob
from stress_scaleout import scaleout_ret, FLOOR

pc = get_cached_constants(); K = k_whole(pc); GRAD_V = pc.v_sol_graduation
SPLIT = 0.6


def entry_idx(swaps):
    i0 = next((i for i, s in enumerate(swaps) if s.vsol_after >= FLOOR), None)
    return i0 if (i0 is not None and i0 >= 2) else None


def show(name, r, grad=None):
    r = np.array(r)
    if len(r) == 0:
        print(f"  {name:<24} (empty)"); return
    extra = f" grad={np.mean(grad):.0%}" if grad is not None else ""
    print(f"  {name:<24} n={len(r):>4} win={(r>0).mean():>5.1%} mean={r.mean():>+6.1%} CI={ci_low(r):>+6.1%}{extra}")


def main():
    toks = load(); toks.sort(key=lambda t: t[2][0].ts)
    s = int(len(toks) * SPLIT)
    early, late = toks[:s], toks[s:]

    ex = []
    for mint, grad, swaps in early:
        i0 = entry_idx(swaps)
        if i0 is not None:
            ex.append((mint, swaps, grad, None, swaps[i0].ts))
    model = fit(build_training_frame(ex), kind="gbm")
    print(f"gbm trained on {len(ex)} early floor-{FLOOR} tokens; scoring late OOS slice\n")

    rows = []
    for mint, grad, swaps in late:
        i0 = entry_idx(swaps)
        if i0 is None:
            continue
        sc = predict_grad_prob(model, swaps, as_of=swaps[i0].ts)
        rows.append((sc, scaleout_ret(swaps, i0, 1.0, 0.01), grad))
    score = np.array([r[0] for r in rows]); ret = np.array([r[1] for r in rows])
    grad = np.array([r[2] for r in rows]); n = len(ret)
    order = np.argsort(-score)

    print(f"=== MODEL SELECTION on scale-out returns (OOS late, n={n}) ===")
    show("ALL (unfiltered basket)", ret, grad)
    for frac in (0.50, 0.33, 0.25):
        top = order[:int(n * frac)]
        show(f"top {int(frac*100)}% by score", ret[top], grad[top])
    show("bottom 50% by score", ret[order[int(n*0.5):]], grad[order[int(n*0.5):]])
    # is the spread real? correlation
    print(f"\n  corr(score, scaleout_ret) = {np.corrcoef(score, ret)[0,1]:+.3f}   "
          f"corr(score, graduated) = {np.corrcoef(score, grad)[0,1]:+.3f}")

    print(f"\n=== SCALE-OUT PARAM ROBUSTNESS (full store, floor={FLOOR}, @1 SOL) ===")
    import stress_scaleout as ss
    ent = ss.build()
    for fm in (1.2, 1.3, 1.4, 1.5):
        for ff in (0.4, 0.5, 0.6):
            ss.FIRST_MULT, ss.FIRST_FRAC = fm, ff
            show(f"bank {ff:.0%}@{fm}x", [ss.scaleout_ret(sw, i, 1.0, 0.01) for sw, i, _, _ in ent])
    ss.FIRST_MULT, ss.FIRST_FRAC = 1.3, 0.5


if __name__ == "__main__":
    main()
