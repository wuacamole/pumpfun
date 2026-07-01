"""MODEL-SELECTION LIFT — does entering only the model's high-P(graduation) subset beat
the full entry-zone basket for the curve-momentum strategy?

The strategy's winners are tokens that climb the curve (toward graduation); the fitted
graduation model has entry-zone OOS AUC 0.75. So selecting high-P(grad) tokens should
preferentially pick curve-climbers -> higher win rate, higher mean, lower variance.

No leakage: the model is REFIT on EARLY tokens only and scores LATE (OOS) tokens at their
entry snapshot. Curve-momentum return uses the frozen rule (TP5x/SL0.3x) and moderate
frictions (lat1/1, MEV3%, haircut2%), 1 SOL. Free; no Helius.
"""
from __future__ import annotations

import numpy as np
from config import get_cached_constants
from curve_slippage_replay import ci_low, k_whole, load
from exec_stress import trade_return
from model import build_training_frame, fit, predict_grad_prob
from peak_exit_replay import ENTRY_VSOL

TP, SL = 5.0, 0.3
SPLIT = 0.6
ENTRY_LAT, EXIT_LAT, MEV, HAIRCUT = 1, 1, 0.03, 0.02


def entry_idx(swaps):
    i0 = next((i for i, s in enumerate(swaps) if s.vsol_after >= ENTRY_VSOL), None)
    return i0 if (i0 is not None and i0 >= 2) else None


def cm_return(swaps, pc, k, grad_v):
    i0 = entry_idx(swaps)
    if i0 is None:
        return None
    ei = min(i0 + ENTRY_LAT, len(swaps) - 1)
    V_e = swaps[ei].vsol_after
    if V_e <= 0 or V_e >= grad_v:
        return None
    fwd = [s for s in swaps[ei + 1:] if s.vsol_after < grad_v]
    if not fwd:
        return None
    tp_r, sl_r = V_e * np.sqrt(TP), V_e * np.sqrt(SL)
    j = next((j for j, s in enumerate(fwd) if s.vsol_after >= tp_r or s.vsol_after <= sl_r), None)
    xi = (len(fwd) - 1) if j is None else min(j + EXIT_LAT, len(fwd) - 1)
    return trade_return(V_e, fwd[xi].vsol_after, 1.0, k, MEV, HAIRCUT)


def main():
    pc = get_cached_constants(); k = k_whole(pc); grad_v = pc.v_sol_graduation
    toks = load(); toks.sort(key=lambda t: t[2][0].ts)
    s = int(len(toks) * SPLIT)
    early, late = toks[:s], toks[s:]

    # Train graduation model on EARLY entry-zone snapshots only.
    ex = []
    for mint, grad, swaps in early:
        i0 = entry_idx(swaps)
        if i0 is not None:
            ex.append((mint, swaps, grad, None, swaps[i0].ts))
    df = build_training_frame(ex)
    model = fit(df, kind="gbm")
    print(f"trained gbm on {len(ex)} early entry-zone tokens; "
          f"scoring {len(late)} late tokens\n")

    # Score + trade each LATE token.
    rows = []
    for mint, grad, swaps in late:
        i0 = entry_idx(swaps)
        if i0 is None:
            continue
        ret = cm_return(swaps, pc, k, grad_v)
        if ret is None:
            continue
        score = predict_grad_prob(model, swaps, as_of=swaps[i0].ts)
        rows.append((score, ret, grad))
    rows.sort(key=lambda r: r[0], reverse=True)             # high score first
    score = np.array([r[0] for r in rows])
    ret = np.array([r[1] for r in rows])
    grad = np.array([r[2] for r in rows])
    n = len(ret)

    def show(name, mask):
        r = ret[mask]
        if len(r) == 0:
            print(f"  {name:<22} (empty)"); return
        print(f"  {name:<22} n={len(r):>4}  grad={grad[mask].mean():>5.1%}  "
              f"mean {r.mean():>+7.1%}  CI-low {ci_low(r):>+7.1%}  win {(r>0).mean():>5.1%}")

    print("curve-momentum return by model-score selection (OOS late slice):")
    show("ALL (basket)", np.ones(n, bool))
    order = np.argsort(-score)
    for frac in (0.50, 0.25, 0.10):
        top = np.zeros(n, bool); top[order[:int(n * frac)]] = True
        show(f"top {int(frac*100)}% by score", top)
    bot = np.zeros(n, bool); bot[order[int(n*0.5):]] = True
    show("bottom 50% by score", bot)


if __name__ == "__main__":
    main()
