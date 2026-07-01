"""Win-rate improvement study (run #15 in the DoF counter — log in DECISIONS.md).

Two pre-specified axes, evaluated with time-ordered train(60%)/test(40%) discipline
on the model-gated (score >= 0.070) floor-38 caught-clean basket:

  A. ENTRY-QUALITY FILTERS (attack the SL bucket — CURVE_MOMENTUM.md §10.1):
     five as-of-entry features NOT already in the model's active set:
       vel        vSol/s first->entry (logged live, never gated)
       sell_share fraction of pre-entry swaps that are sells
       retrace    max fractional vSol pullback on the pre-entry path (spike-dump tell)
       buy_conc   max single-wallet share of pre-entry buy SOL (bundler tell)
       n_buyers   distinct pre-entry buyer wallets
     Direction+threshold (quartiles) chosen on TRAIN only; ALL features reported on
     TEST (no cherry-pick reporting).

  B. EXIT GEOMETRY (mechanical win-rate lever): baseline scale-out vs earlier bank
     (1.2x) vs trailing runner (trail spot-stop at 0.85/0.75 of running peak, with
     breakeven floor). Same train/test discipline.

Costs identical to stress_scaleout (fee 1%/side + MEV 1%, exact constant-product
impact, 1 SOL). PAPER research only."""
from __future__ import annotations

import numpy as np

from curve_slippage_replay import ci_low
from model import ModelStore, predict_grad_prob
from peak_exit_replay import load
from stress_scaleout import FIRST_FRAC, FIRST_MULT, FLOOR, GRAD_V, SL, scaleout_ret, tr

S, MEV = 1.0, 0.01
GATE = 0.070


# ---------- exit variants ----------------------------------------------------
def scaleout_ret_variant(swaps, i0, first_mult=1.3, first_frac=0.5, tp=1.7):
    import stress_scaleout as ss
    old = (ss.FIRST_MULT, ss.FIRST_FRAC, ss.RUNNER_TP)
    ss.FIRST_MULT, ss.FIRST_FRAC, ss.RUNNER_TP = first_mult, first_frac, tp
    try:
        return scaleout_ret(swaps, i0, S, MEV)
    finally:
        ss.FIRST_MULT, ss.FIRST_FRAC, ss.RUNNER_TP = old


def trail_ret(swaps, i0, trail=0.85, first_mult=FIRST_MULT, first_frac=FIRST_FRAC):
    """Scale-out with a TRAILING runner: bank first_frac at first_mult; the runner
    then exits when spot falls to trail*peak-spot (reserve = peakV*sqrt(trail)),
    floored at breakeven. Hard SL before banking, grad force-exit, end-of-data."""
    V_e = swaps[i0].vsol_after
    first_r, sl_r = V_e * np.sqrt(first_mult), V_e * np.sqrt(SL)
    banked, r1, peak = False, 0.0, V_e
    for s in swaps[i0 + 1:]:
        v = s.vsol_after
        if not banked:
            if v <= sl_r:
                return tr(V_e, sl_r, S, MEV)
            if v >= first_r or v >= GRAD_V:
                banked = True
                r1 = tr(V_e, min(v, GRAD_V), S * first_frac, MEV)
                peak = min(v, GRAD_V)
                continue
        else:
            peak = max(peak, min(v, GRAD_V))
            stop = max(peak * np.sqrt(trail), V_e)   # trail with breakeven floor
            if v >= GRAD_V:
                return first_frac * r1 + (1 - first_frac) * tr(V_e, GRAD_V, S * (1 - first_frac), MEV)
            if v <= stop:
                return first_frac * r1 + (1 - first_frac) * tr(V_e, max(v, stop * 0.999), S * (1 - first_frac), MEV)
    fwd = [s for s in swaps[i0 + 1:] if s.vsol_after < GRAD_V]
    last = fwd[-1].vsol_after if fwd else V_e
    if not banked:
        return tr(V_e, last, S, MEV)
    return first_frac * r1 + (1 - first_frac) * tr(V_e, last, S * (1 - first_frac), MEV)


# ---------- entry features (strictly pre-entry, swaps[:i0+1]) -----------------
def entry_features(swaps, i0):
    pre = swaps[:i0 + 1]
    v_e, t_e = pre[-1].vsol_after, pre[-1].ts
    dt = t_e - pre[0].ts
    vel = (v_e - pre[0].vsol_after) / dt if dt > 0 else 0.0
    sells = sum(1 for s in pre if not s.is_buy)
    sell_share = sells / len(pre)
    peak = retrace = 0.0
    for s in pre:
        peak = max(peak, s.vsol_after)
        retrace = max(retrace, (peak - s.vsol_after) / peak if peak > 0 else 0.0)
    buys = {}
    for s in pre:
        if s.is_buy:
            buys[s.wallet] = buys.get(s.wallet, 0.0) + s.sol_amount
    tot = sum(buys.values())
    buy_conc = max(buys.values()) / tot if tot > 0 else 1.0
    return {"vel": vel, "sell_share": sell_share, "retrace": retrace,
            "buy_conc": buy_conc, "n_buyers": float(len(buys))}


def show(name, r, indent="  "):
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        print(f"{indent}{name:<34} (empty)")
        return
    jack = "+" if len(r) > 1 and (r.sum() - r.max()) / (len(r) - 1) > 0 else "-"
    print(f"{indent}{name:<34} n={len(r):>4} win={(r > 0).mean():>5.1%} "
          f"mean={r.mean():>+6.1%} med={np.median(r):>+6.1%} CI={ci_low(r):>+6.1%} jack{jack}")


def main():
    m = ModelStore().load("latest")
    toks = load()
    toks.sort(key=lambda t: t[2][0].ts)
    rows = []                       # (entry_ts, feats, score, swaps, i0)
    for _mint, grad, swaps in toks:
        i0 = next((i for i, s in enumerate(swaps) if s.vsol_after >= FLOOR), None)
        if i0 is None or i0 < 2:
            continue
        sc = predict_grad_prob(m, swaps, as_of=swaps[i0].ts)
        if sc < GATE:               # live gate: study the population we actually trade
            continue
        rows.append((swaps[i0].ts, entry_features(swaps, i0), sc, swaps, i0, grad))
    rows.sort(key=lambda r: r[0])
    base = np.array([scaleout_ret(sw, i0, S, MEV) for _, _, _, sw, i0, _ in rows])
    n = len(rows)
    s = int(n * 0.6)
    tr_idx, te_idx = np.arange(s), np.arange(s, n)
    print(f"model-gated (>= {GATE}) floor-{FLOOR} entries: n={n}  train={s} test={n - s}")
    show("BASELINE train", base[tr_idx])
    show("BASELINE test ", base[te_idx])

    # ---------------- A. entry filters -------------------------------------
    print("\n=== A. ENTRY FILTERS (threshold fit on TRAIN quartiles, applied to TEST) ===")
    feats = ["vel", "sell_share", "retrace", "buy_conc", "n_buyers"]
    fmat = {f: np.array([r[1][f] for r in rows]) for f in feats}
    for f in feats:
        x = fmat[f]
        q25, q50, q75 = (np.percentile(x[tr_idx], q) for q in (25, 50, 75))
        print(f"\n  -- {f}  (train quartiles {q25:.3g}/{q50:.3g}/{q75:.3g})")
        best = None
        for name, mask_fn in [("keep<=q25", lambda z, q=q25: z <= q), ("keep<=q50", lambda z, q=q50: z <= q),
                              ("keep>=q50", lambda z, q=q50: z >= q), ("keep>=q75", lambda z, q=q75: z >= q)]:
            mtr = mask_fn(x[tr_idx])
            rtr = base[tr_idx][mtr]
            if len(rtr) < 80:
                continue
            ci = ci_low(rtr)
            show(f"train {name}", rtr, indent="    ")
            if best is None or ci > best[0]:
                best = (ci, name, mask_fn)
        if best is not None:
            _, name, mask_fn = best
            mte = mask_fn(x[te_idx])
            show(f"TEST  {name}  <-- train pick", base[te_idx][mte], indent="    ")

    # ---------------- B. exit geometry -------------------------------------
    print("\n=== B. EXIT GEOMETRY (no filter; train -> test) ===")
    variants = [
        ("baseline 50%@1.3 tp1.7/be", lambda sw, i: scaleout_ret(sw, i, S, MEV)),
        ("bank early 50%@1.2 tp1.7/be", lambda sw, i: scaleout_ret_variant(sw, i, first_mult=1.2)),
        ("bank 60%@1.3 tp1.7/be", lambda sw, i: scaleout_ret_variant(sw, i, first_frac=0.6)),
        ("trail runner 0.85 peak", lambda sw, i: trail_ret(sw, i, trail=0.85)),
        ("trail runner 0.75 peak", lambda sw, i: trail_ret(sw, i, trail=0.75)),
    ]
    results = {}
    for name, fn in variants:
        rtr = [fn(r[3], r[4]) for r in (rows[i] for i in tr_idx)]
        results[name] = rtr
        show(f"train {name}", rtr)
    best_name = max(results, key=lambda k: ci_low(np.array(results[k])))
    print(f"\n  train winner by CI-low: {best_name}")
    for name, fn in variants:
        if name in (best_name, variants[0][0]):
            rte = [fn(r[3], r[4]) for r in (rows[i] for i in te_idx)]
            show(f"TEST  {name}", rte)


if __name__ == "__main__":
    main()
