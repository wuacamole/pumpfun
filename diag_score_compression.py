"""Diagnose live model-score compression. Score real caught-clean tokens at their
floor=38 entry snapshot through the EXACT live path (predict_grad_prob(latest, swaps,
as_of=entry_ts), no creator — same as live_monitor._cm_score) and inspect:
  1. score distribution of the LIVE model (latest.pkl)  -> is it bunched near 0.065?
  2. score distribution of a FRESH-fit gbm on the same store (the backtest path)
  3. per-feature NaN rate + spread at the entry snapshot (what featurize actually sees)
This separates the three hypotheses: degenerate saved model vs NaN features vs snapshot.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from peak_exit_replay import load
from model import (ModelStore, predict_grad_prob, featurize, fit,
                   build_training_frame, FEATURE_COLUMNS)

FLOOR = 38.0


def entry_idx(swaps):
    i0 = next((i for i, s in enumerate(swaps) if s.vsol_after >= FLOOR), None)
    return i0 if (i0 is not None and i0 >= 2) else None


def pct(x):
    x = np.asarray(x)
    return (f"n={len(x)} min={x.min():.4f} p10={np.percentile(x,10):.4f} "
            f"p50={np.percentile(x,50):.4f} p90={np.percentile(x,90):.4f} "
            f"max={x.max():.4f} std={x.std():.4f} | <=0.07: {(x<=0.07).mean():.0%}")


def main():
    toks = load(); toks.sort(key=lambda t: t[2][0].ts)
    ents = [(m, g, sw, entry_idx(sw)) for m, g, sw in toks]
    ents = [(m, g, sw, i) for m, g, sw, i in ents if i is not None]
    print(f"{len(ents)} caught-clean floor-{FLOOR} entries\n")

    # 1. LIVE model path
    live = ModelStore().load("latest")
    print(f"LIVE model latest.pkl: kind={live.kind} brier={live.brier_score:.4f} "
          f"feature_columns={len(getattr(live,'feature_columns',[]))}")
    live_scores = np.array([predict_grad_prob(live, sw, as_of=sw[i].ts) for m, g, sw, i in ents])
    print("LIVE scores  :", pct(live_scores))

    # 2. FRESH-fit gbm (backtest path) on early 60%, scored on the same entries
    s = int(len(ents) * 0.6)
    ex = [(m, sw, g, None, sw[i].ts) for m, g, sw, i in ents[:s]]
    fresh = fit(build_training_frame(ex), kind="gbm")
    fresh_scores = np.array([predict_grad_prob(fresh, sw, as_of=sw[i].ts) for m, g, sw, i in ents])
    print("FRESH scores :", pct(fresh_scores))

    # 3. feature snapshot diagnostics: NaN rate + spread of each feature at entry
    feats = []
    for m, g, sw, i in ents:
        try:
            feats.append(featurize(sw, as_of=sw[i].ts, creator=None))
        except Exception:
            pass
    df = pd.DataFrame(feats)
    print(f"\nper-feature at entry snapshot (n={len(df)}):")
    print(f"{'feature':<26}{'NaN%':>7}{'nunique':>9}{'std':>12}")
    for c in FEATURE_COLUMNS:
        if c in df.columns:
            col = df[c]
            print(f"{c:<26}{col.isna().mean():>6.0%} {col.nunique():>8} {col.std():>12.4g}")
        else:
            print(f"{c:<26}{'MISSING from featurize output':>40}")

    # which features does the LIVE model actually weight?
    fc = getattr(live, "feature_columns", FEATURE_COLUMNS)
    print(f"\nLIVE model.feature_columns ({len(fc)}): {fc}")


if __name__ == "__main__":
    main()
