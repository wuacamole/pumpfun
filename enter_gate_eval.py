"""Out-of-sample evaluation of the model-based ENTER gate on the MULTI-SNAPSHOT
model, measured at the entry-zone snapshots (where the live gate actually fires).

GroupKFold by mint (no per-token leakage), calibrated model (matches live), gate =
edge = model_P - breakeven > ENTER_EDGE_MARGIN.
"""
import warnings

import numpy as np
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

from conviction_engine import _breakeven_probability
from model import FEATURE_COLUMNS, LABEL_COL, build_training_frame, fit
import refit_model as rm

ENTER_EDGE_MARGIN = 0.05


def main():
    tokens = rm._load_tokens()
    df = build_training_frame(rm.build_snapshots(tokens))
    X, y = df[FEATURE_COLUMNS], df[LABEL_COL].astype(int).values
    groups = df["mint"].values
    vsol = df["vsol"].values

    # OOS calibrated predictions, grouped by mint.
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = fit(df.iloc[tr], kind="gbm", calibrate=True)
        oof[te] = m.predict(X.iloc[te])

    brk = np.array([_breakeven_probability(v) for v in vsol])
    enter = (oof - brk) > ENTER_EDGE_MARGIN

    # Evaluate at the entry-zone snapshots (where the live ENTER decision lands),
    # de-duplicated to one row per token (first entry-zone snapshot) so precision/
    # recall are per-token, not per-snapshot.
    ez = (vsol >= 33) & (vsol <= 50)
    seen, rows = set(), []
    for i in np.where(ez)[0]:
        mint = groups[i]
        if mint in seen:
            continue
        seen.add(mint)
        rows.append((bool(enter[i]), int(y[i])))
    n = len(rows)
    n_grad = sum(g for _, g in rows)
    n_enter = sum(1 for e, _ in rows if e)
    if n_enter:
        prec = sum(g for e, g in rows if e) / n_enter
        rec = sum(1 for e, g in rows if e and g) / n_grad
        ev = prec * 4.0 + (1 - prec) * (-0.9)
        print(f"entry-zone tokens={n}  graduated={n_grad} (base {n_grad/n:.1%})")
        print(f"NEW multi-snapshot gate (edge>{ENTER_EDGE_MARGIN}, OOS GroupKFold):")
        print(f"  ENTER={n_enter}  precision={prec:.1%}  recall={rec:.1%}  illustrative_EV={ev:+.2f}x")
    print("  (old hand gate: precision 6.5%; single-snapshot model: precision 84.3% / recall 93.8%)")


if __name__ == "__main__":
    main()
