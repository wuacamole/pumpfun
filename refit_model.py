"""Refit the graduation-probability model on real outcomes — MULTI-SNAPSHOT.

The single-snapshot model (one example per token at vSol≈36) was accurate AT that
point but the live monitor scores a token continuously across its whole trajectory,
so it extrapolated (fired ENTER at vSol≈30 with over-confidence). Here we generate
MANY no-lookahead snapshots per token (every SNAP_EVERY_SEC, pre-graduation), each
labelled with the token's eventual outcome, so the model is calibrated across the
range it's actually queried at.

Honest evaluation uses GroupKFold by mint — all snapshots of one token stay in the
same fold, else snapshots of the same token leak across train/test and inflate AUC.

Run: uv run python refit_model.py
"""
import warnings
import sqlite3

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

from config import get_cached_constants
from conviction_engine import Swap
from model import FEATURE_COLUMNS, LABEL_COL, ModelStore, build_training_frame, fit

DB = "pumpfun.db"
CATCH_VSOL = 32.0      # only tokens caught before the entry zone (genuine early history)
ENTRY_VSOL = 36.0
SNAP_EVERY_SEC = 20.0  # sample a decision point this often along the trajectory
MIN_SWAPS = 5
MAX_SNAPS = 15         # cap per token so a long-lived token can't dominate
GRAD_VSOL = get_cached_constants().v_sol_graduation


def _load_tokens():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    out = []
    for r in c.execute("SELECT mint,outcome FROM tokens WHERE outcome IN ('graduated','stalled')"):
        swaps = [
            Swap(ts=s["ts"], wallet=s["wallet"], is_buy=bool(s["is_buy"]),
                 sol_amount=s["sol_amount"], vsol_after=s["vsol_after"],
                 via_frontend=bool(s["via_frontend"]))
            for s in c.execute("SELECT ts,wallet,is_buy,sol_amount,vsol_after,via_frontend "
                               "FROM swaps WHERE mint=? ORDER BY ts", (r["mint"],))
        ]
        if swaps and swaps[0].vsol_after < CATCH_VSOL:
            out.append((r["mint"], r["outcome"] == "graduated", swaps))
    c.close()
    return out


def build_snapshots(tokens):
    """Multi-snapshot training examples: (mint, swaps, graduated, None, as_of)."""
    ex = []
    for mint, grad, swaps in tokens:
        t0 = swaps[0].ts
        last_ts = swaps[-1].ts
        k, n = 1, 0
        while n < MAX_SNAPS:
            as_of = t0 + k * SNAP_EVERY_SEC
            if as_of > last_ts:
                break
            seen = [s for s in swaps if s.ts <= as_of]
            k += 1
            if len(seen) < MIN_SWAPS:
                continue
            if seen[-1].vsol_after >= GRAD_VSOL:
                break  # past graduation — not a decision point we model
            ex.append((mint, swaps, grad, None, as_of))
            n += 1
    return ex


def load_examples():
    """Single entry-zone snapshot per token (for the gate eval in enter_gate_eval)."""
    ex = []
    for mint, grad, swaps in _load_tokens():
        as_of = next((s.ts for s in swaps if s.vsol_after >= ENTRY_VSOL), None)
        if as_of is None:
            continue
        if sum(1 for s in swaps if s.ts <= as_of) < 3:
            continue
        ex.append((mint, swaps, grad, None, as_of))
    return ex


def group_cv(df, kind, n_splits=5):
    X, y = df[FEATURE_COLUMNS], df[LABEL_COL].astype(int).values
    groups = df["mint"].values
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        m = fit(df.iloc[tr], kind=kind, calibrate=False)
        oof[te] = m.predict(X.iloc[te])
    return oof, y


def main():
    tokens = _load_tokens()
    ex = build_snapshots(tokens)
    df = build_training_frame(ex)
    y = df[LABEL_COL].astype(int).values
    vsol = df["vsol"].values
    print(f"tokens (caught early): {len(tokens)}")
    print(f"snapshots: {len(df)}  (grad-token snaps={int(y.sum())} stall-token snaps={int((1-y).sum())})")
    print(f"snapshot vSol range: {vsol.min():.0f}..{vsol.max():.0f}\n")

    print("=== GroupKFold-by-mint OOS (no per-token leakage) ===")
    best = None
    for kind in ("logistic", "gbm"):
        oof, yy = group_cv(df, kind)
        overall = roc_auc_score(yy, oof)
        # the metric that matters: AUC at the entry-zone snapshots only
        ez = (vsol >= 33) & (vsol <= 50)
        ez_auc = roc_auc_score(yy[ez], oof[ez]) if len(set(yy[ez])) == 2 else float("nan")
        brier = brier_score_loss(yy, oof)
        print(f"  {kind:>8}: overall AUC={overall:.3f}  entry-zone[33-50] AUC={ez_auc:.3f}  Brier={brier:.3f}")
        if best is None or overall > best[1]:
            best = (kind, overall, oof)

    kind, _, oof = best
    print(f"\n=== {kind} OOS calibration across the whole trajectory ===")
    for lo, hi in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.01)]:
        idx = (oof >= lo) & (oof < hi)
        if idx.sum():
            print(f"  P[{lo:.1f},{hi:.1f}): n={int(idx.sum()):>5}  actual_grad_rate={y[idx].mean():.1%}")

    ms = ModelStore()
    if ms.latest_meta():
        import shutil, time as _t
        for e in ("pkl","json"):
            p = ms.dir / f"latest.{e}"
            if p.exists():
                shutil.copy(p, ms.dir / f"backup_{int(_t.time())}.{e}")
        print("\nbacked up previous model")
    final = fit(df, kind=kind, calibrate=True)
    print(f"saved {kind} -> {ms.save(final,'latest')}  (in-sample Brier={final.brier_score:.3f})")


if __name__ == "__main__":
    main()
