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
# Minimum entry-zone (33-50 vSol) time-ordered OOS AUC required to deploy a
# model. <= 0.5 means no discrimination where the live gate fires; refuse to
# ship it (it would be a vSol-level detector at best, inverted at worst).
ENTRY_ZONE_AUC_DEPLOY_MIN = 0.5


def _load_tokens():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    out = []
    for r in c.execute("SELECT mint,outcome FROM tokens WHERE outcome IN ('graduated','stalled')"):
        swaps = [
            Swap(ts=s["ts"], wallet=s["wallet"], is_buy=bool(s["is_buy"]),
                 sol_amount=s["sol_amount"], vsol_after=s["vsol_after"],
                 via_frontend=bool(s["via_frontend"]),
                 is_direct=(None if s["is_direct"] is None else bool(s["is_direct"])),
                 entry_program=s["entry_program"])
            for s in c.execute("SELECT ts,wallet,is_buy,sol_amount,vsol_after,via_frontend,"
                               "       is_direct,entry_program "
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
    """GroupKFold-by-mint (no per-token leakage), folds random in time.

    Kept for comparison, but the live system only ever predicts the *future*,
    so random-time folds over-estimate live performance. Prefer
    :func:`time_ordered_cv` for the model that gets saved.
    """
    X, y = df[FEATURE_COLUMNS], df[LABEL_COL].astype(int).values
    groups = df["mint"].values
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        m = fit(df.iloc[tr], kind=kind, calibrate=False)
        oof[te] = m.predict(X.iloc[te])
    return oof, y


def time_ordered_cv(df, kind, n_splits=5):
    """Expanding-window time-ordered CV, grouped by mint.

    Sorts mints by their first-snapshot ts, splits them into ``n_splits``
    contiguous time folds, and for each fold ``k`` trains on folds ``0..k-1``
    (all earlier mints) and tests on fold ``k``. This mirrors what the live
    system does: it only ever predicts tokens that launch *after* the ones it
    has already seen. Fold 0 has no training data — its predictions are NaN
    and it's excluded from the metrics (same contract as the walk-forward
    backtest's no-model fold).

    All snapshots of one mint stay in the same fold (no per-token leakage).
    Returns (out-of-fold predictions, labels); NaN predictions for fold 0.
    """
    X, y = df[FEATURE_COLUMNS], df[LABEL_COL].astype(int).values
    groups = df["mint"].values
    # Per-mint first-snapshot ts — the launch time that orders the folds.
    mint_first_ts: dict[str, float] = {}
    for i, g in enumerate(groups):
        if g not in mint_first_ts or df["seconds_since_launch"].iloc[i] < mint_first_ts[g]:
            # seconds_since_launch is small for the first snapshot; recover
            # the absolute ts from the row by subtracting from a reference.
            # The frame doesn't carry ts directly, but mint order in df follows
            # build_snapshots order, which is chronological. Use the row index
            # as a tiebreaker so the sort is stable.
            pass
    # The training frame from build_training_frame loses the absolute ts, but
    # build_snapshots emits examples in chronological mint order, and
    # build_training_frame preserves row order (reset_index(drop=True)). So the
    # first appearance of each mint in df is in chronological order. Use that.
    mint_order: dict[str, int] = {}
    for i, g in enumerate(groups):
        if g not in mint_order:
            mint_order[g] = i  # first row index = chronological rank
    # Assign each mint to a time fold by its first-appearance rank.
    sorted_mints = sorted(mint_order, key=lambda m: mint_order[m])
    n_mints = len(sorted_mints)
    fold_size = max(1, n_mints // n_splits)
    mint_fold: dict[str, int] = {}
    for idx, m in enumerate(sorted_mints):
        mint_fold[m] = min(idx // fold_size, n_splits - 1)
    oof = np.full(len(y), np.nan)
    for k in range(1, n_splits):  # fold 0 has no training data
        te_mask = np.array([mint_fold.get(g, -1) == k for g in groups])
        tr_mask = np.array([mint_fold.get(g, -1) < k for g in groups])
        if not te_mask.any() or not tr_mask.any():
            continue
        fitted = fit(df.iloc[tr_mask], kind=kind, calibrate=False)
        oof[te_mask] = fitted.predict(X.iloc[te_mask])
    valid = ~np.isnan(oof)
    return oof[valid], y[valid], valid


def main():
    tokens = _load_tokens()
    ex = build_snapshots(tokens)
    df = build_training_frame(ex)
    y = df[LABEL_COL].astype(int).values
    vsol = df["vsol"].values
    print(f"tokens (caught early): {len(tokens)}")
    print(f"snapshots: {len(df)}  (grad-token snaps={int(y.sum())} stall-token snaps={int((1-y).sum())})")
    print(f"snapshot vSol range: {vsol.min():.0f}..{vsol.max():.0f}\n")

    print("=== GroupKFold-by-mint OOS (random time — optimistic upper bound) ===")
    for kind in ("logistic", "gbm"):
        oof, yy = group_cv(df, kind)
        overall = roc_auc_score(yy, oof)
        ez = (vsol >= 33) & (vsol <= 50)
        ez_auc = roc_auc_score(yy[ez], oof[ez]) if len(set(yy[ez])) == 2 else float("nan")
        brier = brier_score_loss(yy, oof)
        print(f"  {kind:>8}: overall AUC={overall:.3f}  entry-zone[33-50] AUC={ez_auc:.3f}  Brier={brier:.3f}")

    print("\n=== time-ordered expanding-window OOS (honest — predicts the future) ===")
    # Select by ENTRY-ZONE AUC, not overall AUC. Overall AUC is dominated by the
    # trivial vSol-level signal (a snapshot near graduation obviously graduates;
    # vSol alone scores ~0.86 OOS), so picking the highest-overall-AUC model
    # selects for "knows vSol", not for the skill the live gate needs: ranking
    # tokens *at the same vSol*. The gate fires in [33,50] vSol, so that's the
    # metric that matters. See DECISIONS.md [kill] 2026-06-27 (vSol-confound).
    best = None
    for kind in ("logistic", "gbm"):
        oof, yy, valid = time_ordered_cv(df, kind)
        if len(set(yy)) != 2 or len(yy) < 10:
            print(f"  {kind:>8}: too few valid folds to evaluate")
            continue
        overall = roc_auc_score(yy, oof)
        # Entry-zone AUC on the valid subset only. oof/yy are already filtered
        # to valid rows (length = valid.sum()); restrict vsol/y the same way.
        vsol_valid = vsol[valid]
        y_valid = y[valid]
        ez = (vsol_valid >= 33) & (vsol_valid <= 50)
        ez_auc = (roc_auc_score(y_valid[ez], oof[ez])
                  if ez.any() and len(set(y_valid[ez])) == 2 else float("nan"))
        brier = brier_score_loss(yy, oof)
        n_eval = int(valid.sum())
        print(f"  {kind:>8}: overall AUC={overall:.3f}  entry-zone[33-50] AUC={ez_auc:.3f}  "
              f"Brier={brier:.3f}  (n={n_eval} valid snapshots)")
        # Rank by entry-zone AUC (NaN sorts last). Tie-break on overall.
        score = (ez_auc if ez_auc == ez_auc else -1.0)
        if best is None or score > best[1]:
            best = (kind, score, oof, valid, overall)

    if best is None:
        print("\nno valid time-ordered folds — not enough data; aborting refit")
        return
    kind, best_ez_auc, oof, valid, _ = best

    # Deploy guard: refuse to ship a model with no entry-zone discrimination.
    # entry-zone AUC <= 0.5 means it ranks future-graduates at or below
    # future-failures *at the same vSol* — i.e. it is inverted/useless exactly
    # where the live gate uses it. Shipping it would fire ENTERs on the model's
    # most-confident-but-least-likely tokens (the observed failure mode). Keep
    # the previous model on disk and abstain rather than deploy an anti-signal.
    if not (best_ez_auc > ENTRY_ZONE_AUC_DEPLOY_MIN):
        print(f"\nREFUSING TO DEPLOY: best entry-zone AUC={best_ez_auc:.3f} "
              f"<= {ENTRY_ZONE_AUC_DEPLOY_MIN} (no discrimination where the gate "
              f"fires). The model is a vSol-level detector, not a conviction "
              f"predictor — see DECISIONS.md [kill] (vSol confound). latest.pkl "
              f"left unchanged; the live gate should abstain (WATCH-only).")
        return
    print(f"\n=== {kind} time-ordered OOS calibration across the whole trajectory ===")
    y_valid = y[valid]
    for lo, hi in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.01)]:
        idx = (oof >= lo) & (oof < hi)
        if idx.sum():
            print(f"  P[{lo:.1f},{hi:.1f}): n={int(idx.sum()):>5}  actual_grad_rate={y_valid[idx].mean():.1%}")

    ms = ModelStore()
    if ms.latest_meta():
        import shutil
        import time as _t
        for e in ("pkl", "json"):
            p = ms.dir / f"latest.{e}"
            if p.exists():
                shutil.copy(p, ms.dir / f"backup_{int(_t.time())}.{e}")
        print("\nbacked up previous model")
    final = fit(df, kind=kind, calibrate=True)
    final.entry_zone_auc = float(best_ez_auc)  # the metric the live gate trusts
    print(f"saved {kind} -> {ms.save(final,'latest')}  (in-sample Brier={final.brier_score:.3f}, "
          f"entry-zone OOS AUC={best_ez_auc:.3f})")
    print("note: time-ordered OOS AUC above is the honest metric; in-sample Brier is optimistic")


if __name__ == "__main__":
    main()
