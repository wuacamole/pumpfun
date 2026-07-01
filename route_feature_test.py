"""Route-category feature test — does *who routed the trade* separate graduates
AT FIXED vSol?

Context (DECISIONS.md [kill] 2026-06-27 + the is_direct route audit):
  * The model's apparent edge is a vSol-level artifact; behavioural bot features
    DEGRADE it. The bot axis it used (via_frontend size-heuristic, and the
    is_direct invoke-depth flag) is conflated:
      - is_direct=1 (top-level pump call) = pump.fun WEBSITE humans + naive bots
      - is_direct=0 (wrapped) = the sophisticated bots (flashx, mayhem, maestro)
        + Jupiter humans
    On-chain inspection confirmed flashx & mayhem are third-party automated
    trading wrappers, NOT the frontend. So the real human/bot split is per-PROGRAM
    (entry_program -> route_category), not per-invoke-depth.

This script builds route_category SHARE features (no lookahead) and asks the only
question that matters for the gate: in the entry zone [33,50] vSol, do route
features rank future-graduates above future-failures (time-ordered OOS AUC > 0.5),
and do they add anything over vSol alone? If a *correct* route taxonomy still
can't beat 0.5 at fixed vSol, that is K-criterion evidence: the paper's bot-share
premise does not hold in this data.

This is a TEST, not a model change: it does not touch FEATURE_COLUMNS or latest.pkl.
A feature must prove OOS lift here before it earns a place in the model (the same
bar top_holder_share was held to and failed).

Run: uv run python route_feature_test.py
"""
from __future__ import annotations

import sqlite3
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from conviction_engine import Swap
from helius import route_category
from refit_model import (
    CATCH_VSOL,
    GRAD_VSOL,
    MAX_SNAPS,
    MIN_SWAPS,
    SNAP_EVERY_SEC,
)

DB = "pumpfun.db"
# Only evaluate where entry_program is reliably populated. From the hourly
# coverage audit, entry_program null hits 0% from ~hr15; this is the same
# clean-window cutoff DECISIONS.md used (min_first_seen_ts). Before it, route is
# 'unknown' for ~everything and the shares carry no information.
ROUTE_MIN_TS = 1782517950.0
ENTRY_LO, ENTRY_HI = 33.0, 50.0  # the vSol band where the live gate fires
# Outcome-maturity guard. Graduation labels are real-time but a 'stalled' label
# is only written when the paper broker times a position out at entry +
# time_stop_sec (paper_trade.PaperBroker.time_stop_sec = 1800s / 30 min), so the
# most-recent cohort is RIGHT-CENSORED: its resolved tokens are artificially
# graduate-heavy (grad rate by launch hour climbs to a fake 100% in the final
# hours). Route-adoption also rose over the window, so route share proxies
# "launched late" => censored => fake-high graduation. Exclude tokens launched
# within MATURITY_SEC of the last observed swap. Buffer = time_stop_sec (1800) +
# entry-lag/settle cushion (~1800) = 3600. See the within-hour control in the
# DECISIONS audit: held at fixed launch hour, the route advantage disappears.
MATURITY_SEC = 3600.0

# Route-category share columns. route_category() maps the known programs to names
# (pump_direct/mayhem/maestro/jupiter/flashx), unknown programs to 'other', and a
# missing entry_program (None) to 'unknown'.
ROUTE_CATS = ["pump_direct", "mayhem", "maestro", "jupiter", "flashx", "other", "unknown"]
SHARE_COLS = [f"share_{c}" for c in ROUTE_CATS]
# The semantic grouping the on-chain audit established: the sophisticated bot
# wrappers vs the bare website/naive path. This is the variable the paper's
# "bot share" was *trying* to measure.
DERIVED_COLS = ["share_bot_wrapper", "share_web_or_naive", "n_routes"]
ROUTE_COLS = SHARE_COLS + DERIVED_COLS


def load_tokens() -> list[tuple[str, bool, list[Swap]]]:
    """(mint, graduated, swaps) for resolved tokens caught early, in the
    route-populated window. Ordered by first-swap ts so time folds are honest."""
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    out = []
    max_ts = c.execute("SELECT MAX(ts) FROM swaps").fetchone()[0]
    rows = c.execute(
        "SELECT mint, outcome FROM tokens "
        "WHERE outcome IN ('graduated','stalled') AND first_seen_ts >= ? "
        "AND first_seen_ts <= ?",  # maturity guard: settled outcomes only
        (ROUTE_MIN_TS, max_ts - MATURITY_SEC),
    ).fetchall()
    for r in rows:
        swaps = [
            Swap(
                ts=s["ts"], wallet=s["wallet"], is_buy=bool(s["is_buy"]),
                sol_amount=s["sol_amount"], vsol_after=s["vsol_after"],
                via_frontend=bool(s["via_frontend"]),
                is_direct=(None if s["is_direct"] is None else bool(s["is_direct"])),
                entry_program=s["entry_program"],
            )
            for s in c.execute(
                "SELECT ts,wallet,is_buy,sol_amount,vsol_after,via_frontend,"
                "is_direct,entry_program FROM swaps WHERE mint=? ORDER BY ts",
                (r["mint"],),
            )
        ]
        if swaps and swaps[0].vsol_after < CATCH_VSOL:
            out.append((r["mint"], r["outcome"] == "graduated", swaps))
    c.close()
    out.sort(key=lambda t: t[2][0].ts)  # chronological by first swap
    return out


def route_feats(seen: list[Swap]) -> dict[str, float]:
    """Route-category shares over swaps observed so far. No lookahead: caller
    passes only swaps with ts <= as_of."""
    n = len(seen)
    cnt: Counter[str] = Counter(route_category(s.entry_program) for s in seen)
    f = {f"share_{c}": cnt.get(c, 0) / n for c in ROUTE_CATS}
    f["share_bot_wrapper"] = (cnt.get("mayhem", 0) + cnt.get("flashx", 0)
                              + cnt.get("maestro", 0)) / n
    f["share_web_or_naive"] = cnt.get("pump_direct", 0) / n
    f["n_routes"] = float(len({route_category(s.entry_program)
                               for s in seen if s.entry_program}))
    return f


def build_frame(tokens) -> pd.DataFrame:
    """No-lookahead multi-snapshot frame: one row per (mint, as_of) decision
    point, with route features + vsol + label + the absolute first-swap ts used
    to order time folds."""
    rows = []
    for mint, grad, swaps in tokens:
        t0 = swaps[0].ts
        last_ts = swaps[-1].ts
        k, n = 1, 0
        while n < MAX_SNAPS:
            as_of = t0 + k * SNAP_EVERY_SEC
            k += 1
            if as_of > last_ts:
                break
            seen = [s for s in swaps if s.ts <= as_of]
            if len(seen) < MIN_SWAPS:
                continue
            if seen[-1].vsol_after >= GRAD_VSOL:
                break  # past graduation — not a decision point
            row = route_feats(seen)
            row["vsol"] = seen[-1].vsol_after
            row[LABEL] = int(grad)
            row["mint"] = mint
            row["t0"] = t0
            rows.append(row)
            n += 1
    return pd.DataFrame(rows)


LABEL = "graduated"


def _make_clf(kind: str):
    if kind == "logistic":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    return GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=42,
    )


def time_ordered_oos(df: pd.DataFrame, cols: list[str], kind: str,
                     n_splits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Expanding-window, grouped-by-mint OOS predictions. Mints are ordered by
    first-swap ts; fold k trains on all earlier mints and tests on fold k. All
    snapshots of one mint stay in one fold (no per-token leakage). Mirrors
    refit_model.time_ordered_cv but on an explicit ts order and arbitrary cols."""
    X = df[cols].fillna(0.0).values
    y = df[LABEL].astype(int).values
    mints = df["mint"].values
    # mint -> first ts, then rank
    first_ts = df.groupby("mint")["t0"].min().to_dict()
    ordered = sorted(first_ts, key=lambda m: first_ts[m])
    fold_size = max(1, len(ordered) // n_splits)
    mint_fold = {m: min(i // fold_size, n_splits - 1) for i, m in enumerate(ordered)}
    fold = np.array([mint_fold[m] for m in mints])
    oof = np.full(len(y), np.nan)
    for k in range(1, n_splits):
        tr, te = fold < k, fold == k
        if not te.any() or not tr.any() or len(set(y[tr])) < 2:
            continue
        clf = _make_clf(kind).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    valid = ~np.isnan(oof)
    return oof[valid], df.index[valid].values


def auc_at(df, oof, idx, mask_extra=None):
    y = df.loc[idx, LABEL].astype(int).values
    p = oof
    if mask_extra is not None:
        m = mask_extra.loc[idx].values
        y, p = y[m], p[m]
    return roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")


def main():
    tokens = load_tokens()
    n_grad = sum(1 for _, g, _ in tokens if g)
    print(f"route-window tokens (caught <{CATCH_VSOL:.0f} vSol, first_seen>={ROUTE_MIN_TS:.0f}): "
          f"{len(tokens)}  (graduated={n_grad}, stalled={len(tokens)-n_grad})")

    df = build_frame(tokens).reset_index(drop=True)
    if df.empty:
        print("no snapshots — aborting")
        return
    # Launch-time control in [0,1]. The baseline includes this so route features
    # must beat a model that ALREADY knows when the token launched — neutralising
    # the censoring/time-cohort confound that the raw route share rides on.
    lo, hi = df["t0"].min(), df["t0"].max()
    df["launch_frac"] = (df["t0"] - lo) / (hi - lo) if hi > lo else 0.0
    y = df[LABEL].values
    ez = (df["vsol"] >= ENTRY_LO) & (df["vsol"] <= ENTRY_HI)
    print(f"snapshots: {len(df)}  (grad-snaps={int(y.sum())}, stall-snaps={int((1-y).sum())})")
    print(f"entry-zone[{ENTRY_LO:.0f},{ENTRY_HI:.0f}] snapshots: {int(ez.sum())} "
          f"(grad={int(y[ez].sum())})\n")

    # ---- 1. Univariate: route shares, graduates vs non-graduates (entry zone)
    print("=== entry-zone route-share means: graduate vs non-graduate snapshots ===")
    g, ng = df[ez & (df[LABEL] == 1)], df[ez & (df[LABEL] == 0)]
    print(f"  {'feature':<20} {'grad':>8} {'non-grad':>9} {'uni-AUC':>8}")
    for col in ROUTE_COLS:
        ua = (roc_auc_score(y[ez], df.loc[ez, col].values)
              if len(set(y[ez])) == 2 else float("nan"))
        print(f"  {col:<20} {g[col].mean():>8.3f} {ng[col].mean():>9.3f} {ua:>8.3f}")

    # ---- 2. Entry-zone time-ordered OOS AUC: route-only vs vsol-only vs combined
    print("\n=== time-ordered OOS AUC (honest; predicts the future) ===")
    print(f"  {'feature set':<22} {'kind':>9} {'overall':>8} {'entry-zone':>11}")
    feature_sets = {
        "vsol+time (baseline)": ["vsol", "launch_frac"],
        "route_only": ROUTE_COLS,
        "route+vsol+time": ["vsol", "launch_frac"] + ROUTE_COLS,
    }
    results = {}
    for name, cols in feature_sets.items():
        for kind in ("logistic", "gbm"):
            oof, idx = time_ordered_oos(df, cols, kind)
            overall = auc_at(df, oof, idx)
            ez_idx = ez.loc[idx]
            ez_auc = auc_at(df, oof, idx, mask_extra=ez)
            results[(name, kind)] = ez_auc
            print(f"  {name:<22} {kind:>9} {overall:>8.3f} {ez_auc:>11.3f}")

    # ---- 3. Verdict. Route must beat the time-controlled baseline in BOTH model
    # classes (a GBM-only lift is an overfit/leak signature, not a robust edge),
    # by a margin (LIFT_MIN) larger than fold noise at this sample size.
    LIFT_MIN = 0.02
    print("\n=== VERDICT (entry-zone OOS AUC vs a baseline that already knows launch time) ===")
    lifts = {k: results[("route+vsol+time", k)] - results[("vsol+time (baseline)", k)]
             for k in ("logistic", "gbm")}
    for k in ("logistic", "gbm"):
        base, comb = results[("vsol+time (baseline)", k)], results[("route+vsol+time", k)]
        print(f"  {k:>8}: baseline(vsol+time)={base:.3f}  +route={comb:.3f}  "
              f"lift={lifts[k]:+.3f}")
    robust = all(v == v and v > LIFT_MIN for v in lifts.values())  # both, and not NaN
    if robust:
        print(f"  => route features add >{LIFT_MIN} entry-zone AUC in BOTH model classes,\n"
              "     net of the launch-time confound — candidate for a full refit_model A/B.")
    else:
        print("  => route features do NOT robustly beat a time-aware baseline.\n"
              "     The raw cross-sectional route signal was a right-censoring / time-cohort\n"
              "     artifact (route adoption rose over the window; recent resolved tokens are\n"
              "     fake-graduate-heavy). Held at fixed launch time the edge disappears.\n"
              "     Consistent with DECISIONS.md [kill]: the paper's bot-share premise does\n"
              "     not hold here. K-criterion evidence, not a new edge.")


if __name__ == "__main__":
    main()
