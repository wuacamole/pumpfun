"""TAIL-PREDICTABILITY TEST — the surviving edge hypothesis after [kill] winside-bracket.

The strategy died because GRADUATION is the wrong target: most graduations exit at a
loss (median 0.65x). But the payoff is heavy-tailed (max 96x). The only edge left is:
among tokens that GRADUATE, can PRE-graduation behaviour predict WHICH ones tail-win
(high post-graduation exit) vs dump? If yes, you enter those; if the tail is random
w.r.t. anything observable before graduating, the whole family is dead.

Honest design:
  * Universe: graduated tokens with a REAL recovered exit price, caught clean
    (first vSol < CATCH_VSOL), mature (right-censoring guard).
  * Target: payoff at a FIXED entry (vSol 40) = exit_price / curve_price(40). Label
    'profitable' = payoff > 1. Continuous log-payoff also reported.
  * Features: model.featurize at the entry-zone snapshot (first vSol >= ENTRY_VSOL),
    NO LOOKAHEAD (only swaps at/before the snapshot). The target is fully
    post-graduation, so features and target are time-separated.
  * Eval: time-ordered split by launch ts (train past -> predict future, like live).
    Compared against the trivial baseline (entry vSol alone). Small-n: report counts
    and a bootstrap CI on the AUC.
"""
from __future__ import annotations

import sqlite3
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import get_cached_constants
from model import FEATURE_COLUMNS, Swap, curve_price_per_token, featurize

DB = "pumpfun.db"
CATCH_VSOL = 32.0
ENTRY_VSOL = 36.0
MATURITY_SEC = 3600.0
ENTRY_PAYOFF_VSOL = 40.0  # value every position as if entered here (fixed, fair)


def load():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    max_ts = c.execute("SELECT MAX(ts) FROM swaps").fetchone()[0]
    cutoff = max_ts - MATURITY_SEC
    rows = c.execute(
        "SELECT t.mint, g.price_sol_per_unit FROM tokens t "
        "JOIN graduation_prices g ON g.mint = t.mint "
        "WHERE t.outcome='graduated' AND g.price_sol_per_unit > 0 "
        "AND t.first_seen_ts <= ?", (cutoff,)).fetchall()
    out = []
    for r in rows:
        swaps = [
            Swap(ts=s["ts"], wallet=s["wallet"], is_buy=bool(s["is_buy"]),
                 sol_amount=s["sol_amount"], vsol_after=s["vsol_after"],
                 via_frontend=bool(s["via_frontend"]),
                 is_direct=(None if s["is_direct"] is None else bool(s["is_direct"])),
                 entry_program=s["entry_program"])
            for s in c.execute(
                "SELECT ts,wallet,is_buy,sol_amount,vsol_after,via_frontend,is_direct,"
                "entry_program FROM swaps WHERE mint=? ORDER BY ts", (r["mint"],))
        ]
        if swaps and swaps[0].vsol_after < CATCH_VSOL:
            out.append((r["mint"], swaps, float(r["price_sol_per_unit"])))
    c.close()
    return out


def main():
    pc = get_cached_constants()
    entry_price = curve_price_per_token(ENTRY_PAYOFF_VSOL)
    toks = load()
    X, y, payoff, vsol_at_entry, launch = [], [], [], [], []
    for mint, swaps, exit_px in toks:
        as_of = next((s.ts for s in swaps if s.vsol_after >= ENTRY_VSOL), None)
        if as_of is None:
            continue
        seen = [s for s in swaps if s.ts <= as_of]
        if len(seen) < 5:
            continue
        feats = featurize(seen, as_of=as_of)
        X.append([feats.get(c, 0.0) for c in FEATURE_COLUMNS])
        p = exit_px / entry_price
        payoff.append(p)
        y.append(1 if p > 1.0 else 0)
        vsol_at_entry.append(seen[-1].vsol_after)
        launch.append(swaps[0].ts)
    X = np.array(X); y = np.array(y); payoff = np.array(payoff)
    vsol_at_entry = np.array(vsol_at_entry); launch = np.array(launch)
    n = len(y)
    print(f"priced graduates caught in entry zone: n={n}   "
          f"profitable (payoff>1x): {y.sum()} ({y.mean():.1%})   "
          f"median payoff {np.median(payoff):.2f}x   mean {payoff.mean():.2f}x")
    if n < 30 or y.sum() < 5 or (n - y.sum()) < 5:
        print("UNDERPOWERED — too few in a class to test predictability."); return

    # Time-ordered split: train on earliest 60%, test on latest 40% (live-like).
    order = np.argsort(launch)
    X, y, payoff, vsol_at_entry = X[order], y[order], payoff[order], vsol_at_entry[order]
    k = int(n * 0.6)
    Xtr, Xte, ytr, yte = X[:k], X[k:], y[:k], y[k:]
    print(f"\ntime-ordered split: train={len(ytr)} (grad earliest) test={len(yte)} "
          f"(test profitable rate {yte.mean():.1%})")
    if ytr.sum() < 3 or yte.sum() < 3 or (len(yte) - yte.sum()) < 3:
        print("test fold too thin for a stable AUC; reporting baselines only.")

    def auc_ci(yt, sc, n_boot=2000):
        if len(set(yt)) < 2:
            return float("nan"), (float("nan"), float("nan"))
        a = roc_auc_score(yt, sc)
        rng = np.random.default_rng(0); boots = []
        idx = np.arange(len(yt))
        for _ in range(n_boot):
            b = rng.choice(idx, len(idx), replace=True)
            if len(set(yt[b])) < 2:
                continue
            boots.append(roc_auc_score(yt[b], sc[b]))
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        return a, (lo, hi)

    # Baseline: entry vSol alone (the trivial level — lower vSol = cheaper entry =
    # higher payoff mechanically). The model must BEAT this to add anything.
    a_b, ci_b = auc_ci(yte, -vsol_at_entry[k:])  # lower vSol -> predict profitable
    print(f"\nBASELINE  entry-vSol-only AUC : {a_b:.3f}  95%CI[{ci_b[0]:.3f},{ci_b[1]:.3f}]")

    # Drop columns that are constant / all-NaN in the training fold (degenerate for
    # HistGBM binning and uninformative for logistic at this small n).
    valid = np.array([len(np.unique(Xtr[~np.isnan(Xtr[:, j]), j])) >= 2
                      for j in range(Xtr.shape[1])])
    cols_used = [c for c, v in zip(FEATURE_COLUMNS, valid) if v]
    Xtr, Xte = Xtr[:, valid], Xte[:, valid]
    print(f"usable features: {len(cols_used)}/{len(FEATURE_COLUMNS)}")

    for name, clf in [
        ("logistic", make_pipeline(SimpleImputer(strategy="median"),
                                   StandardScaler(), LogisticRegression(max_iter=2000))),
        ("gbm", HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                               learning_rate=0.05)),
    ]:
        clf.fit(Xtr, ytr)
        sc = clf.predict_proba(Xte)[:, 1]
        a, ci = auc_ci(yte, sc)
        print(f"MODEL     {name:8} OOS AUC      : {a:.3f}  95%CI[{ci[0]:.3f},{ci[1]:.3f}]"
              f"   lift vs vSol-baseline {a - a_b:+.3f}")

    # Univariate: Spearman of each feature with log-payoff (direction of any signal).
    from scipy.stats import spearmanr
    logp = np.log(np.clip(payoff, 1e-6, None))
    print("\ntop univariate |Spearman| feature vs log-payoff (all n, no split):")
    cors = []
    for j, c in enumerate(FEATURE_COLUMNS):
        col = X[:, j]
        m = ~np.isnan(col)
        if m.sum() < 3 or len(np.unique(col[m])) < 2:
            continue
        rho, p = spearmanr(col[m], logp[m])
        if not np.isnan(rho):
            cors.append((abs(rho), rho, p, c))
    for ar, rho, p, c in sorted(cors, reverse=True)[:6]:
        print(f"  {c:22} rho={rho:+.3f}  p={p:.3f}")


if __name__ == "__main__":
    main()
