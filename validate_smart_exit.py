"""Validate the EXIT signal OOS: when >=2 distinct TRAIN-smart wallets SELL a token,
does the price fall MORE over the next window than a typical point in the same tokens?
If yes, "2+ smart sells -> exit" avoids drawdown; if not, it's noise.

No leakage: smart list built on ts<=cutoff (train); the signal is measured on holdout
tokens (ts>cutoff). Population = holdout tokens that reach the entry zone (vSol>=38,
caught clean) — the ones the strategy actually holds.
"""
from __future__ import annotations
import sqlite3
import numpy as np
from smart_wallets import score_wallets, TOP_K, EARLY_VSOL, PEAK_FRAC

DB = "pumpfun.db"
FLOOR, CATCH = 38.0, 32.0
WIN_SEC = 60.0          # forward window after a trigger / baseline point
N_SMART_SELLS = 2       # trigger threshold


def main():
    con = sqlite3.connect(DB)
    tmin, tmax = con.execute("SELECT MIN(ts), MAX(ts) FROM swaps").fetchone()
    cutoff = con.execute(
        "SELECT ts FROM swaps ORDER BY ts LIMIT 1 "
        "OFFSET (SELECT CAST(COUNT(*)*0.60 AS INT) FROM swaps)").fetchone()[0]
    smart = set(score_wallets(con, tmin - 1, cutoff).head(TOP_K).index)
    print(f"train-smart set: {len(smart)} wallets; testing on holdout (ts>{cutoff:.0f})\n")

    # holdout entry-zone, caught-clean mints
    mints = [m for (m,) in con.execute(
        "SELECT mint FROM tokens WHERE first_seen_ts > ? AND peak_vsol >= ?", (cutoff, FLOOR))]
    print(f"candidate holdout mints (peak>= {FLOOR}): {len(mints):,}")

    trig_fwd, base_fwd, trig_dd = [], [], []
    n_trig_tokens = 0
    for mint in mints:
        rows = con.execute(
            "SELECT ts, wallet, is_buy, vsol_after FROM swaps WHERE mint=? ORDER BY ts", (mint,)).fetchall()
        if len(rows) < 5 or rows[0][3] is None or rows[0][3] >= CATCH:
            continue
        # must reach entry zone
        if not any(r[3] is not None and r[3] >= FLOOR for r in rows):
            continue
        ts = np.array([r[0] for r in rows], float)
        v = np.array([r[3] if r[3] is not None else np.nan for r in rows], float)

        # forward price-return over WIN_SEC from index i (price ∝ v^2)
        def fwd_ret(i):
            j = np.searchsorted(ts, ts[i] + WIN_SEC)
            j = min(j, len(v) - 1)
            if np.isnan(v[i]) or np.isnan(v[j]) or v[i] <= 0:
                return None
            return (v[j] / v[i]) ** 2 - 1.0

        # trigger: index where the 2nd distinct smart wallet has SOLD
        sold = set(); trig_i = None
        for i, r in enumerate(rows):
            if r[2] == 0 and r[1] in smart:
                sold.add(r[1])
                if len(sold) >= N_SMART_SELLS:
                    trig_i = i
                    break
        if trig_i is not None:
            fr = fwd_ret(trig_i)
            if fr is not None:
                n_trig_tokens += 1
                trig_fwd.append(fr)
                # max drawdown over the rest of the token from trigger
                seg = v[trig_i:]
                seg = seg[~np.isnan(seg)]
                if len(seg) > 1 and seg[0] > 0:
                    trig_dd.append((seg.min() / seg[0]) ** 2 - 1.0)
        # baseline: forward return from every entry-zone swap (unconditional)
        for i in range(len(rows)):
            if v[i] >= FLOOR:
                fr = fwd_ret(i)
                if fr is not None:
                    base_fwd.append(fr)

    tf, bf = np.array(trig_fwd), np.array(base_fwd)
    print(f"\ntokens with a 2+ smart-sell trigger: {n_trig_tokens}")
    print(f"\n=== forward {WIN_SEC:.0f}s PRICE return ===")
    print(f"  after 2+ smart sells : n={len(tf):>5} mean {tf.mean():+.1%}  median {np.median(tf):+.1%}  P(down) {(tf<0).mean():.0%}")
    print(f"  baseline (any entry) : n={len(bf):>5} mean {bf.mean():+.1%}  median {np.median(bf):+.1%}  P(down) {(bf<0).mean():.0%}")
    if len(tf):
        print(f"  -> signal {tf.mean()-bf.mean():+.1%} vs baseline "
              f"[{'PREDICTS DROP' if tf.mean() < bf.mean()-0.02 else 'no edge'}]")
        td = np.array(trig_dd)
        print(f"\n  forward max drawdown after trigger: median {np.median(td):+.1%}  (worse-than -30%: {(td<-0.3).mean():.0%})")
    con.close()


if __name__ == "__main__":
    main()
