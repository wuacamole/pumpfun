"""Smart-wallet tracker built from pumpfun.db swap history (SQL-first; ~1GB RAM safe).

Scores each wallet on behaviour that PRECEDES profit (so it applies forward without
hindsight). Per (wallet, token) the wallet bought:
  * realized PnL  = sum(sell SOL) - sum(buy SOL)        -> win = PnL>0
  * early entry   = min buy vSol < EARLY_VSOL            (caught on the curve early)
  * sold near peak= max sell vSol >= PEAK_FRAC*peak_vsol (good exit timing)
  * graduation    = token.outcome == 'graduated'
Per-wallet (>= MIN_TOKENS tokens): win_rate, early_rate, near_peak_rate, grad_rate,
total_pnl. Score = weighted percentile-rank blend. All the per-(wallet,mint) and
per-wallet aggregation runs INSIDE SQLite; only the small eligible-wallet table comes
back to Python.

DISCIPLINE: list is built on a TRAINING window and its skill is checked OOS on the
held-out tail — a wallet is "smart" only if its win-rate PERSISTS. The live list must
be built strictly before the live period (no leakage).

Run: uv run python smart_wallets.py   ->  writes smart_wallets.json (+ OOS report)
"""
from __future__ import annotations
import json
import sqlite3
import numpy as np
import pandas as pd

DB = "pumpfun.db"
EARLY_VSOL = 45.0
PEAK_FRAC = 0.80
MIN_TOKENS = 10
TOP_K = 400
TRAIN_FRAC = 0.60

_AGG_SQL = """
SELECT wallet,
       COUNT(*)                         AS n_tokens,
       AVG(win)                         AS win_rate,
       SUM(pnl)                         AS total_pnl,
       AVG(early)                       AS early_rate,
       AVG(near_peak)                   AS near_peak_rate,
       SUM(graduated)                   AS n_grads
FROM (
  SELECT s.wallet AS wallet, s.mint AS mint,
    (SUM(CASE WHEN s.is_buy=0 THEN s.sol_amount ELSE 0 END)
     - SUM(CASE WHEN s.is_buy=1 THEN s.sol_amount ELSE 0 END))            AS pnl,
    CASE WHEN (SUM(CASE WHEN s.is_buy=0 THEN s.sol_amount ELSE 0 END)
     - SUM(CASE WHEN s.is_buy=1 THEN s.sol_amount ELSE 0 END)) > 0 THEN 1 ELSE 0 END AS win,
    CASE WHEN MIN(CASE WHEN s.is_buy=1 THEN s.vsol_after END) < {early} THEN 1 ELSE 0 END AS early,
    COALESCE(CASE WHEN MAX(CASE WHEN s.is_buy=0 THEN s.vsol_after END)
                       >= {peak}*t.peak_vsol THEN 1 ELSE 0 END, 0)         AS near_peak,
    CASE WHEN t.outcome='graduated' THEN 1 ELSE 0 END                      AS graduated
  FROM swaps s LEFT JOIN tokens t ON s.mint=t.mint
  WHERE s.ts > {lo} AND s.ts <= {hi}
  GROUP BY s.wallet, s.mint
  HAVING SUM(CASE WHEN s.is_buy=1 THEN 1 ELSE 0 END) >= 1
)
GROUP BY wallet
HAVING n_tokens >= {mint}
"""


def score_wallets(con, lo: float, hi: float) -> pd.DataFrame:
    sql = _AGG_SQL.format(early=EARLY_VSOL, peak=PEAK_FRAC, lo=lo, hi=hi, mint=MIN_TOKENS)
    w = pd.read_sql_query(sql, con).set_index("wallet")
    w["grad_rate"] = w.n_grads / w.n_tokens
    pr = lambda s: s.rank(pct=True)
    w["score"] = (0.40 * pr(w.win_rate) + 0.20 * pr(w.early_rate)
                  + 0.20 * pr(w.near_peak_rate) + 0.20 * pr(w.grad_rate))
    return w.sort_values("score", ascending=False)


def main():
    con = sqlite3.connect(DB)
    tmin, tmax = con.execute("SELECT MIN(ts), MAX(ts) FROM swaps").fetchone()
    cutoff = con.execute(
        f"SELECT ts FROM swaps ORDER BY ts LIMIT 1 "
        f"OFFSET (SELECT CAST(COUNT(*)*{TRAIN_FRAC} AS INT) FROM swaps)").fetchone()[0]
    print(f"swaps span {tmin:.0f}..{tmax:.0f}; train cutoff ts={cutoff:.0f} ({TRAIN_FRAC:.0%})\n")

    wtrain = score_wallets(con, tmin - 1, cutoff)
    smart = wtrain.head(TOP_K)
    print(f"eligible wallets (>= {MIN_TOKENS} tokens, train): {len(wtrain):,}; smart list = top {TOP_K}\n")
    print("top 12 smart wallets (train):")
    print(f"  {'wallet':<10}{'n':>5}{'win%':>7}{'early%':>8}{'peak%':>7}{'grads':>7}{'pnl_sol':>10}{'score':>7}")
    for wid, r in smart.head(12).iterrows():
        print(f"  {wid[:8]:<10}{int(r.n_tokens):>5}{r.win_rate:>6.0%}{r.early_rate:>8.0%}"
              f"{r.near_peak_rate:>7.0%}{int(r.n_grads):>7}{r.total_pnl:>+10.1f}{r.score:>7.2f}")

    # ---- OOS PERSISTENCE ----
    whold = score_wallets(con, cutoff, tmax + 1)
    sids = set(smart.index)
    inh = whold[whold.index.isin(sids)]; base = whold[~whold.index.isin(sids)]
    print("\n=== OOS PERSISTENCE (holdout win-rate: TRAIN-smart vs the rest) ===")
    if len(inh) >= 10:
        lift = inh.win_rate.mean() - base.win_rate.mean()
        print(f"  train-smart active in holdout: {len(inh)}  (median {inh.n_tokens.median():.0f} tokens)")
        print(f"    holdout win-rate : {inh.win_rate.mean():.1%}   near-peak {inh.near_peak_rate.mean():.1%}"
              f"   grad-rate {inh.grad_rate.mean():.2%}")
        print(f"    baseline win-rate: {base.win_rate.mean():.1%}")
        print(f"    -> LIFT {lift:+.1%}  [{'PERSISTS' if lift > 0.03 else 'NO EDGE / hindsight'}]")
    else:
        print(f"  too few re-active ({len(inh)}) for a clean test")

    # ---- production list (built on ALL swaps up to now) ----
    prod = score_wallets(con, tmin - 1, tmax + 1).head(TOP_K)
    out = {"built_at": float(tmax), "n": int(len(prod)),
           "params": {"EARLY_VSOL": EARLY_VSOL, "PEAK_FRAC": PEAK_FRAC,
                      "MIN_TOKENS": MIN_TOKENS, "TOP_K": TOP_K},
           "wallets": list(prod.index)}
    json.dump(out, open("smart_wallets.json", "w"))
    con.close()
    print(f"\nwrote smart_wallets.json: {len(prod)} wallets (built on full history up to {tmax:.0f})")


if __name__ == "__main__":
    main()
