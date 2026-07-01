"""Integration A/B: does adding "exit on 2+ smart sells" IMPROVE the scale-out strategy
on the holdout? Simulate each holdout entry-zone token with the real scale-out exit
(bank 50%@1.3x, runner 1.7x/breakeven, hard-SL 0.2x), WITH vs WITHOUT the smart-exit
override (force-close the remainder the moment a 2nd distinct train-smart wallet sells).

No leakage: smart list = train (ts<=cutoff); tested on holdout tokens.
"""
from __future__ import annotations
import sqlite3
import numpy as np
from config import get_cached_constants
from curve_slippage_replay import trade_return, ci_low, k_whole
from smart_wallets import score_wallets, TOP_K

DB = "pumpfun.db"
FLOOR, CATCH = 38.0, 32.0
FIRST_MULT, FIRST_FRAC, TP, SL = 1.3, 0.5, 1.7, 0.2
S = 1.0


def scaleout(rows, i0, K, grad_v, smart=None):
    """rows: list of (ts,wallet,is_buy,vsol). Returns (ret, used_smart_exit)."""
    V_e = rows[i0][3]
    first_r, tp_r, sl_r = V_e*np.sqrt(FIRST_MULT), V_e*np.sqrt(TP), V_e*np.sqrt(SL)
    banked = False; r1 = 0.0; sold = set()
    for r in rows[i0+1:]:
        ts, w, is_buy, v = r
        if v is None or v >= grad_v:
            ve = grad_v if (v is not None and v >= grad_v) else V_e
            if banked:
                return FIRST_FRAC*r1 + (1-FIRST_FRAC)*trade_return(V_e, ve, S*(1-FIRST_FRAC), K), False
            return trade_return(V_e, ve, S, K), False
        # smart-exit override
        if smart is not None and is_buy == 0 and w in smart:
            sold.add(w)
            if len(sold) >= 2:
                if banked:
                    return FIRST_FRAC*r1 + (1-FIRST_FRAC)*trade_return(V_e, v, S*(1-FIRST_FRAC), K), True
                return trade_return(V_e, v, S, K), True
        if not banked:
            if v <= sl_r:
                return trade_return(V_e, sl_r, S, K), False
            if v >= first_r:
                banked = True
                r1 = trade_return(V_e, v, S*FIRST_FRAC, K)
        else:
            if v >= tp_r:
                return FIRST_FRAC*r1 + (1-FIRST_FRAC)*trade_return(V_e, v, S*(1-FIRST_FRAC), K), False
            if v <= V_e:
                return FIRST_FRAC*r1 + (1-FIRST_FRAC)*trade_return(V_e, V_e, S*(1-FIRST_FRAC), K), False
    last = next((r[3] for r in reversed(rows[i0+1:]) if r[3] is not None), V_e)
    if banked:
        return FIRST_FRAC*r1 + (1-FIRST_FRAC)*trade_return(V_e, last, S*(1-FIRST_FRAC), K), False
    return trade_return(V_e, last, S, K), False


def summary(name, r):
    r = np.array(r)
    print(f"  {name:<26} n={len(r):>4} win={(r>0).mean():>5.1%} mean={r.mean():>+6.1%} "
          f"med={np.median(r):>+6.1%} CI={ci_low(r):>+6.1%}")


def main():
    pc = get_cached_constants(); K = k_whole(pc); grad_v = pc.v_sol_graduation
    con = sqlite3.connect(DB)
    tmin, tmax = con.execute("SELECT MIN(ts),MAX(ts) FROM swaps").fetchone()
    cutoff = con.execute("SELECT ts FROM swaps ORDER BY ts LIMIT 1 "
                         "OFFSET (SELECT CAST(COUNT(*)*0.60 AS INT) FROM swaps)").fetchone()[0]
    smart = set(score_wallets(con, tmin-1, cutoff).head(TOP_K).index)
    mints = [m for (m,) in con.execute(
        "SELECT mint FROM tokens WHERE first_seen_ts>? AND peak_vsol>=?", (cutoff, FLOOR))]
    print(f"smart={len(smart)} | holdout entry-zone mints={len(mints):,}\n")

    base, withx, n_fired = [], [], 0
    for mint in mints:
        rows = con.execute("SELECT ts,wallet,is_buy,vsol_after FROM swaps WHERE mint=? ORDER BY ts",
                           (mint,)).fetchall()
        if len(rows) < 5 or rows[0][3] is None or rows[0][3] >= CATCH:
            continue
        i0 = next((i for i, r in enumerate(rows) if r[3] is not None and r[3] >= FLOOR), None)
        if i0 is None or i0 < 2:
            continue
        rb, _ = scaleout(rows, i0, K, grad_v, smart=None)
        rw, fired = scaleout(rows, i0, K, grad_v, smart=smart)
        base.append(rb); withx.append(rw); n_fired += fired

    print(f"positions={len(base)}  smart-exit fired on {n_fired} ({n_fired/max(len(base),1):.0%})\n")
    print("=== HOLDOUT scale-out A/B ===")
    summary("baseline (no smart-exit)", base)
    summary("+ smart-exit (2+ sells)", withx)
    b, w = np.array(base), np.array(withx)
    print(f"\n  delta mean {w.mean()-b.mean():+.2%}  delta win {(w>0).mean()-(b>0).mean():+.1%}  "
          f"delta median {np.median(w)-np.median(b):+.2%}")
    # on the subset where it fired, did it help?
    fired_mask = w != b
    if fired_mask.any():
        print(f"\n  on the {fired_mask.sum()} positions where it changed the exit:")
        summary("    baseline (those)", b[fired_mask])
        summary("    + smart-exit (those)", w[fired_mask])
    con.close()


if __name__ == "__main__":
    main()
