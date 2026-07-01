"""§5.4 REGIME TEST — is the scale-out edge regime-specific, or does it survive in
disjoint market windows?

The forward paper-test is a real-time OOS on the latest window. stress_scaleout.py
splits into thirds but fits nothing walk-forward. This goes further:

  * Split the 72h of history into N sequential, DISJOINT entry-time windows (regimes).
  * For each window k, every FITTED decision (the smart-wallet list, used for the
    smart-exit override + smart-buy confidence) is built from data STRICTLY BEFORE
    window k. No window sees its own future, and no later window leaks back.
  * The mechanical scale-out (50%@1.3x, runner 1.7x/breakeven, hard-SL 0.2x, floor=38)
    is fixed — it was the param set already validated; we are NOT re-tuning it here.
  * Each window is scored independently: win / mean / median / bootstrap CI-low.

Pass bar: the edge (positive mean, and CI-low not deeply negative given the small
per-window n) appears in EVERY regime window, not just pooled. A window that flips
hard negative = the edge is regime-specific = do not deploy.
"""
from __future__ import annotations
import sqlite3
import numpy as np
from peak_exit_replay import load
from curve_slippage_replay import ci_low, k_whole
from config import get_cached_constants
from stress_scaleout import scaleout_ret, FLOOR
from smart_wallets import score_wallets, TOP_K

pc = get_cached_constants(); K = k_whole(pc); GRAD_V = pc.v_sol_graduation
DB = "pumpfun.db"
N_WINDOWS = 6
S = 1.0
MEV = 0.01


def build_entries():
    """Each entry: (swaps, i0, entry_ts). Caught-clean, floor=38, >=2 pre-entry swaps."""
    ent = []
    for mint, grad, swaps in load():
        i0 = next((i for i, s in enumerate(swaps) if s.vsol_after >= FLOOR), None)
        if i0 is None or i0 < 2:
            continue
        ent.append((swaps, i0, swaps[i0].ts))
    return ent


def smart_exit_ret(swaps, i0, smart):
    """Scale-out, but force-close the remainder the moment a 2nd distinct smart wallet
    sells. Returns (ret, fired)."""
    base = scaleout_ret(swaps, i0, S, MEV)
    if not smart:
        return base, False
    from stress_scaleout import tr, FIRST_MULT, FIRST_FRAC, RUNNER_TP, SL
    V_e = swaps[i0].vsol_after
    first_r, tp_r, sl_r = V_e*np.sqrt(FIRST_MULT), V_e*np.sqrt(RUNNER_TP), V_e*np.sqrt(SL)
    banked, r1, sold = False, 0.0, set()
    for s in swaps[i0 + 1:]:
        v = s.vsol_after
        if v >= GRAD_V:
            break
        if (not s.is_buy) and (s.wallet in smart):
            sold.add(s.wallet)
            if len(sold) >= 2:
                if banked:
                    return FIRST_FRAC*r1 + (1-FIRST_FRAC)*tr(V_e, v, S*(1-FIRST_FRAC), MEV), True
                return tr(V_e, v, S, MEV), True
        if not banked:
            if v <= sl_r:
                break
            if v >= first_r:
                banked = True; r1 = tr(V_e, v, S*FIRST_FRAC, MEV)
        else:
            if v >= tp_r or v <= V_e:
                break
    return base, False


def row(name, r, extra=""):
    r = np.array(r)
    cl = ci_low(r) if len(r) > 1 else float("nan")
    flag = "PASS" if (len(r) > 1 and r.mean() > 0 and cl > -0.05) else "FAIL"
    print(f"  {name:<22} n={len(r):>4} win={(r>0).mean():>5.1%} mean={r.mean():>+6.1%} "
          f"med={np.median(r):>+6.1%} CI-low={cl:>+6.1%} [{flag}] {extra}")


def main():
    ent = build_entries()
    ent.sort(key=lambda e: e[2])
    tmin, tmax = ent[0][2], ent[-1][2]
    edges = np.linspace(tmin, tmax, N_WINDOWS + 1)
    con = sqlite3.connect(DB)
    print(f"regime test: {len(ent)} entries over {(tmax-tmin)/3600:.1f}h, "
          f"{N_WINDOWS} windows (~{(tmax-tmin)/3600/N_WINDOWS:.1f}h each)\n")

    print("== A. SCALE-OUT edge per disjoint regime window (no fitting; mechanical) ==")
    seg_rets = []
    for k in range(N_WINDOWS):
        lo, hi = edges[k], edges[k + 1]
        seg = [e for e in ent if lo <= e[2] < hi] if k < N_WINDOWS - 1 else \
              [e for e in ent if lo <= e[2] <= hi]
        r = [scaleout_ret(sw, i, S, MEV) for sw, i, _ in seg]
        seg_rets.append(r)
        row(f"W{k+1}", r)

    print("\n== B. WALK-FORWARD smart list (fit on data BEFORE each window) ==")
    print("   smart-exit override + buy-confidence, list built strictly from the past.\n")
    for k in range(1, N_WINDOWS):          # W1 has no past to fit on
        lo, hi = edges[k], edges[k + 1]
        seg = [e for e in ent if lo <= e[2] < hi] if k < N_WINDOWS - 1 else \
              [e for e in ent if lo <= e[2] <= hi]
        # smart list from all swaps strictly before this window's start
        smart = set(score_wallets(con, tmin - 1, lo).head(TOP_K).index)
        base, withx, fired = [], [], 0
        for sw, i, _ in seg:
            b = scaleout_ret(sw, i, S, MEV)
            w, f = smart_exit_ret(sw, i, smart)
            base.append(b); withx.append(w); fired += f
        row(f"W{k+1} base", base)
        row(f"W{k+1} +smart", withx, extra=f"(smart={len(smart)}, exit fired {fired})")

    print("\n== C. pooled (all windows, mechanical scale-out) ==")
    allr = [x for r in seg_rets for x in r]
    row("pooled", allr)
    con.close()


if __name__ == "__main__":
    main()
