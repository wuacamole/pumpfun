"""PRE-GRADUATION PEAK-EXIT REPLAY — the one direction that sidesteps the liquidity wall.

The graduation strategy died because post-graduation PumpSwap pools are dead (~80%
unsellable; [kill] liquidity-wall). But the BONDING CURVE is liquid by construction —
you can always sell back to it. So this tests a different exit: enter in the entry
zone, and exit ON THE CURVE *before* graduation (where the exit is real), not into the
dead post-grad pool.

Free replay on existing swap data — no Helius. Honest design:
  * Universe: graduated + stalled, caught clean (first vSol < CATCH_VSOL), mature.
  * Entry: first vSol >= ENTRY_VSOL (no lookahead), entry price = curve_price(vSol).
  * Costs: fee_at(entry).entry_cost() + exit_cost_curve() (~7% round trip).
  * Exit rules tested:
      - CEILING (clairvoyant, uses lookahead): sell at the max pre-graduation vSol
        the token reaches after entry. This is the UPPER BOUND — if even this isn't
        clearly positive after costs, the idea is dead.
      - REALISTIC: walk swaps in order, exit at first of {take-profit TP, stop SL,
        graduation (force-exit at last pre-grad vSol), end of data}. No lookahead.
  * A graduating token must be exited at its last vSol BEFORE crossing graduation
    (you cannot sell into the dead post-grad pool) — enforced by capping at GRAD_VSOL.
"""
from __future__ import annotations

import sqlite3
import numpy as np

from config import fee_at, get_cached_constants
from model import Swap, curve_price_per_token

DB = "pumpfun.db"
CATCH_VSOL = 32.0
ENTRY_VSOL = 36.0
MATURITY_SEC = 3600.0


def load():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    max_ts = c.execute("SELECT MAX(ts) FROM swaps").fetchone()[0]
    cutoff = max_ts - MATURITY_SEC
    out = []
    for r in c.execute(
        "SELECT mint,outcome FROM tokens WHERE outcome IN ('graduated','stalled') "
        "AND first_seen_ts <= ?", (cutoff,)):
        swaps = [Swap(ts=s["ts"], wallet=s["wallet"], is_buy=bool(s["is_buy"]),
                      sol_amount=s["sol_amount"], vsol_after=s["vsol_after"],
                      via_frontend=bool(s["via_frontend"]),
                      is_direct=None, entry_program=None)
                 for s in c.execute("SELECT ts,wallet,is_buy,sol_amount,vsol_after,"
                                    "via_frontend FROM swaps WHERE mint=? ORDER BY ts",
                                    (r["mint"],))]
        if swaps and swaps[0].vsol_after < CATCH_VSOL:
            out.append((r["mint"], r["outcome"] == "graduated", swaps))
    c.close(); return out


def replay(tp_mult, sl_mult):
    pc = get_cached_constants()
    grad_v = pc.v_sol_graduation
    ceil_rets, real_rets, grad_flags = [], [], []
    real_exit_reason = {"tp": 0, "sl": 0, "grad": 0, "end": 0}
    for mint, graduated, swaps in load():
        i0 = next((i for i, s in enumerate(swaps) if s.vsol_after >= ENTRY_VSOL), None)
        if i0 is None or i0 < 2:
            continue
        v_entry = swaps[i0].vsol_after
        p_entry = curve_price_per_token(v_entry, pc)
        if p_entry <= 0:
            continue
        fee = fee_at(swaps[i0].ts)
        cost = (1.0 - fee.exit_cost_curve()) / (1.0 + fee.entry_cost())
        fwd = [s for s in swaps[i0 + 1:] if s.vsol_after < grad_v]  # pre-grad only
        grad_flags.append(graduated)

        # CEILING: best pre-grad vSol after entry (clairvoyant).
        v_peak = max([v_entry] + [s.vsol_after for s in fwd])
        ceil_rets.append(curve_price_per_token(v_peak, pc) / p_entry * cost - 1.0)

        # REALISTIC: first trigger walking forward.
        exit_p = None; reason = "end"
        for s in fwd:
            ratio = curve_price_per_token(s.vsol_after, pc) / p_entry
            if ratio >= tp_mult:
                exit_p = p_entry * tp_mult; reason = "tp"; break
            if ratio <= sl_mult:
                exit_p = curve_price_per_token(s.vsol_after, pc); reason = "sl"; break
        if exit_p is None:
            # never hit TP/SL: force-exit at last pre-grad vSol (if it graduated,
            # that's the last liquid price; else the stall price).
            last_v = fwd[-1].vsol_after if fwd else v_entry
            exit_p = curve_price_per_token(last_v, pc)
            reason = "grad" if graduated else "end"
        real_exit_reason[reason] += 1
        real_rets.append(exit_p / p_entry * cost - 1.0)

    ceil = np.array(ceil_rets); real = np.array(real_rets)
    g = np.array(grad_flags)
    n = len(real)

    def ci_low(x, nb=3000):
        rng = np.random.default_rng(0)
        return np.percentile([rng.choice(x, len(x)).mean() for _ in range(nb)], 2.5)

    print(f"=== TP={tp_mult}x  SL={sl_mult}x   n={n} entry-zone tokens "
          f"({g.mean():.1%} graduate) ===")
    print(f"  CEILING (clairvoyant peak exit) mean {ceil.mean():+.1%}  "
          f"median {np.median(ceil):+.1%}  win-rate {(ceil>0).mean():.1%}")
    print(f"  REALISTIC (TP/SL/forced)        mean {real.mean():+.1%}  "
          f"median {np.median(real):+.1%}  win-rate {(real>0).mean():.1%}  "
          f"CI-low {ci_low(real):+.1%}")
    print(f"    exit reasons: {real_exit_reason}")
    print(f"    realistic mean by outcome: graduated {real[g].mean():+.1%}  "
          f"stalled {real[~g].mean():+.1%}")


def main():
    print("Pre-graduation peak-exit replay (curve is liquid; exit is realizable).\n")
    for tp, sl in [(99.0, 0.0), (2.0, 0.5), (1.5, 0.6), (3.0, 0.4)]:
        replay(tp, sl); print()


if __name__ == "__main__":
    main()
