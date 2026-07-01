"""Stress gauntlet for the win-rate winner: PARTIAL SCALE-OUT 50%@1.3x (runner->1.7x /
breakeven), at floor=38. Tests it the way real money would feel it:
  1. CAPACITY     — exact constant-product slippage at 0.5/1/2/5 SOL (both tranches).
  2. FRICTION     — extra adverse haircut (MEV/latency/bad fill) stacked on base fee+MEV.
  3. TEMPORAL OOS — sort by entry time, split into thirds, each third must hold.
  4. JACKKNIFE    — still positive after deleting the single best trade?
  5. STACK        — scale-out AND top-50% entry-velocity together.
Pass bar: CI-low > 0 across sizes and time thirds, jackknife+, friction-survivable.
"""
from __future__ import annotations
import numpy as np
from peak_exit_replay import load
from curve_slippage_replay import ci_low, k_whole
from config import get_cached_constants

pc = get_cached_constants(); K = k_whole(pc); GRAD_V = pc.v_sol_graduation
FLOOR = 38.0
FEE = 0.01
FIRST_MULT, FIRST_FRAC, RUNNER_TP, SL = 1.3, 0.5, 1.7, 0.2


def tr(V_e, V_x, S, mev):
    """exact slippage with fee both sides + adverse haircut `mev` on entry."""
    S_in = S * (1.0 - FEE - mev)
    Q = K / V_e - K / (V_e + S_in)
    if Q <= 0:
        return -1.0
    sol_raw = V_x - K / (K / V_x + Q)
    return (sol_raw * (1.0 - FEE)) / S - 1.0


def scaleout_ret(swaps, i0, S, mev):
    V_e = swaps[i0].vsol_after
    first_r, tp_r, sl_r = V_e * np.sqrt(FIRST_MULT), V_e * np.sqrt(RUNNER_TP), V_e * np.sqrt(SL)
    banked, r1 = False, 0.0
    for s in swaps[i0 + 1:]:
        v = s.vsol_after
        if not banked:
            if v <= sl_r:
                return tr(V_e, sl_r, S, mev)
            if v >= first_r or v >= GRAD_V:
                banked = True
                r1 = tr(V_e, min(v, GRAD_V), S * FIRST_FRAC, mev)
                continue
        else:
            if v >= tp_r or v >= GRAD_V:
                return FIRST_FRAC * r1 + (1 - FIRST_FRAC) * tr(V_e, min(v, GRAD_V), S * (1 - FIRST_FRAC), mev)
            if v <= V_e:                                   # runner breakeven stop
                return FIRST_FRAC * r1 + (1 - FIRST_FRAC) * tr(V_e, V_e, S * (1 - FIRST_FRAC), mev)
    fwd = [s for s in swaps[i0 + 1:] if s.vsol_after < GRAD_V]
    last = fwd[-1].vsol_after if fwd else V_e
    if not banked:
        return tr(V_e, last, S, mev)
    return FIRST_FRAC * r1 + (1 - FIRST_FRAC) * tr(V_e, last, S * (1 - FIRST_FRAC), mev)


def jackknife_pos(x):
    return len(x) > 1 and (x.sum() - x.max()) / (len(x) - 1) > 0


def build():
    ent = []
    for mint, grad, swaps in load():
        i0 = next((i for i, s in enumerate(swaps) if s.vsol_after >= FLOOR), None)
        if i0 is None or i0 < 2:
            continue
        dt = swaps[i0].ts - swaps[0].ts
        vel = (swaps[i0].vsol_after - swaps[0].vsol_after) / dt if dt > 0 else 0.0
        ent.append((swaps, i0, swaps[i0].ts, vel))
    return ent


def row(name, r, extra=""):
    r = np.array(r)
    print(f"{name:<30} n={len(r):>4} win={(r>0).mean():>5.1%} mean={r.mean():>+6.1%} "
          f"med={np.median(r):>+6.1%} CI={ci_low(r):>+6.1%} jack{'+' if jackknife_pos(r) else '-'} {extra}")


def main():
    ent = build()
    print(f"scale-out 50%@1.3x, floor={FLOOR}, {len(ent)} entries\n")

    print("== 1. CAPACITY (size sweep, base friction MEV=1%) ==")
    for S in (0.5, 1.0, 2.0, 5.0):
        row(f"size={S} SOL", [scaleout_ret(sw, i, S, 0.01) for sw, i, _, _ in ent])

    print("\n== 2. FRICTION (size=1 SOL, adverse haircut sweep) ==")
    for mev in (0.01, 0.02, 0.03, 0.05):
        row(f"haircut={mev:.0%}", [scaleout_ret(sw, i, 1.0, mev) for sw, i, _, _ in ent])

    print("\n== 3. TEMPORAL OOS (size=1, base friction, sort by entry ts -> thirds) ==")
    es = sorted(ent, key=lambda e: e[2])
    t = len(es) // 3
    for name, seg in [("early", es[:t]), ("mid", es[t:2 * t]), ("late", es[2 * t:])]:
        row(name, [scaleout_ret(sw, i, 1.0, 0.01) for sw, i, _, _ in seg])

    print("\n== 4. STACK: scale-out + top-50% entry velocity (size=1) ==")
    vel = np.array([e[3] for e in ent])
    thr = np.percentile(vel, 50)
    fast = [e for e in ent if e[3] >= thr]
    row(f"fast-only (>= {thr:.2f} vsol/s)", [scaleout_ret(sw, i, 1.0, 0.01) for sw, i, _, _ in fast])
    # stacked + heavier friction sanity
    row("  fast + 3% haircut", [scaleout_ret(sw, i, 1.0, 0.03) for sw, i, _, _ in fast])


if __name__ == "__main__":
    main()
