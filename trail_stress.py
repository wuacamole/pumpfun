"""Stress the trailing-runner exit (improve_entry_filter.py found it ~doubles the edge).

The first-pass trail exited AT the stop level — optimistic on gappy curves. Here the
exit price is PESSIMISTIC: the breaching swap's own vsol (however far it gapped
through the stop), with optional one-swap latency (exit at the swap AFTER the breach)
and adverse haircut sweep, exact constant-product impact. Also: time-thirds OOS,
jackknife, size sweep, and 6-window regime slices (same construction as regime_test).
Baseline comparison uses the same pessimistic convention (be/sl exits at the
breaching swap's vsol too) so the A/B isolates the trail geometry itself."""
from __future__ import annotations

import numpy as np

from curve_slippage_replay import ci_low
from model import ModelStore, predict_grad_prob
from peak_exit_replay import load
from stress_scaleout import FLOOR, GRAD_V, SL, tr

S, MEV, GATE = 1.0, 0.01, 0.070
FIRST_MULT, FIRST_FRAC = 1.3, 0.5


def _exit_px(swaps, j, latency):
    """Pessimistic realized exit reserve: the breaching swap's vsol, or with
    one-swap latency the NEXT swap's vsol (capped pre-grad)."""
    jj = min(j + latency, len(swaps) - 1)
    return min(swaps[jj].vsol_after, GRAD_V)


def trail_ret_pess(swaps, i0, trail, mev, latency=0, tp_fixed=None):
    """Trailing runner, pessimistic fills. tp_fixed=None -> pure trail;
    tp_fixed=x -> baseline fixed TP at x (for A/B under identical conventions)."""
    V_e = swaps[i0].vsol_after
    first_r, sl_r = V_e * np.sqrt(FIRST_MULT), V_e * np.sqrt(SL)
    tp_r = V_e * np.sqrt(tp_fixed) if tp_fixed else None
    banked, r1, peak = False, 0.0, V_e
    seq = swaps[i0 + 1:]
    for j, s in enumerate(seq):
        v = s.vsol_after
        if not banked:
            if v <= sl_r:
                return tr(V_e, _exit_px(seq, j, latency), S, mev)
            if v >= first_r or v >= GRAD_V:
                banked = True
                r1 = tr(V_e, min(v, GRAD_V), S * FIRST_FRAC, mev)
                peak = min(v, GRAD_V)
                continue
        else:
            peak = max(peak, min(v, GRAD_V))
            if tp_r is not None and v >= tp_r:                     # baseline fixed TP
                px = min(v, GRAD_V)
                return FIRST_FRAC * r1 + (1 - FIRST_FRAC) * tr(V_e, px, S * (1 - FIRST_FRAC), mev)
            if v >= GRAD_V:
                return FIRST_FRAC * r1 + (1 - FIRST_FRAC) * tr(V_e, GRAD_V, S * (1 - FIRST_FRAC), mev)
            stop = max(peak * np.sqrt(trail), V_e) if tp_r is None else V_e
            if v <= stop:                                          # trail/breakeven breach
                px = _exit_px(seq, j, latency)
                return FIRST_FRAC * r1 + (1 - FIRST_FRAC) * tr(V_e, px, S * (1 - FIRST_FRAC), mev)
    fwd = [s for s in seq if s.vsol_after < GRAD_V]
    last = fwd[-1].vsol_after if fwd else V_e
    if not banked:
        return tr(V_e, last, S, mev)
    return FIRST_FRAC * r1 + (1 - FIRST_FRAC) * tr(V_e, last, S * (1 - FIRST_FRAC), mev)


def show(name, r):
    r = np.asarray(r, dtype=float)
    jack = "+" if len(r) > 1 and (r.sum() - r.max()) / (len(r) - 1) > 0 else "-"
    print(f"  {name:<36} n={len(r):>4} win={(r > 0).mean():>5.1%} "
          f"mean={r.mean():>+6.1%} med={np.median(r):>+6.1%} CI={ci_low(r):>+6.1%} jack{jack}")


def main():
    m = ModelStore().load("latest")
    toks = load()
    toks.sort(key=lambda t: t[2][0].ts)
    ents = []
    for _mint, _grad, swaps in toks:
        i0 = next((i for i, s in enumerate(swaps) if s.vsol_after >= FLOOR), None)
        if i0 is None or i0 < 2:
            continue
        if predict_grad_prob(m, swaps, as_of=swaps[i0].ts) < GATE:
            continue
        ents.append((swaps[i0].ts, swaps, i0))
    ents.sort(key=lambda e: e[0])
    print(f"model-gated floor-{FLOOR} entries n={len(ents)} (pessimistic fills)\n")

    print("== A/B same conventions (1 SOL, MEV 1%, no latency) ==")
    show("baseline tp1.7/be (pess fills)", [trail_ret_pess(sw, i, 0.85, MEV, 0, tp_fixed=1.7) for _, sw, i in ents])
    for t in (0.85, 0.75):
        show(f"trail {t} (pess fills)", [trail_ret_pess(sw, i, t, MEV, 0) for _, sw, i in ents])

    print("\n== LATENCY (exit at swap AFTER breach) ==")
    show("baseline + 1-swap latency", [trail_ret_pess(sw, i, 0.85, MEV, 1, tp_fixed=1.7) for _, sw, i in ents])
    for t in (0.85, 0.75):
        show(f"trail {t} + 1-swap latency", [trail_ret_pess(sw, i, t, MEV, 1) for _, sw, i in ents])

    print("\n== FRICTION sweep (trail 0.85, 1-swap latency) ==")
    for mev in (0.01, 0.03, 0.05):
        show(f"haircut={mev:.0%}", [trail_ret_pess(sw, i, 0.85, mev, 1) for _, sw, i in ents])

    print("\n== SIZE sweep (trail 0.85, latency 1, MEV 1%) ==")
    global S
    for size in (0.5, 1.0, 2.0, 5.0):
        S = size
        show(f"size={size} SOL", [trail_ret_pess(sw, i, 0.85, MEV, 1) for _, sw, i in ents])
    S = 1.0

    print("\n== TEMPORAL: thirds + 6 regime windows (trail 0.85, latency 1) ==")
    n = len(ents)
    t3 = n // 3
    for name, seg in [("early", ents[:t3]), ("mid", ents[t3:2 * t3]), ("late", ents[2 * t3:])]:
        show(f"third {name}", [trail_ret_pess(sw, i, 0.85, MEV, 1) for _, sw, i in seg])
    edges = np.linspace(0, n, 7).astype(int)
    for w in range(6):
        seg = ents[edges[w]:edges[w + 1]]
        base_w = [trail_ret_pess(sw, i, 0.85, MEV, 1, tp_fixed=1.7) for _, sw, i in seg]
        trl_w = [trail_ret_pess(sw, i, 0.85, MEV, 1) for _, sw, i in seg]
        print(f"  W{w + 1}: baseline CI={ci_low(np.array(base_w)):+6.1%}  "
              f"trail CI={ci_low(np.array(trl_w)):+6.1%}  "
              f"(means {np.mean(base_w):+.1%} -> {np.mean(trl_w):+.1%})")


if __name__ == "__main__":
    main()
