"""Unit tests for the curve-momentum paper strategy (curve_momentum.py).

Covers: entry gating (caught-clean + score + single-decision), the three exits
(TP / SL / graduation force-exit), no re-entry, and the size-dependent slippage of
the exact constant-product return math. Run: uv run python test_curve_momentum.py
(also pytest-discoverable).
"""
from __future__ import annotations

from curve_momentum import CATCH_VSOL, ENTRY_VSOL, CurveMomentumPaper
from model import Swap


def mk(vsol, ts):
    return Swap(ts=ts, wallet="w", is_buy=True, sol_amount=0.1, vsol_after=vsol,
               via_frontend=True, is_direct=None, entry_program=None)


def feed(b, mint, vsols, score=0.5):
    last = None
    for i, v in enumerate(vsols):
        last = b.on_swap(mint, mk(v, float(i)), score_fn=lambda: score)
    return last


def test_take_profit_exit():
    b = CurveMomentumPaper(size_sol=1.0)
    # caught clean (20<32), enter at 40, bank first tranche at 50 (>=40*sqrt1.3=45.6),
    # runner take-profit at 70 (>=40*sqrt1.7=52.2). Blended return is positive.
    feed(b, "m", [20, 40, 50, 70])
    assert len(b.closed) == 1
    t = b.closed[0]
    assert t.reason == "tp" and t.entry_vsol == 40 and t.ret > 0


def test_scaleout_breakeven_runner():
    b = CurveMomentumPaper(size_sol=1.0)
    # enter 40, bank half at 50, runner falls back to entry (40) -> breakeven stop.
    # Net positive (banked tranche profit, runner ~flat).
    feed(b, "m", [20, 40, 50, 40])
    assert len(b.closed) == 1
    t = b.closed[0]
    assert t.reason == "be" and t.ret > 0


def test_stop_loss_exit():
    b = CurveMomentumPaper(size_sol=1.0, sl_mult=0.3)   # explicit SL for the geometry
    # enter at 40, drop below 40*sqrt(0.3)=21.9 -> SL, big loss
    feed(b, "m", [20, 40, 20])
    assert len(b.closed) == 1
    t = b.closed[0]
    assert t.reason == "sl" and t.ret < -0.5


def test_graduation_force_exit():
    b = CurveMomentumPaper(size_sol=1.0, tp_mult=5.0)   # high TP so grad triggers first
    # enter at 60 (TP would be 60*sqrt5=134 > grad 115, never hits); graduate -> exit
    # at the last pre-grad vSol (114), reason 'grad', positive (114/60)^2 move.
    feed(b, "m", [20, 60, 114, 116])
    assert len(b.closed) == 1
    t = b.closed[0]
    assert t.reason == "grad" and t.exit_vsol == 114 and t.ret > 0


def test_smart_sells_logged_not_exit():
    # Smart-exit was DEMOTED to log-only (regime_test.py: mild net drag). 2+ smart wallets
    # selling must NOT force an exit; the position stays open and smart_sells is counted.
    b = CurveMomentumPaper(size_sol=1.0, smart_wallets={"S1", "S2"})
    b.on_swap("m", mk(20, 0.0), score_fn=lambda: 0.5)        # caught clean
    b.on_swap("m", mk(40, 1.0), score_fn=lambda: 0.5)        # enter at 40
    assert len(b.positions) == 1
    b.on_swap("m", Swap(ts=2.0, wallet="S1", is_buy=False, sol_amount=0.1, vsol_after=45,
                        via_frontend=True, is_direct=None, entry_program=None))
    # second distinct smart seller -> still NO exit (log-only), position remains open
    t = b.on_swap("m", Swap(ts=3.0, wallet="S2", is_buy=False, sol_amount=0.1, vsol_after=44,
                            via_frontend=True, is_direct=None, entry_program=None))
    assert t is None and len(b.positions) == 1 and len(b.closed) == 0
    # the counter is still tracked, and surfaces on the eventual close
    t = b.force_close("m", ts=9.0)
    assert t.smart_sells == 2


def test_smart_buys_logged_at_entry():
    b = CurveMomentumPaper(size_sol=1.0, smart_wallets={"S1", "S2"})
    # two smart wallets buy early, then we enter -> smart_buys recorded
    b.on_swap("m", Swap(ts=0.0, wallet="S1", is_buy=True, sol_amount=0.1, vsol_after=20,
                        via_frontend=True, is_direct=None, entry_program=None), score_fn=lambda: 0.5)
    b.on_swap("m", Swap(ts=1.0, wallet="S2", is_buy=True, sol_amount=0.1, vsol_after=30,
                        via_frontend=True, is_direct=None, entry_program=None), score_fn=lambda: 0.5)
    b.on_swap("m", mk(40, 2.0), score_fn=lambda: 0.5)        # enter
    t = b.force_close("m", ts=9.0)
    assert t.smart_buys == 2


def test_caught_clean_filter():
    b = CurveMomentumPaper(size_sol=1.0)
    # first vSol 40 >= CATCH (32): never a valid early-history entry
    feed(b, "m", [40, 95])
    assert len(b.closed) == 0 and len(b.positions) == 0


def test_score_filter_and_single_decision():
    b = CurveMomentumPaper(size_sol=1.0)
    # caught clean but score below threshold at the entry trigger -> skip, and do
    # NOT re-enter later at higher vSol.
    feed(b, "m", [20, 40, 95], score=0.01)
    assert len(b.closed) == 0 and len(b.positions) == 0
    assert "m" in b.traded   # decision was made (and was 'skip')


def test_no_reentry_after_close():
    b = CurveMomentumPaper(size_sol=1.0)
    feed(b, "m", [20, 40, 95, 40, 95])   # TP closes; later swaps must not re-open
    assert len(b.closed) == 1 and len(b.positions) == 0


def test_slippage_scales_with_size():
    # Same TP move; a larger position realizes LESS (more exit impact).
    small = CurveMomentumPaper(size_sol=0.1); feed(small, "m", [20, 40, 50, 70])
    big = CurveMomentumPaper(size_sol=5.0); feed(big, "m", [20, 40, 50, 70])
    assert small.closed[0].ret > big.closed[0].ret


def test_threshold_boundary_enters():
    b = CurveMomentumPaper(size_sol=1.0, score_min=0.065)
    feed(b, "m", [20, 40, 50, 70], score=0.065)   # exactly at threshold -> enters
    assert len(b.closed) == 1 and b.closed[0].reason == "tp"


def test_force_close_on_eviction():
    b = CurveMomentumPaper(size_sol=1.0)
    feed(b, "m", [20, 40, 50])          # enter at 40, no TP/SL/grad -> stays open
    assert len(b.positions) == 1 and len(b.closed) == 0
    t = b.force_close("m", ts=999.0)    # monitor evicts -> close at last price (50)
    assert t is not None and t.reason == "evict" and t.exit_vsol == 50
    assert len(b.positions) == 0 and len(b.closed) == 1
    assert b.force_close("m", ts=1000.0) is None   # idempotent: no double close


def test_time_stop():
    b = CurveMomentumPaper(size_sol=1.0)
    # enter at 40 (ts=1); a later swap > MAX_HOLD_SEC away with no TP/SL -> time exit
    b.on_swap("m", mk(20, 0.0), score_fn=lambda: 0.5)
    b.on_swap("m", mk(40, 1.0), score_fn=lambda: 0.5)
    t = b.on_swap("m", mk(50, 1.0 + 4000.0), score_fn=lambda: 0.5)
    assert t is not None and t.reason == "time"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
