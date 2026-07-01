"""Clean A/B: hold the data fixed, vary ONLY the win-side valuation of graduations.

Three plugs for UNPRICED graduations (priced ones always use their real exit):
  flat5x  — legacy optimistic placeholder (empirical_exit_price=None)
  mean    — empirical mean recovered exit price (tail-inclusive; assumes you
            realize the 96x tail at quoted price & size — optimistic bracket)
  median  — empirical median recovered exit price (typical trade / no tail
            capture — conservative bracket)

Same examples for all three, so any difference in ENTER return is attributable to
the win-side assumption alone (not a resample). This isolates whether the prior
"+77%" was real edge or a flat-5x placeholder artifact.
"""
from __future__ import annotations

import statistics as st

from backtest import load_examples_from_store, run_backtest

MIN_FIRST_SEEN_TS = 1782517950.0
MATURITY_SEC = 3600.0


def main() -> None:
    examples, gp, sm = load_examples_from_store(
        "pumpfun.db", min_first_seen_ts=MIN_FIRST_SEEN_TS, maturity_sec=MATURITY_SEC)
    px = [float(r["price_sol_per_unit"]) for r in gp.values()
          if isinstance(r.get("price_sol_per_unit"), (int, float))
          and r["price_sol_per_unit"] > 0]
    mean_px, median_px = sum(px) / len(px), st.median(px)
    print(f"examples={len(examples)} priced_grads={len(px)} "
          f"mean_exit={mean_px:.3e} median_exit={median_px:.3e}\n")

    plugs = [("flat5x", None), ("mean", mean_px), ("median", median_px)]
    print(f"{'plug':8} {'n_ent':>5} {'n_grad':>6} {'hit':>6} {'lift':>7} "
          f"{'mean_ret':>9} {'CI_low':>8} {'jack':>5} {'fold':>5}")
    for name, plug in plugs:
        r = run_backtest(
            examples, model_kind="gbm",
            graduation_prices=gp, stop_exit_multiples=sm,
            empirical_exit_price=plug,
        )
        print(f"{name:8} {r.n_enter:>5} {r.n_grad_in_enter:>6} "
              f"{r.enter_hit_rate:>6.1%} {r.enter_lift_vs_vsol:>+7.1%} "
              f"{r.enter_mean_return:>+9.1%} {r.bootstrap_ci_low:>+8.1%} "
              f"{str(r.jackknife_positive):>5} {str(r.fold_stability):>5}")


if __name__ == "__main__":
    main()
