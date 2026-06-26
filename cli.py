"""
cli.py
======
Entry points for the Phase 2.3 backtest + Phase 2.1 model fit, runnable from
the command line so the workflow is:

  1. python live_monitor.py --paper --dashboard   # collect data for days/weeks
  2. python cli.py backtest                         # honest walk-forward eval
  3. python cli.py fit --kind gbm                   # train + save the model
  4. python live_monitor.py --paper --dashboard --model latest   # use the fit

Run ``python cli.py <command> -h`` for options.
"""

from __future__ import annotations

import argparse
import sys

from backtest import load_examples_from_store, run_backtest, write_report
from conviction_engine import Swap
from model import CreatorHistory, ModelStore, build_training_frame, fit
from storage import Store


def cmd_backtest(args: argparse.Namespace) -> int:
    from config import HOLDOUT_START_TS
    from decisions import append_decision, count_backtest_runs, record_backtest_run

    holdout = HOLDOUT_START_TS if not args.holdout_eval else None
    if args.holdout_eval and HOLDOUT_START_TS is None:
        print("ERROR: --holdout-eval requires HOLDOUT_START_TS to be set in config.py "
              "(charter §9.1 — reserve the holdout before evaluating the gate).")
        return 2
    if args.holdout_eval:
        print("WARNING: --holdout-eval consumes the locked holdout (charter §9.1). "
              "This is a single-shot evaluation; re-running voids the pre-registration.")
    examples, graduation_prices = load_examples_from_store(args.db, holdout_start_ts=holdout)
    if not examples:
        print(f"no labelled tokens in {args.db} — collect data first")
        return 1
    config = {
        "decision_every": args.decision_every,
        "fold_size": args.fold_size,
        "kind": args.kind,
        "n_examples": len(examples),
    }
    run_num = record_backtest_run(config)
    print(f"running walk-forward backtest on {len(examples)} labelled tokens...")
    print(f"  (backtest run #{run_num} — researcher degrees of freedom; "
          f"reported at the §5.4 gate)")
    report = run_backtest(
        examples,
        decision_every_sec=args.decision_every,
        fold_size=args.fold_size,
        model_kind=args.kind,
        graduation_prices=graduation_prices,
    )
    out = write_report(report, args.out)
    print("\n=== backtest report ===")
    print(f"  tokens             : {report.n_tokens}")
    print(f"  decisions          : {report.n_decisions}")
    print(f"  ENTER / WATCH / SKIP: {report.n_enter} / {report.n_watch} / {report.n_skip}")
    print(f"  baseline hit rate   : {report.baseline_hit_rate:.1%}  (P(graduation) overall)")
    print(f"  ENTER hit rate      : {report.enter_hit_rate:.1%}")
    print(f"  ENTER mean return   : {report.enter_mean_return:+.1%}  (post fees/slippage/MEV)")
    print(f"  calibration Brier   : {report.brier_score:.4f}  (lower=better, 0=perfect)")
    print(f"  PR-AUC              : {report.pr_auc:.4f}  (primary for <1% positives)")
    print(f"  log-loss            : {report.log_loss:.4f}")
    print(f"  Brier (ENTER only)  : {report.brier_score_enter:.4f}  (calibration where the gate decides)")
    print(f"  reliability slope   : {report.reliability_slope:.3f}  (charter §5.4: [0.8, 1.2])")
    print(f"  bootstrap CI low    : {report.bootstrap_ci_low:+.1%}  (charter §5.4: must be > 0)")
    print(f"  jackknife positive  : {report.jackknife_positive}  (positive after deleting best trade)")
    print(f"  n_grad in ENTER     : {report.n_grad_in_enter}  (charter §5.4: >= 30)")
    print(f"  fold stability      : {report.fold_stability}  (positive in every final-window fold)")
    if graduation_prices:
        n_logged = sum(1 for m in graduation_prices if m in {e[0] for e in examples})
        print(f"  graduation prices   : {len(graduation_prices)} logged ({n_logged} in this run)")
    else:
        print("  graduation prices   : 0 logged — using flat 5x placeholder payoff")
    print(f"\nreport written to {out}")
    print(f"backtest runs to date : {count_backtest_runs()}")
    # Log the run to DECISIONS.md (charter §9.2 — no silent decisions).
    append_decision(
        kind="run",
        author="cli",
        metrics={
            "n_tokens": report.n_tokens, "n_enter": report.n_enter,
            "enter_hit_rate": report.enter_hit_rate,
            "enter_mean_return": report.enter_mean_return,
            "brier": report.brier_score,
        },
        decision="note",
        rationale=f"backtest run #{run_num} with config {config}",
    )
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    store = Store(args.db)
    labelled = store.labelled_mints()
    if not labelled:
        print(f"no labelled tokens in {args.db} — collect data first")
        return 1
    examples: list[tuple[str, list[Swap], bool, CreatorHistory | None, float | None]] = []
    for mint, outcome in labelled:
        swaps = store.swaps_for(mint)
        if not swaps:
            continue
        graduated = outcome == "graduated"
        # Train on features at the last decision point (no lookahead).
        examples.append((mint, swaps, graduated, None, swaps[-1].ts))
    store.close()
    df = build_training_frame(examples)
    if df.empty or df["graduated"].nunique() < 2:
        print("need both graduated and non-graduated tokens to fit; collect more data")
        return 1
    print(f"fitting {args.kind} on {len(df)} examples "
          f"({int(df['graduated'].sum())} graduated)...")
    model = fit(df, kind=args.kind)
    ms = ModelStore()
    path = ms.save(model, name=args.name)
    print(f"saved model: {path}  (Brier on train = {model.brier_score:.4f})")
    print("note: this is in-sample Brier — run `cli.py backtest` for OOS Brier.")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    store = Store(args.db)
    p = store.export_parquet(args.out_dir)
    print(f"snapshot written: {p}")
    # Optional retention: now that the full history is captured in the immutable
    # snapshot we just wrote, drop swaps older than the rolling window from
    # SQLite so the live db (and every swaps_for scan) stays small. The Parquet
    # snapshot remains the complete analytical record.
    if args.prune_days is not None:
        import time
        cutoff = time.time() - args.prune_days * 86400.0
        deleted = store.prune_swaps_before(cutoff)
        print(f"pruned {deleted} swap rows older than {args.prune_days} days "
              f"(retained in snapshot {p})")
    store.close()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="pumpfun-cli")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="walk-forward backtest on logged data")
    b.add_argument("--db", default="pumpfun.db")
    b.add_argument("--out", default="backtest_report.json")
    b.add_argument("--decision-every", type=float, default=30.0)
    b.add_argument("--fold-size", type=int, default=50)
    b.add_argument("--kind", default="gbm", choices=["gbm", "logistic"])
    b.add_argument(
        "--holdout-eval", action="store_true",
        help="evaluate on the locked holdout (charter §5.4; single-shot; "
             "consumes the holdout — see CHARTER.md §9.1)",
    )
    b.set_defaults(func=cmd_backtest)

    f = sub.add_parser("fit", help="fit graduation model on logged data")
    f.add_argument("--db", default="pumpfun.db")
    f.add_argument("--name", default="latest")
    f.add_argument("--kind", default="gbm", choices=["gbm", "logistic"])
    f.set_defaults(func=cmd_fit)

    e = sub.add_parser("export", help="export SQLite -> Parquet snapshot")
    e.add_argument("--db", default="pumpfun.db")
    e.add_argument("--out-dir", default="snapshots")
    e.add_argument(
        "--prune-days", type=float, default=None,
        help="after exporting, delete swaps older than this many days from "
             "SQLite (rolling window; the snapshot keeps the full history)",
    )
    e.set_defaults(func=cmd_export)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()