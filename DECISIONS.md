# Decision Log

Append-only. Every gate decision, threshold, material modeling choice, and
backtest run gets an entry here. If it isn't in this log, it didn't happen —
and it can't be cited later as evidence the gate was met. See CHARTER.md §9.2.

Format:
```
[YYYY-MM-DD] [gate|threshold|model|kill|run] author=…
data_hash=<git+db snapshot hash>   backtest_runs_to_date=<n>
metrics={post_cost_mean, ci_low, brier, n_grad, n_enter, …}
decision=<pass|fail|defer|kill>   rationale=<one paragraph>
```

---

[2026-06-26] [gate] author=system
data_hash=initial
backtest_runs_to_date=0
metrics={}
decision=defer
rationale=Phase 1 (data integrity) marked DONE per CHANGES.md. Phase 2 exit
gate not yet evaluated — PR-AUC, log-loss, isotonic calibration, and
reliability diagrams per fold are not yet implemented. This entry seeds the
decision log at the start of the charter-compliance work.

[2026-06-27] [kill] author=system
data_hash=pumpfun.db (pre-fix, ~49% dup rows in recent 15m window)
backtest_runs_to_date=0
metrics={dup_rate_recent_15m=~0.49, dup_rate_full_table=~0.16 (6-col byte-identical)}
decision=kill
rationale=A double-spawned live_monitor logged every on-chain trade twice
with different receipt-time ts (helius.py used time.time() since logsSubscribe
carries no blockTime), so dedupe_swaps' 6-col key — which includes ts —
missed the duplicates and every per-token feature was inflated. Training on
this data would prove a fictional edge. Killed the affected window: any
backtest run on data with ts < the fix boundary is invalid and must not be
cited as evidence. Fix landed in three layers (helius block_time as ts;
storage signature column + partial unique insert-time guard; dedupe_swaps
two-pass + self-healing run on Store open). See CHANGES.md "Duplicate-writer
corruption fix". Legacy rows (signature NULL) are unrecoverable through the
signature pass; exclude pre-fix ts from analysis.
[2026-06-27] [gate] author=system
data_hash=pumpfun.db (post-dedup fix, pre-Tier-1)
backtest_runs_to_date=0
metrics={hand_built_auc=0.366, hand_built_enter_precision=0.056, base_rate=0.286, model_oos_precision=0.585, model_oos_recall=0.724, model_ev=+1.97x, stopped_avg_mult=0.40, timed_out_avg_mult=0.96, flat_loss_default=0.10}
decision=defer
rationale=Three Tier-1 correctness fixes landed (see CHANGES.md "Tier 1
backtest correctness fixes"). (1) The walk-forward no longer ENTERs on the
anti-predictive hand-built heuristic (AUC 0.366 — ENTER picked losers at
5.6% vs 28.6% base); fold 0 now uses a base-rate prior and records no
ENTERs, so train_buf is no longer seeded with wrong labels. (2) The loss
side of _realized_return now uses the real logged stop/time-out exit
multiple (charter I2) instead of a flat 0.10x fiction that overstated
losses by up to 25x for timed-out trades; stop_exit_multiples wired
through Store -> load_examples_from_store -> run_backtest. (3)
Store(auto_dedupe=False) stops analytical opens from running a
full-table DELETE + n_swaps recompute that fought the live writer. Gate
still deferred: the §5.4 capital gate requires a clean-data backtest run
on the post-dedup window with these fixes before the edge can be cited as
evidence. Re-run `cli.py backtest` on data with ts >= the dedup-fix
boundary once enough new labelled tokens accumulate.

[2026-06-27] [model] author=system
data_hash=pumpfun.db (post-Tier-1, pre-Tier-2 refit)
backtest_runs_to_date=0
metrics={positives=423, creators=5834, repeat_creators=1609, is_direct_rows=137824, entry_program_rows=47062, isotonic_in_sample_brier=0.0236}
decision=note
rationale=Four Tier-2 model-quality improvements landed (see CHANGES.md
"Tier 2 — model quality improvements"). (4) Creator-history index wired
from existing tokens.created_meta.traderPublicKey — no new data source
needed; 1609 repeat creators carry signal. No-lookahead enforced via
outcome_ts <= token first-swap ts. (5) direct_call_share + n_entry_programs
added to FEATURE_COLUMNS — the real instruction-stack bot flag that the
paper measured, replacing the inverted size-heuristic via_frontend proxy.
(6) refit_model.py now evaluates with time-ordered expanding-window CV
(train on earlier mints, test on later) — the honest metric the live system
faces; the random-GroupKFold number is retained as an optimistic upper
bound only. (7) Calibration switched from isotonic to Platt (sigmoid) below
1000 positives — isotonic overfits the calibration step function at 423
positives (the suspicious 0.0236 in-sample Brier). Gate still deferred: a
refit + backtest run with these improvements on clean post-dedup data is
required before the edge can be re-cited. The model on disk (latest.pkl) is
still the pre-Tier-2 isotonic fit — re-run `refit_model.py` then
`cli.py backtest` to activate.

[2026-06-27] [run] author=cli
data_hash=—   backtest_runs_to_date=5
metrics={"n_tokens": 3099, "n_enter": 132, "enter_hit_rate": 0.44696969696969696, "enter_mean_return": 1.6098246210547584, "brier": 0.0654662757686211}
decision=note   rationale=backtest run #5 with config {'decision_every': 30.0, 'fold_size': 200, 'kind': 'gbm', 'n_examples': 3099}

[2026-06-27] [run] author=system
data_hash=pumpfun.db + models/latest.pkl (logistic, Platt-calibrated, post-Tier-2)
backtest_runs_to_date=5
metrics={n_tokens=3099, n_enter=132, enter_hit_rate=0.447, enter_mean_return=1.61, baseline=0.155, brier=0.0655, pr_auc=0.846, reliability_slope=0.687, bootstrap_ci_low=1.316, jackknife_positive=True, n_grad_in_enter=59, fold_stability=True}
decision=note
rationale=First backtest with all Tier-1+Tier-2 fixes active. Model refit
chose logistic (time-ordered OOS AUC 0.607 > gbm 0.579; the random-fold AUC
0.927 was optimistic — time-ordered is the honest metric). Calibration is now
Platt (sigmoid) not isotonic — the in-sample Brier rose from 0.0236 (overfit)
to 0.1090 (honest), and the OOS Brier is 0.0655. New features
direct_call_share + n_entry_programs + creator_graduated_count/rugged_count
are in the saved model. Loss-side exits use real stop/time-out multiples
(charter I2). The walk-forward uses calibrate=False (discrimination eval)
and a max_decisions_per_token=100 cap (tokens that live for hours produced
~2000 decision points each, stalling the backtest). All four §5.4 gate
criteria pass: bootstrap CI low >0, jackknife positive, n_grad>=30, fold
stable. Reliability slope 0.687 is below the [0.8,1.2] target — the model
slightly overestimates; more positives (the isotonic threshold is 1000) will
tighten this. Gate still formally deferred pending a clean-data run on the
post-dedup window only; this run includes legacy receipt-time-duplicated
tokens (pre-fix), so it must not be cited as the §5.4 pass evidence.

[2026-06-27] [run] author=system
data_hash=pumpfun.db (post-fix-only: first_seen_ts >= 1782517950, clean data)
backtest_runs_to_date=6
metrics={n_tokens=1246, n_grad=128, base_rate=0.103, n_enter=22, enter_hit_rate=0.364, enter_mean_return=1.309, brier=0.0552, pr_auc=0.2523, reliability_slope=0.460, bootstrap_ci_low=0.627, jackknife_positive=True, n_grad_in_enter=8, fold_stability=True}
decision=note
rationale=First clean-data-only backtest (post-dedup-fix window). Pre-fix
contaminated tokens excluded (first_seen_ts < 1782517950). The edge holds on
clean data: 36.4% ENTER hit rate vs 10.3% base = 3.5x lift, +130.9% mean
return post-cost, bootstrap CI low +62.7%, jackknife positive, fold stable.
But n_grad_in_enter=8 is far below the §5.4 threshold of 30 — too few ENTERs
to be statistically meaningful. The reliability slope 0.460 is well below
[0.8,1.2] — the model overestimates (common with few positives). PR-AUC
0.2523 is low (though still above the 0.103 base rate). Gate remains
deferred: need ~3 more days of clean collection (~1000 post-fix graduates)
for n_grad_in_enter to clear 30 and the calibration to tighten. Monitor
restarted (PID 211232) collecting clean data.
