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

[2026-06-27] [run] author=system
data_hash=pumpfun.db (post-fix-only: first_seen_ts >= 1782517950, clean data; +58 labelled tokens since run #6)
backtest_runs_to_date=6 (ad-hoc script — not counted; superseded by the cli run below)
metrics={n_tokens=1304, base_rate=0.102, n_enter=72, enter_hit_rate=0.278, enter_mean_return=0.938, brier=0.0559, pr_auc=0.3458, reliability_slope=0.917, bootstrap_ci_low=0.588, jackknife_positive=True, n_grad_in_enter=20, fold_stability=True}
decision=note
rationale=Re-run of the clean-window backtest (run #6 method: same first_seen_ts
floor, logistic, decision_every=30s, fold_size=50) on current data. Run via an
ad-hoc script (NOT counted in backtest_runs_to_date) because, at the time,
load_examples_from_store had no post-fix floor arg — only holdout_start_ts. That
reproducibility gap is now fixed: a --min-first-seen-ts flag was added to
cli.py backtest + load_examples_from_store, so this clean-window run is now a
first-class, auto-logged cli command (see the author=cli entry immediately
below, which reproduces this run and IS counted). Headline change is calibration:
reliability slope moved 0.460 -> 0.917, now INSIDE the §5.4 [0.8,1.2] band (the
few-positives overestimation resolved as more labelled tokens accumulated).
PR-AUC improved 0.2523 -> 0.3458. The model now ENTERs more (22 -> 72), so
n_grad_in_enter rose 8 -> 20 (still < 30 gate); ENTER hit rate eased
36.4% -> 27.8% and mean return +130.9% -> +93.8% as more marginal ENTERs
entered, but still a 2.7x lift over the 10.2% base with bootstrap CI low
+58.8%, jackknife positive, fold stable. Four of five §5.4 criteria now pass;
only n_grad_in_enter remains short. Gate still deferred — keep collecting;
~10 more clean graduates landing in the ENTER zone should clear it.
Monitor still PID 211232.

[2026-06-27] [run] author=cli
data_hash=—   backtest_runs_to_date=6
metrics={"n_tokens": 1346, "n_enter": 78, "enter_hit_rate": 0.32051282051282054, "enter_mean_return": 1.1117090932749927, "brier": 0.054786871368707524}
decision=note   rationale=backtest run #6 with config {'decision_every': 30.0, 'fold_size': 50, 'kind': 'logistic', 'n_examples': 1346, 'min_first_seen_ts': 1782517950.0}
  Full §5.4 metrics (auto-logger stores only 5): base_rate=0.109,
  enter_hit_rate=0.321 (2.9x lift), enter_mean_return=+111.2%, pr_auc=0.3495,
  reliability_slope=0.857 (IN [0.8,1.2] band), bootstrap_ci_low=+76.2% (>0),
  jackknife_positive=True, n_grad_in_enter=25 (gate >=30 — still short),
  fold_stability=True. First clean-window run via the new --min-first-seen-ts
  flag (reproducible + counted). 4 of 5 gate criteria pass; n_grad_in_enter=25
  is the lone blocker and is climbing (20 -> 25). Gate deferred pending ~5 more
  clean graduates in the ENTER zone.

[2026-06-27] [run] author=cli
data_hash=—   backtest_runs_to_date=7
metrics={"n_tokens": 1446, "n_enter": 95, "enter_hit_rate": 0.29473684210526313, "enter_mean_return": 1.0078818962671916, "brier": 0.05746002477293715}
decision=note   rationale=backtest run #7 with config {'decision_every': 30.0, 'fold_size': 50, 'kind': 'logistic', 'n_examples': 1446, 'min_first_seen_ts': 1782517950.0}

[2026-06-27] [run] author=cli
data_hash=—   backtest_runs_to_date=8
metrics={"n_tokens": 1477, "n_enter": 91, "enter_hit_rate": 0.32967032967032966, "enter_mean_return": 1.1519757924852796, "brier": 0.05850784877117671}
decision=note   rationale=backtest run #8 with config {'decision_every': 30.0, 'fold_size': 50, 'kind': 'logistic', 'n_examples': 1477, 'min_first_seen_ts': 1782517950.0}

[2026-06-27] [model] author=system
data_hash=pumpfun.db (clean window) + models/latest.pkl (refit, gbm)
backtest_runs_to_date=8
metrics={ths_univariate_auc=0.667, ths_grad_median=0.443, ths_nongrad_median=0.725, ab_gbm_delta_auc=-0.005, ab_gbm_entryzone_delta=-0.009, ab_logistic_delta_auc=+0.006, ab_logistic_entryzone_delta=+0.005, run8_slope=0.759, run8_n_grad_in_enter=30, run8_ci_low=0.836}
decision=note
rationale=Dead-feature cleanup. (a) smart_money_present DROPPED outright — it
required a curated profitable-wallet set that does not exist; labelling "smart"
from realized PnL would be target leakage. It was 100% NaN (imputed to median),
contributing only noise + false completeness. (b) top_holder_share was 100% NaN
(no holder snapshot) but is reconstructable as-of from the swap stream with zero
new RPC: per swap, infer token amount = sol_amount / curve_price_per_token(
vsol_after), sum per wallet (buys add, sells subtract, negatives clamped to 0),
share = max holding / total positive holdings. Wired into model.featurize (no
lookahead — only swaps with ts<=as_of). It HAS univariate signal: graduates are
far less concentrated (median 0.443) than non-graduates (0.725), univariate
AUC 0.667 — economically sensible (one whale = fragile/rug-prone). BUT a
controlled A/B on identical data (same snapshot frame, toggle the column) shows
it is REDUNDANT: marginal time-ordered OOS AUC delta is +/-0.005-0.009 (noise),
slightly NEGATIVE for gbm. The signal is already carried by n_unique_wallets /
sol_per_human_wallet / bot_trade_share (one whale => few wallets => low
SOL/human). By the "a feature must prove OOS lift to earn its place" rule it does
NOT make the active FEATURE_COLUMNS — but featurize still computes it (cheap,
tested, documented) so re-adding is one line if a future interacting feature
appears. Model refit on the cleaned 15-feature set; refit auto-selected gbm
(time-ordered OOS AUC 0.641 > logistic 0.623). latest.pkl updated (takes effect
on next monitor restart; running PID 211232 keeps its in-memory model until
then). Clean-window backtest run #8 on the cleaned set: n_grad_in_enter just
reached 30 (gate >=30) as data accumulated, but reliability slope 0.759 is now
the lone sub-band criterion (was n_grad). Criteria are jittering near the
boundary at this sample size; gate remains DEFERRED pending a stable
clean-window run with slope in [0.8,1.2] AND n_grad_in_enter comfortably >30.

[2026-06-27] [run] author=cli
data_hash=—   backtest_runs_to_date=9
metrics={"n_tokens": 1580, "n_enter": 104, "enter_hit_rate": 0.33653846153846156, "enter_mean_return": 1.1766891958594519, "brier": 0.05798941981320831}
decision=note   rationale=backtest run #9 with config {'decision_every': 30.0, 'fold_size': 50, 'kind': 'logistic', 'n_examples': 1580, 'min_first_seen_ts': 1782517950.0}

[2026-06-27] [kill] author=system
data_hash=pumpfun.db (clean window, first_seen_ts>=1782517950) + time-ordered OOS
backtest_runs_to_date=9
metrics={vsol_alone_auc=0.860, model_overall_auc_gbm=0.649, model_overall_auc_logit=0.593, entryzone_auc_gbm=0.377, entryzone_auc_gbm_ci=[0.306,0.452], entryzone_auc_logit=0.454, entryzone_calib_gbm={p0-15:8.1%, p15-30:11.0%, p30-50:0.0%, p50-100:0.8%}, enter_hit=0.337, vsol_matched_baseline=0.267, enter_lift_vs_vsol=0.070}
decision=kill
rationale=The model's apparent edge is a vSol-level ARTIFACT, not skill, and must
NOT be cited as §5.4 gate evidence. Evidence: (1) vSol-level ALONE scores
time-ordered OOS AUC 0.860; the full model scores only 0.649 (gbm) / 0.593
(logistic) — the behavioural features DEGRADE the trivial "high vSol => near
graduation" signal. (2) Within every vSol band the model is anti-predictive:
entry-zone[33-50] AUC 0.377 (gbm), bootstrap 95% CI [0.306,0.452] ENTIRELY below
0.5 — statistically significant inversion (the conviction-engine-inverted finding
reincarnated in the GBM). (3) Entry-zone calibration is inverted AND overconfident:
tokens the model rates >50% graduate 0.8% (n=128); tokens it rates 0-15% graduate
8.1%. (4) The backtest "+edge" came from ENTERing at median implied vSol ~72
(already 63% to graduation); against a vSol-matched baseline the lift is only
+7.0pp (33.7% vs 26.7%) at n=104, much of which is noise — NOT the +21pp the old
overall-baseline comparison implied. Two code bugs enabled the fiction and are
fixed: refit selected by OVERALL AUC (deployed gbm, the worst entry-zone model)
-> now selects by entry-zone AUC and REFUSES to deploy when it is <= 0.5 (it
refused on this data); the backtest baseline was not vSol-matched -> now reports
vsol_matched_baseline + enter_lift_vs_vsol. The live ENTER gate is neutered
(WATCH-only) until a model with entry_zone_auc > 0.5 exists. Net: there is NO
demonstrated alpha. The paper's premise (early behavioural conviction predicts
graduation) does not hold in this data. Do NOT deploy capital. Next: find a
feature with real at-fixed-vSol signal, or accept the strategy is unviable.

[2026-06-27] [model] author=route-feature-test
data_hash=git=ea8c5a1 db=pumpfun.db(window first_seen>=1782517950, mature<=maxts-7200)   backtest_runs_to_date=10
metrics={"n_tokens": 601, "n_grad": 74, "entryzone_snaps": 474, "entryzone_grad": 44, "share_mayhem_univariate_auc": 0.829, "vsol_time_baseline_ez_auc_logit": 0.523, "vsol_time_baseline_ez_auc_gbm": 0.677, "route_vsol_time_ez_auc_logit": 0.488, "route_vsol_time_ez_auc_gbm": 0.721, "route_lift_logit": -0.034, "route_lift_gbm": 0.044, "within_hour_mayhem_grad_gap": "vanishes/flips (hr17 12.9 vs 14.8; hr19 21.5 vs 29.9)"}
decision=note   rationale=Tested route_category(entry_program) shares as graduation features — the real per-program bot/human taxonomy (pump_direct=website/naive, mayhem/flashx/maestro=automated wrappers, jupiter=aggregator), the correct version of the discredited is_direct invoke-depth flag (on-chain audit: flashx & mayhem are third-party trading wrappers, NOT the frontend; is_direct=1 conflates website humans with naive bots). REJECTED. The eye-catching signal (share_mayhem entry-zone univariate AUC 0.829, grad 0.685 vs non-grad 0.260) is a RIGHT-CENSORING x time-cohort artifact: mayhem adoption rose monotonically across the capture window AND graduation rate by launch hour climbs to a fake 100% in the final hours (recent tokens that will stall are not yet labelled 'stalled'), so route share proxies 'launched late' => censored => fake-graduate-heavy. Held at fixed launch hour the mayhem advantage disappears and flips. Against a baseline that already knows launch time (vsol+launch_frac), route adds lift in GBM only (+0.044) and HURTS logistic (-0.034) — a GBM-only, not-robust result = overfit/residual-confound, not skill, at 44 entry-zone positives. Reinforces DECISIONS [kill] 2026-06-27: the paper's bot-share premise does not hold here; no at-fixed-vSol edge from route identity. K-criterion evidence. SIDE FINDING (data integrity, Phase 1 gap): resolved-outcome labels are right-censored on the recent tail; refit_model.py and backtest.py lack an outcome-maturity guard and are partly exposed to the same gradient. Adding a maturity guard (exclude first_seen > maxts-7200) to both. New harness route_feature_test.py is self-defending (maturity guard + launch-time control + both-model-class robustness).

[2026-06-27] [model] author=backfill-stalled
data_hash=git=ea8c5a1 db=pumpfun.db (post stalled-backfill; audit backfill_stalled_1782548503.jsonl)   backtest_runs_to_date=10
metrics={"stalled_before": 3148, "stalled_after": 15234, "labelled": 12086, "resolved_grad_rate_before": 0.2, "resolved_grad_rate_after": 0.049, "unknown_after": 18762, "unknown_zero_swap": 18429, "unknown_unsettled": 336}
decision=note   rationale=Closed the stalled-labelling SELECTION BIAS (Charter I3). Root cause: 'stalled' was only ever written by the paper broker when an *entered* position timed out (live_monitor.py:606), so the resolved negative class was the paper-entered subset (~20% of real stalls, non-randomly selected) — inflating apparent graduation rate and biasing every fit. Fix (2 parts): (1) live_monitor._gc now labels EVERY inactivity-evicted, non-graduated, currently-unknown token 'stalled' (outcome_ts = last_swap + STALE_AFTER_SEC, only_if_unknown so a resurrection->graduation isn't clobbered); storage.set_outcome gained only_if_unknown; Store now sets busy_timeout=30000 so a concurrent maintenance writer doesn't error the monitor. (2) backfill_stalled.py relabelled 12086 historical settled/traded/never-graduated unknowns -> stalled. Result: stalled 3148->15234, resolved grad rate 20.0%->4.9% (de-biased toward the true ~caught-with-swaps base rate). Remaining 18762 unknown = 18429 zero-swap create-only (left unknown: never traded, not featurizable) + 336 not-yet-settled tail. Reversible: backfilled stalls have no paper_trades row; audit JSONL written. NOTE: live _gc fix needs a monitor RESTART to take effect going forward (running PID 238864 has the old code). This materially changes the training distribution — all prior model/AUC numbers were fit on the biased label set and should be re-derived on the de-biased data before any are cited.

[2026-06-27] [run] author=cli
data_hash=—   backtest_runs_to_date=12
metrics={"n_tokens": 6136, "n_enter": 76, "enter_hit_rate": 0.40789473684210525, "enter_mean_return": 1.2384158977517423, "brier": 0.030420151784099545}
decision=note   rationale=backtest run #12 with config {'decision_every': 30.0, 'fold_size': 50, 'kind': 'gbm', 'n_examples': 6136, 'min_first_seen_ts': 1782517950.0, 'maturity_sec': 3600.0}

[2026-06-27] [model] author=debias-reeval
data_hash=git=ea8c5a1 db=pumpfun.db(de-biased labels) backtest run#12 clean-window+maturity   backtest_runs_to_date=12
metrics={"model_entryzone_oos_auc_gbm": 0.754, "model_entryzone_oos_auc_logit": 0.69, "route_lift_gbm": 0.102, "route_lift_logit": 0.157, "mayhem_grad_within_hour": "persists 2-3x every hr (de-biased) vs vanished (biased)", "bt_n_enter": 76, "bt_n_grad_in_enter": 31, "bt_enter_hit": 0.408, "bt_vsol_matched_base": 0.335, "bt_enter_lift_vs_vsol": 0.073, "bt_enter_mean_return_PLACEHOLDER5x": 1.238, "bt_ci_low_PLACEHOLDER5x": 0.827, "bt_reliability_slope": 0.82, "bt_grad_prices_real_in_run": 0}
decision=note   rationale=REOPENS DECISIONS [kill] 2026-06-27. The kill concluded 'the paper's bot-share/conviction premise does not hold; no edge; model is an inverted vSol detector.' That conclusion was substantially an ARTIFACT OF THE STALLED-LABELLING SELECTION BIAS (now fixed): the negative class was the paper-entered failures, which made the model look anti-predictive in the entry zone. On DE-BIASED labels every prior negative result flips: model entry-zone time-ordered OOS AUC 0.690-0.754 (both classes, was <=0.5); route features lift over a launch-time-aware baseline in BOTH classes (+0.10/+0.16); and the mayhem grad-rate advantage PERSISTS 2-3x within every launch hour and among settled-only mature tokens (was an artifact before). 'mayhem' is a bundler/coordinated-burst launch tool (avg life 91s, 49 swaps): the signal is real and not a censoring artifact. HOWEVER this is NOT a §5.4 pass and must not be cited as one: (1) the backtest's +123.8% return / +82.7% CI-low use the FLAT 5x graduation PLACEHOLDER — 0 real PumpSwap exit prices in the entered set — so the economic return is an assumption, not a measurement; (2) underpowered: n_enter=76 (<<300 required), n_grad_in_enter=31 (~30 floor); (3) entries sit at vSol where the matched base is already 33.5% (late/compressed payoff) and mayhem's ~10% entry-zone grad rate is at/below the (vSol/115)^2 breakeven by vSol~40 — a real STATISTICAL signal may still not be an ECONOMIC edge (Charter §4); (4) run #12, non-holdout, rising researcher DoF. NET: the kill's *labeling-artifact premise* is overturned (important), but no tradeable post-cost edge is demonstrated. DO NOT deploy capital; DO NOT touch the locked holdout. Next: collect real graduation prices + reach N_enter>=300/N_grad>=30 via paper, re-value the win side with real prices, then re-test. New latest.pkl (entry_zone_auc 0.754) passes the deploy guard; running monitor still WATCH-only until restart.

[2026-06-28] [run] author=cli
data_hash=—   backtest_runs_to_date=13
metrics={"n_tokens": 13675, "n_enter": 159, "enter_hit_rate": 0.4276729559748428, "enter_mean_return": 0.7716575593795655, "brier": 0.03167260608871668}
decision=note   rationale=backtest run #13 with config {'decision_every': 30.0, 'fold_size': 50, 'kind': 'gbm', 'n_examples': 13675, 'min_first_seen_ts': 1782517950.0, 'maturity_sec': 3600.0}

[2026-06-28] [model] author=grad-price-recovery
data_hash=db=pumpfun.db; recover_grad_prices.py; audit none (idempotent upsert)   backtest_runs_to_date=13
metrics={"grad_prices_before": 66, "grad_prices_after": 287, "recovery_rate_population": 0.19, "recovery_rate_oldest_cohort": 0.95, "live_capture_success_all_time": "1 of 1211", "recovered_payoff_multiple_median": 0.65, "recovered_payoff_q1": 0.27, "recovered_payoff_mean": 3.66, "recovered_payoff_max": 96.19, "frac_below_1x": 0.57, "graduated_total": 1241, "graduations_still_placeholder_5x": 954, "bt_run13_n_enter": 159, "bt_run13_n_grad_in_enter": 68, "bt_run13_enter_hit": 0.428, "bt_run13_vsol_matched_base": 0.37, "bt_run13_lift_vs_vsol": 0.058, "bt_run13_enter_mean_return_MIXED": 0.772, "bt_run13_ci_low": 0.498, "bt_run13_reliability_slope": 0.922, "bt_run13_real_prices_in_run": 100, "gate_subcriteria_passing": "5 of 6 (CIlow>0, n_grad>=30, jackknife, fold_stable, slope in[.8,1.2]); N_enter 159<300 FAILS"}
decision=note   rationale=CRITICAL DATA-INTEGRITY + WIN-SIDE finding. (1) LIVE graduation-price capture is broken: _record_graduation_price fires AT the migrate event when the PumpSwap pool has no trades yet, so _fetch_pumpswap_price returns None and the except:pass swallows it -> only 1 of 1211 graduations ever logged a price live. Built recover_grad_prices.py (same extractor, run LATER when trades exist; idempotent). Sweep raised real prices 66->287. (2) Recovery caps at 19% population because _pool_price_from_tx returns None even on confirmed PumpSwap txs: it needs ONE owner holding both base-mint and WRAPPED-SOL in postTokenBalances, but PumpSwap base/quote vaults have DIFFERENT owners and some txs show only the WSOL leg -> robust coverage needs identifying the pool vault accounts (derivable from the create_pool tx helius.py already parses). (3) THE BIG ONE: the recovered exit-price distribution OVERTURNS the flat-5x placeholder. Real exits are HEAVY-TAILED LOSSES: median payoff 0.65x, Q1 0.27x, 57% below 1x (a loss), mean 3.66x carried by a thin right tail (max 96x). The flat 5x default is not merely optimistic, it is inverted from the typical token. backtest run #13 still values 954/1241 graduations at 5x (only 100 real in-run), so its +77.2% mean return / +49.8% CI-low are STILL upward-biased; the trend across runs (124%@0-real -> 77%@100-real) points DOWN as coverage rises. (4) On the §5.4 gate, run #13 passes 5 of 6 sub-criteria (CI-low>0, n_grad=68>=30, jackknife+, fold-stable, slope 0.922) and fails only N_enter (159<300) — but this is on a placeholder-inflated win side, so the partial pass MUST NOT be read as near-deployable. Edge thesis now reduces to one testable question: does the gate preferentially ENTER the right-tail winners rather than the median 0.65x losers? Unanswered. NEXT (gating work, in priority): (a) fix _pool_price_from_tx to read native-SOL/different-owner pool reserves (identify vaults via create_pool) so coverage -> high; (b) until then, replace the flat-5x backtest fallback with the EMPIRICAL recovered-exit distribution (Charter: move win-side assumption toward conservative/measured) and re-run; (c) fix the live hook to defer/retry the fetch post-migrate, or schedule recover_grad_prices.py. DO NOT deploy capital; DO NOT touch the locked holdout. Reinforces [[conviction-engine-inverted]].

[2026-06-28] [run] author=cli
data_hash=—   backtest_runs_to_date=14
metrics={"n_tokens": 14470, "n_enter": 164, "enter_hit_rate": 0.4634146341463415, "enter_mean_return": -0.41200870341052265, "brier": 0.03217257510767798}
decision=note   rationale=backtest run #14 with config {'decision_every': 30.0, 'fold_size': 50, 'kind': 'gbm', 'n_examples': 14470, 'min_first_seen_ts': 1782517950.0, 'maturity_sec': 3600.0, 'win_fallback': 'empirical_mean', 'empirical_exit_price': 3.0086943238394635e-07, 'n_priced_graduations': 287}

[2026-06-28] [kill] author=winside-bracket
data_hash=db=pumpfun.db window first_seen>=1782517950 maturity<=maxts-3600; scratch_winside_bracket.py; runs #14 + bracket   backtest_runs_to_date=14
metrics={"examples": 14833, "n_enter": 168, "n_grad_in_enter": 82, "enter_hit_rate": 0.488, "enter_lift_vs_vsol": 0.106, "priced_grads": 287, "unpriced_grads_valued_by_plug": "rest", "mean_exit_price": 3.009e-07, "median_exit_price": 5.341e-08, "AB_identical_data_only_winside_varies": true, "flat5x_mean_return": 0.99, "flat5x_ci_low": 0.706, "flat5x_jackknife": true, "empirical_mean_return": -0.41, "empirical_mean_ci_low": -0.48, "empirical_mean_jackknife": false, "empirical_median_return": -0.766, "empirical_median_ci_low": -0.808, "empirical_median_jackknife": false, "recovered_exit_median_multiple": 0.65, "frac_grads_exit_below_1x": 0.57}
decision=kill   rationale=DEFINITIVE NO-ECONOMIC-EDGE, proven by a clean A/B that holds data+model+entries fixed and varies ONLY the win-side valuation of graduations. Identical across all three plugs: n_enter=168, n_grad_in_enter=82, enter_hit=48.8%, lift_vs_vsol=+10.6% — so the model and its decisions are unchanged. Yet ENTER mean return swings flat5x=+99.0% (CI+70.6%, jack True) -> empirical-MEAN=-41.0% (CI-48.0%, jack False) -> empirical-MEDIAN=-76.6% (CI-80.8%, jack False). The ENTIRE apparent profitability (and the earlier 5-of-6 gate pass) was an artifact of the flat 5x graduation PLACEHOLDER over-valuing the ~954 unpriced graduations. With wins valued at the MEASURED PumpSwap exit distribution (287 recovered real prices: median payoff 0.65x, 57% exit BELOW entry, heavy right tail max 96x), the strategy is deeply negative under BOTH brackets — even the optimistic mean-plug (which assumes you realize the 96x tail at quoted price and at size, implausible in thin pools) is -41%. KEY REFRAME vs the original 2026-06-27 [kill]: that kill said the model had no/inverted statistical signal — that was a stalled-LABELLING artifact and was WRONG. The TRUE reason there is no edge is different and was only now measurable: the model DOES have real graduation-discrimination skill (entry-zone AUC 0.75; +10.6pp graduation lift over the vSol-matched base, robust across plugs), but GRADUATION IS THE WRONG TARGET — most graduations exit at a LOSS, so selecting graduators more accurately does not make money. The paper hypothesis (early behavioural conviction -> graduation -> profit) breaks at the graduation->profit link, not the conviction->graduation link. To have an edge one would need to predict the post-graduation right TAIL (which graduates THEN 10x+), or exit pre-graduation at peak vSol, or accept the strategy is unviable. As specified, it is unviable. Charter outcome: a successful NO-TRADE finding. DO NOT deploy capital; DO NOT touch the locked holdout. Guards already in place keep live ENTER paper-only. See [[conviction-engine-inverted]], [[pumpfun-data-quality-history]].

[2026-06-28] [model] author=guard-hardening
data_hash=live_monitor.py:enter_validated, model.py:FittedModel; PID 298681 verified WATCH-only   backtest_runs_to_date=14
metrics={"gate_condition_old": "entry_zone_auc>0.5", "gate_condition_new": "entry_zone_auc>0.5 AND economic_gate_passed", "economic_gate_passed_current": false, "live_decisions_now": "WATCH-only"}
decision=note   rationale=Hardened the live ENTER gate after [kill] winside-bracket. Was: enter_validated = (entry_zone_auc>0.5) — a purely STATISTICAL criterion the bracket proved insufficient (a model can rank graduations perfectly, AUC 0.75/+10.6pp lift, and still lose -41% to -77% because graduations exit at a loss). Now: enter_validated also requires FittedModel.economic_gate_passed (new field, default False) = True only when a backtest valued at the MEASURED PumpSwap exit distribution shows positive post-cost ENTER return with bootstrap CI-low>0. Current model: economic_gate_passed=False -> live gate WATCH-only (verified PID 298681). This makes the kill durable: a sharp-but-unprofitable model can no longer auto-enable ENTER on a restart/retrain, and it can never be the flag that authorizes real capital. Monitor stays a paper data-collector (graduation outcomes+prices flow regardless of ENTER). To re-enable ENTER someone must explicitly set economic_gate_passed=True from a measured -win-side §5.4 pass. STILL OWED (off critical path, see [[pumpfun-data-quality-history]]): (a) _pool_price_from_tx native-SOL/different-owner pool reserve parsing for full price coverage; (b) defer/retry live price capture or schedule recover_grad_prices.py. DO NOT deploy capital; locked holdout untouched.

[2026-06-28] [model] author=tail-test
data_hash=tail_predictability_test.py; db=pumpfun.db priced graduates n=55   backtest_runs_to_date=14
metrics={"n": 55, "profitable_rate": 0.655, "median_payoff_at_v40": 8.26, "population_median_payoff": 0.65, "oos_auc_logistic": 0.5, "oos_auc_gbm": 0.5, "auc_ci": "[0.25,0.75]", "lift_vs_vsol": -0.018, "baseline_vsol_auc": 0.518, "univariate_insample": "n_unique_wallets +0.48 p<.001, seconds_since_launch +0.37, sol_per_min -0.35"}
decision=defer   rationale=Ran the surviving-hypothesis test after [kill] winside-bracket: do PRE-graduation features predict which GRADUATORS tail-win (high post-grad exit) vs dump? Result INCONCLUSIVE, not a kill. OOS AUC 0.500 (both classes, lift -0.018 vs entry-vSol baseline) = no measurable predictability, BUT (1) severely underpowered (n=55, test=22, AUC bootstrap CI [0.25,0.75] — cannot distinguish 0.5 from 0.75), and (2) the priced+caught sample is SURVIVORSHIP-BIASED (65.5% profitable, median 8.26x at entry vSol40) vs the true population (57% losses, median 0.65x) because price recovery skews toward still-trading survivors. In-sample univariate hints (broad accumulation: n_unique_wallets rho+0.48 p<.001, seconds_since_launch +0.37, sol_per_min -0.35) echo the paper but are on biased data = not evidence. CONCLUSION: the blocker is DATA (small + biased), not the hypothesis. Prerequisite to a real answer = improvement #1 pool-address-based price discovery (prices every graduate regardless of survival -> large UNBIASED sample), then re-run this test. Until then the tail edge is UNRESOLVED. Capital still NOT deployable (winside [kill] stands; economic gate False). Holdout untouched. See [[conviction-engine-inverted]].

[2026-06-28] [kill] author=liquidity-wall
data_hash=db=pumpfun.db; random 30 unpriced graduates x getSignaturesForAddress(limit50)+getTransaction   backtest_runs_to_date=14
metrics={"random_sample": 30, "have_any_pumpswap_tx": 0.2, "zero_pumpswap_activity": 0.8, "median_pumpswap_txs_per_token": 0, "priceable_random": 0.2, "priced_universe_is_the_20pct_survivors": true, "survivor_exit_median": 0.65, "recovery_rate_random_limit10": 0.03, "recovery_rate_random_limit50": 0.1}
decision=kill   rationale=TERMINAL economic finding — a ~15k-credit diagnostic that obviated the pool-discovery pricing build. On a RANDOM sample of unpriced graduates, only 20% have ANY PumpSwap activity; 80% graduated then went DEAD (median 0 post-grad swaps, no exit liquidity). This is not a discovery artifact: active pools DO surface in the mints signatures (one had 48 PumpSwap txs), so the 80% zeros are real illiquidity. Therefore the strategys WIN (graduation) is UNSELLABLE ~80% of the time, and the ~20% with a market exit at a loss (median 0.65x, per [kill] winside-bracket). The priced 287 were precisely the survivor 20% — so even the tail test (n=55) and the winside brackets were computed on the BEST-case slice; the true population EV is worse than the -77% median-plug. IMPLICATIONS: (1) the graduation strategy is conclusively, terminally unviable on liquidity grounds alone, independent of any model skill. (2) The tail hypothesis is practically dead: predicting WHICH graduate moons cannot help when ~80% of targets have no liquidity to exit into — the binding constraint is liquidity, not predictability. (3) Building pool-discovery pricing is NOT worth it: it would price dead tokens at unrealizable initial-reserve prices, spending credits to confirm dead tokens are dead. RECOMMENDATION: STOP. No further data collection or pricing. Firehose stays OFF (saves ~2.15M credits/day). Charter outcome: a fully-resolved, strongly-supported NO-TRADE verdict. DO NOT deploy capital; holdout untouched. See [[conviction-engine-inverted]], [[pumpfun-data-quality-history]].

[2026-06-28] [model] author=curve-momentum
data_hash=peak_exit_replay.py + curve_slippage_replay.py; db=pumpfun.db mature entry-zone n=3195   backtest_runs_to_date=14
metrics={"n": 3195, "rule": "TP3x_SL0.4x_curve_exit_pre_graduation", "grad_rate": 0.12, "flat2pct_mean": 0.076, "curveaware_mean_0.5SOL": 0.23, "curveaware_cilow_0.5SOL": 0.184, "curveaware_mean_1SOL": 0.192, "curveaware_cilow_1SOL": 0.149, "curveaware_mean_2SOL": 0.125, "curveaware_cilow_2SOL": 0.084, "capacity_dies_at_SOL": 5.0, "jackknife_pos_upto_2SOL": true, "median_ret": -0.41, "win_rate": 0.31, "ceiling_clairvoyant_mean": 1.773, "caveat_tp_tuned_in_sample": true, "caveat_unconditional_no_model": true}
decision=note   rationale=NEW DIRECTION — does NOT overturn the graduation [kill]s; it is a DIFFERENT strategy that sidesteps the liquidity wall. The kills concern holding THROUGH graduation and exiting on the DEAD post-grad PumpSwap pool (~80% unsellable). This instead enters the entry zone and EXITS ON THE BONDING CURVE before graduation, where liquidity exists by construction. Replay (peak_exit_replay.py) on 3195 mature caught-clean entry-zone tokens: clairvoyant peak-exit ceiling +177%; realistic TP3x/SL0.4x +7.6% mean (flat-2% slippage). DECISIVE stress test (curve_slippage_replay.py) replaced flat slippage with EXACT constant-product entry+exit impact and swept position size: the edge SURVIVES and improves for realistic sizes (flat-2% was too pessimistic for small trades) — 0.5 SOL mean +23.0% CI-low +18.4%, 1 SOL +19.2%/+14.9%, 2 SOL +12.5%/+8.4%, all jackknife-positive; dies at ~5 SOL (capacity limit from thin early-curve liquidity). This is the FIRST realizable, well-powered, robust positive expectancy found in the project. NOT YET AN EDGE: (1) TP/SL tuned in-sample (4 configs tried; TP2x/1.5x were negative) — MUST pass time-ordered OOS validation or it is overfit; (2) tail-carried (median -41%, 31% win, +mean rides the ~12% curve-climbers); (3) execution realism (MEV beyond 1%, monitoring latency, stop gap-through) unmodeled; (4) UNCONDITIONAL — no model selection yet (AUC-0.75 model might lift it). Right-censoring guarded (mature-only). NEXT: time-ordered OOS test of the TP/SL rule (decisive vs overfit), then model-selection lift, then execution stress. Graduation kills stand; capital NOT deployable. Holdout untouched. See [[conviction-engine-inverted]].

[2026-06-28] [model] author=curve-momentum-oos
data_hash=oos_validation.py; db=pumpfun.db; early/late time split of mature entry-zone tokens; size 1 SOL   backtest_runs_to_date=14
metrics={"traded_n_total": 3195, "early_select_share": 0.6, "selected_rule": "TP5x_SL0.3x", "early_mean": 0.286, "oos_late_mean": 0.369, "oos_late_cilow": 0.268, "oos_jackknife_pos": true, "oos_win_rate": 0.251, "ref_tp3sl04_oos_mean": 0.213, "ref_tp3sl04_oos_cilow": 0.148, "whole_grid_oos_avg": 0.124, "whole_grid_frac_positive": 0.82, "early_late_rank_spearman": 0.99}
decision=note   rationale=Time-ordered OOS validation of the pre-graduation curve-momentum rule (guards the in-sample TP/SL tuning that produced false positives earlier). Selected TP/SL on the EARLIEST 60% of mature entry-zone tokens by early mean return, froze it, tested on the LATEST 40% (launched after everything used to select). RESULT: PASSES. Selected TP5x/SL0.3x -> OOS mean +36.9%, CI-low +26.8%, jackknife-positive, 25% win; early-vs-late config ranking Spearman 0.99 (tuning transfers near-perfectly); 82% of the whole TP/SL grid is OOS-positive (broad, not cherry-picked); reference TP3x/SL0.4x also holds OOS (+21.3%, CI-low +14.8%). Exact constant-product slippage, 1 SOL size. So the #1 threat (overfit selection) is CLEARED and the edge is robust within-window. REMAINING, UNVALIDATED (in priority): (1) SINGLE REGIME — all data is one ~1.5-day capture window; this OOS is out-of-sample but NOT out-of-regime; cross-time/regime generalization is untested and is now the dominant risk (cannot be settled on existing data — needs forward collection or more history). (2) Execution realism unmodeled: MEV beyond 1%, exec/monitoring latency at the TP trigger, stop gap-through on fast dumps. (3) High-variance lottery (25% win, tail-carried) — needs drawdown/bankroll analysis. (4) Unconditional (no model selection). NEXT FREE TESTS (no Helius): conservative execution-friction stress, then drawdown analysis, then model-selection lift. Forward paper-testing (addresses regime+execution) needs the firehose back ON = credits. Graduation [kill]s stand; capital NOT deployable (Charter §5.4 unmet: execution+regime unproven); holdout untouched. See [[conviction-engine-inverted]].

[2026-06-28] [model] author=curve-momentum-execstress
data_hash=exec_stress.py; OOS late slice; frozen TP5x/SL0.3x; size 1 SOL   backtest_runs_to_date=14
metrics={"n_oos": 1469, "baseline_mean": 0.369, "baseline_cilow": 0.269, "exit_lat1_mean": 0.303, "entry_lat1_mean": 0.217, "mev3_mean": 0.194, "haircut2_mean": 0.147, "haircut2_cilow": 0.061, "pessimistic_mean": 0.101, "pessimistic_cilow": 0.018, "pessimistic_jackknife": true, "pessimistic_assumptions": "entry+exit latency 1 swap, MEV 5%, adverse haircut 3%/side", "win_rate": 0.22}
decision=note   rationale=Execution-friction stress on the OOS late slice with the frozen OOS-selected rule (TP5x/SL0.3x). Layered realistic frictions: exit latency (fill at the NEXT swap after the TP/SL trigger -> captures TP reversion and stop gap-through), entry latency, MEV beyond 1%, and an adverse-fill haircut. RESULT: edge SURVIVES every step monotonically and never turns negative — pessimistic combined scenario (entry+exit latency, MEV 5%, haircut 3%/side) still mean +10.1%, CI-low +1.8%, jackknife-positive. Frictions eat ~2/3 of the perfect-fill edge (+37%->+10%), so EXECUTION QUALITY is now a first-order driver: a clean low-latency bot keeps most of it; a sloppy one lands near the +1.8% marginal edge. VALIDATION SCORECARD now: slippage (exact curve, capacity ~2 SOL) PASS; overfit (time-ordered OOS, rank-transfer rho 0.99, 82% of grid positive) PASS; execution frictions PASS. REMAINING: (a) SINGLE-REGIME generalization (~1.5-day window) — dominant risk, NOT testable on existing data; (b) high-variance lottery (22-25% win, tail-carried) — needs drawdown/bankroll analysis (free, next); (c) model-selection lift (free, optional). Forward paper-test addresses (a)+real execution but needs the firehose back ON = credits. Graduation [kill]s stand; capital NOT deployable (Charter §5.4: regime + live execution unproven); holdout untouched. See [[conviction-engine-inverted]].

[2026-06-28] [model] author=curve-momentum-drawdown
data_hash=drawdown_analysis.py; OOS slice; 1 SOL/trade; moderate frictions   backtest_runs_to_date=14
metrics={"n": 1469, "span_hr": 13.7, "trades_per_hr": 108, "win_rate": 0.225, "mean_pnl_sol": 0.147, "total_window_pnl_sol": 215.8, "worst_trade_sol": -0.961, "longest_losing_streak": 31, "max_drawdown_sol": 34.5, "maxdd_pct_of_pnl": 0.16, "peak_concurrent": 26, "bankroll_to_survive_sol": 60, "return_on_bankroll_window": 3.57, "per_trade_std": 1.742, "return_over_maxdd": 6.3}
decision=note   rationale=Drawdown/bankroll analysis of the curve-momentum edge on the OOS slice, 1 SOL/trade, moderate frictions (+14.7% mean scenario). Time-ordered equity curve: total +215.8 SOL over 13.7h, max drawdown 34.5 SOL (return/maxDD ~6.3), longest losing streak 31 trades, peak 26 concurrent positions -> ~60 SOL bankroll to survive, worst single trade -0.96 SOL (stop does not save a straight gap-down to the vSol floor), per-trade std 174%. VERDICT: SURVIVABLE but high-variance and capital-intensive; the strong return/risk holds ONLY in-regime. The +357%/14h on bankroll is an IN-REGIME number, NOT a forward expectation, and must not be cited as one. FREE VALIDATION GAUNTLET NOW COMPLETE and PASSED: slippage/capacity (~2 SOL/pos, ~52 SOL strategy capacity), overfit (time-ordered OOS, rho 0.99), execution frictions (pessimistic +10%/CI+1.8%), drawdown (survivable on ~60 SOL). The SOLE remaining material risk is REGIME GENERALIZATION (~1.5-day single window) — NOT testable on existing data. To settle it requires forward paper-testing (firehose ON, ~2.15M credits/day; ~5.2M remaining = ~2.4 days = one fresh out-of-regime sample) and/or more historical data. Remaining FREE optional test: model-selection lift (AUC-0.75 subset vs basket — may raise win rate / cut variance). Graduation [kill]s stand; capital NOT deployable (Charter §5.4: regime + live execution unproven); holdout untouched. DECISION POINT: spend ~2 days of credits to forward paper-test (decisive on regime), or stop at promising-but-unproven-out-of-regime. See [[conviction-engine-inverted]].

[2026-06-28] [model] author=curve-momentum-modelsel
data_hash=model_selection_lift.py; gbm refit on EARLY entry-zone, score LATE OOS; TP5x/SL0.3x moderate frictions   backtest_runs_to_date=14
metrics={"basket_n": 1469, "basket_mean": 0.147, "basket_cilow": 0.058, "basket_win": 0.225, "top50_n": 734, "top50_mean": 0.24, "top50_cilow": 0.106, "top50_win": 0.26, "top50_grad": 0.142, "top25_mean": 0.154, "top25_cilow": -0.025, "top10_mean": -0.057, "top10_cilow": -0.282, "bottom50_mean": 0.054, "bottom50_grad": 0.071}
decision=note   rationale=Model-selection lift, leak-free (gbm graduation model REFIT on EARLY entry-zone tokens only, scores LATE OOS tokens at entry snapshot). Entering the model top-50% subset LIFTS the curve-momentum edge: mean +24.0% vs basket +14.7%, CI-low +10.6% vs +5.8%, win 26.0% vs 22.5%, grad 14.2% vs 10.6% — on HALF the trades (better capital efficiency). Clear top-vs-bottom separation (top50 +24.0% vs bottom50 +5.4%) confirms the AUC-0.75 model discriminates curve-winners. CAVEAT: non-monotonic at finer cuts — top-25% +15.4% (CI-low -2.5%, n.s.) and top-10% -5.7% (CI-low -28%, n=146) go noisy/negative; at 174% per-trade std this is likely noise but is an honest flag that the highest-confidence picks are NOT reliably the best trades. USE a moderate filter (~top 50%), not the extreme tail. FREE VALIDATION+IMPROVEMENT GAUNTLET COMPLETE: slippage PASS, OOS-overfit PASS, execution-friction PASS, drawdown survivable, model-selection LIFTS (+24%/CI+10.6% top-50%). SOLE remaining material risk = REGIME generalization (single ~1.5-day window), only settleable by forward paper-testing (credits) or more history. Graduation [kill]s stand; capital NOT deployable (Charter §5.4: regime + live execution unproven); holdout untouched. See [[conviction-engine-inverted]].

[2026-06-30] [model] author=ops-and-demotions
data_hash=git curve_momentum.py/live_monitor.py/config.py 2026-06-30   backtest_runs_to_date=14
metrics={"forward_n": 722, "forward_cilow": 0.047, "regime_windows_pos": "6/6", "smart_exit_regime_delta": "W2 +11.5%->+10.0% (drag)"}
decision=note   rationale=Batch of operational decisions, logged late (entry was interrupted 2026-06-30). (1) SMART-EXIT DEMOTED TO LOG-ONLY: walk-forward regime test (regime_test.py, 6 disjoint windows, smart list fit strictly pre-window) showed the 2-seller force-exit was a mild net drag and fired ~0x live; smart_sells still recorded per trade. (2) REGIME TEST PASSED: all 6 windows CI-low positive (+0.3% .. +11.3%), pooled n=5063 — but all windows are INTRA-week (one ~4-day capture); the >=2-distinct-weeks §5.4 criterion is NOT met by this data. (3) HOLDOUT NOT FENCED: HOLDOUT_START_TS stays None — user budget cannot fund a continuous >=2wk capture; firehose switched to interval duty-cycle (90s/3600s, later 300s/3600s 2026-07-01) ~54k->180k credits/day vs 2.15M continuous. Interval-era data is TRAINING/base-rate collection, NOT a holdout (duty-cycling can't observe full trajectories). §11 pre-registration marked PREPARED-NOT-ACTIVE. (4) Gap filter added to the forward reporter (_chk.py + interval_windows.jsonl): trades whose lifetime spans an OFF blackout are excluded (path-blind exits); 6 such evict-trades dropped as of 2026-07-01. Capital remains NOT deployable.

[2026-07-02] [threshold] author=winrate-improvement-study
data_hash=pumpfun.db 4.11M swaps / 189.7k tokens thru 2026-07-01; latest.pkl (de-skewed 11-feat)   backtest_runs_to_date=17
metrics={"forward_n": 940, "fwd_all": {"win": 0.586, "mean": 0.082, "cilow": 0.035}, "fwd_band_070_115": {"win": 0.551, "mean": 0.014, "cilow": -0.077}, "fwd_gate115": {"win": 0.599, "mean": 0.107, "cilow": 0.050}, "replay_test_070": {"win": 0.601, "cilow": 0.092}, "replay_test_115": {"win": 0.620, "cilow": 0.116}, "regime_windows_115_better": "5/6, win never lower"}
decision=pass   rationale=Win-rate improvement study (runs #15-17: improve_entry_filter.py, trail_stress.py, gate A/B). NEGATIVE results first, so they can't be retried later as "new" ideas: (a) five as-of-entry quality filters (velocity, sell_share, pre-entry retrace, buy concentration, n_buyers) add ~nothing beyond the model gate on the time-ordered test slice (best +0.8pt CI-low, within noise) — the model's 11 features already carry the signal; (b) the trailing-runner exit's apparent doubling of mean (+29% vs +17%) was a FILL-CONVENTION ARTIFACT — exiting at the stop level instead of the breaching swap's price; under pessimistic fills + 1-swap latency it is +1-2pt mean, win rate FLAT-to-lower, and loses to baseline in 4/6 regime windows. REJECTED. POSITIVE: MODEL_SCORE_MIN 0.070 -> 0.115. This was PRE-COMMITTED (the 0.070 comment said tighten to ~0.115 only if the forward test confirms) and the trigger fired: forward true-OOS n=940 shows the [0.070,0.115) band is dead weight (win 55.1%, +1.4%, CI-low -7.7%) vs >=0.115 (win 59.9%, +10.7%, CI-low +5.0%); time-ordered replay test slice agrees (win 62.0% vs 60.1%, CI-low +11.6% vs +9.2%); 6-window regime check: equal-or-better CI in 5/6, win rate never lower. Keeps ~70% of entries. Velocity stays LOG-ONLY (forward-confirmed directionally but no additive lift over the score gate; corr(vel,ret)=+0.036). §11 amended + re-frozen (no holdout existed; nothing burned). Forward log cutover: trades after this restart trade the 0.115 gate — slice cm_trades.jsonl by model_score for cross-era comparisons. Capital remains NOT deployable (§5.4 unmet: no >=2wk holdout).
