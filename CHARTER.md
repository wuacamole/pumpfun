# Project Charter — Pump.fun Conviction System

> The constitution of this project. Where this document and any other document,
> commit message, or in-the-moment temptation disagree, **this document wins**
> until it is formally amended via the change protocol in §9. Read it before
> every phase gate.

---

## 1. Purpose (the one sentence)

Determine, honestly and reproducibly, **whether a real-time, post-cost trading
edge exists** in pump.fun launches — and refuse to deploy capital until that
question is answered *yes* by evidence that was specified before it was seen.

The deliverable of this project is **a true answer**, not a profit. "No edge,
do not trade" is a successful outcome, not a failure. A system that reaches that
verdict and stops has done its job perfectly.

---

## 2. Prime directive (the anti-drift principle)

> **A fast system trading a non-existent edge loses money efficiently.**

Every optimization — latency, Rust, more features, more capital — multiplies
whatever sign the edge has. If the edge is negative or zero, optimization makes
the loss *worse*, faster. Therefore: **nothing that increases trade frequency,
capital at risk, or execution speed may be built until the edge is proven by the
gate in §5.4.** Proof comes first. Speed is a reward for proof, never a substitute.

---

## 3. Invariants (true in every phase, no exceptions)

These do not have unlock conditions. They are not negotiable by a good backtest.

| # | Invariant | Enforced by |
|---|-----------|-------------|
| I1 | **No real capital** moves until the §5.4 capital gate passes on the locked holdout. Phase 4 stays unbuilt until then. | Human discipline + this charter |
| I2 | **The goal metric is post-cost realized PnL of the actual strategy**, never model AUC/Brier in isolation. Calibration is supporting evidence, never sufficient evidence. | §7 G7 |
| I3 | **Log every token's outcome**, including ones the entry rule would skip. The training set must reflect the full population, not the entered subset. | `storage.py` set_outcome on all mints |
| I4 | **No feature may use information past its `as_of` slot.** Lookahead is a correctness bug, not a tuning choice. | `model.featurize` no-lookahead tests |
| I5 | **Costs are date-keyed and real**: every fill pays the fee schedule live at its timestamp, plus slippage and MEV. No cost may be softened to make a result pass. | `config.fee_at(ts)`, `paper_trade`, `backtest` |
| I6 | **The locked holdout (§9.1) is touched exactly once.** Looking at it twice voids it. | Decision log + git |
| I7 | **Edge decays.** A pass is never permanent. Live calibration and base-rate monitors run continuously; capital de-risks automatically on drift. | `monitor` AlertStream + §8 |

If you find yourself wanting to relax an invariant to make a number look better,
**that is the signal to stop and write in the decision log**, not to relax it.

---

## 4. The falsifiable hypothesis (pre-registered)

State it sharply enough that data can kill it:

> There exists a **real-time-computable subset** of pump.fun launches — defined
> by conviction signals (fast SOL accumulation via few human trades, low bot
> share, and the model's calibrated `P(graduation)`) — for which
> `P(graduation) > (vSol / graduation_vsol)²` by a margin that survives fees,
> slippage, and MEV, **and this holds out-of-time on data the selection process
> never saw**, producing a post-cost mean return whose 95% bootstrap CI lower
> bound is greater than zero.

Grounding: the paper (Lillo et al., arXiv 2602.14860) finds fast accumulation
via few trades is the strongest graduation predictor and high bot share predicts
failure — but a *statistical* predictor is not yet an *economic* edge. This
project's job is to test whether the gap between them closes after costs. The
base rate is ~1% and drifts (0.63% in the paper's window, ~1.15% in early 2026),
so the binding constraint is **statistical power**, addressed in §5.4.

The hypothesis is **rejected by default.** The burden of proof is on the edge.

---

## 5. Phase gates (quantitative; entry and exit per phase)

Each phase is locked until its entry gate passes. You may not skip ahead, and you
may not start optimizing a later phase to "save time" — see §10.

### 5.1 Phase 1 — Data integrity (DONE per CHANGES.md)
**Exit gate (must hold before trusting any downstream number):**
- Fee schedule is date-keyed and reproduces the known 2026-05-21 transition.
- `graduation_vsol` is derived on-chain, not hardcoded.
- USDC-paired tokens are detected and excluded from SOL-denominated scoring.
- PumpSwap migration is detected; graduation events fire reliably.
- Slot-gap detection + backfill active; gap rate measured and logged.
- All atomic-write paths verified (no torn swap/token rows, no half-written Parquet).

### 5.2 Phase 2 — Honest measurement
**Entry:** Phase 1 exit gate green.
**Exit gate:**
- Backtest reports **PR-AUC and log-loss** (not hit rate/accuracy) as primary
  classifier metrics, given <1% positives.
- Probabilities are **calibrated** (isotonic/Platt); a reliability diagram is
  produced each fold.
- Lookahead audit passed on every feature, especially trader-history and
  creator-history features (success labels computed strictly as-of trade slot).
- Walk-forward edge-over-time is plotted; the result is **not** a single lucky fold.

### 5.3 Phase 3 — Live paper validation
**Entry:** Phase 2 exit gate green.
**Exit gate:**
- ≥ **6 continuous weeks** of live paper trading on the production feed, fees and
  slippage included, surviving restarts (PnL track unbroken).
- Live calibration tracks backtest calibration within tolerance (no train/live gap).
- Gap rate low enough that missed-event bias is negligible (pre-register the ceiling).

### 5.4 Phase 4 — Capital gate (THE gate; all must pass on the locked holdout)
**Entry:** Phase 3 exit gate green **and** the locked holdout (§9.1) is still untouched.

Evaluated **once**, on data strictly later in time than all training and all model
*selection*:

| Criterion | Threshold (pre-register exact values in §11) |
|-----------|----------------------------------------------|
| Post-cost mean return per ENTER | 95% bootstrap CI **lower bound > 0** |
| Robustness to luck | Result stays positive after **deleting the single best trade** (jackknife) |
| Power: graduations observed | ≥ **N_grad** graduations *inside the entered subset* (default 30; lottery-skewed returns need this) |
| Power: signals observed | ≥ **N_enter** independent ENTER signals (default 300) |
| Calibration | Reliability slope in [0.8, 1.2]; predicted vs realized graduation rate within **±20%** |
| Regime coverage | Holdout spans ≥ **2 distinct weeks** and ≥ **1 fee or base-rate regime** boundary if available |
| Stability | Edge sign positive in **every** walk-forward fold of the final window, not just the aggregate |

**If any single criterion fails, the gate fails.** No partial credit, no
"close enough," no re-running with a tweaked subset (that consumes the holdout —
see §9.1). A failed gate routes to §6.

---

## 6. Kill criteria (when to conclude "no edge" and stop)

Pre-committing these is the whole point. Declare **no edge** and halt capital work
permanently (absent a §9 amendment with a genuinely new hypothesis) if any hold:

- **K1 — Time/data box:** after **12 weeks** of live paper data *or* **M observed
  graduations** in the full population (default 500), no candidate subset clears
  the §5.4 gate.
- **K2 — Holdout failure:** the locked holdout is consumed (§5.4 evaluated) and the
  gate fails. The holdout is gone; you do not get another from the same era.
- **K3 — Edge is luck:** the only positive results depend on ≤ a handful of trades
  (fails the jackknife) or appear in a single fold.
- **K4 — Calibration collapse:** model probabilities cannot be calibrated to within
  the §5.4 band on out-of-time data — the breakeven comparison is then meaningless.
- **K5 — Cost-eaten edge:** a raw-probability edge exists but is entirely consumed
  by fees+slippage+MEV at realistic sizes. This is still "no tradeable edge."

Reaching a kill criterion is a **win for the project's purpose.** Record it in the
decision log, write the post-mortem, and keep the dataset — it has scientific value
and the next era's hypothesis may differ.

---

## 7. Drift: the specific ways this project will try to fool you

Named so you can catch them in yourself. Each has a guardrail.

| ID | Drift pattern | What it looks like | Guardrail |
|----|---------------|--------------------|-----------|
| G1 | **Premature execution** | "Let me just wire a tiny live order to test plumbing." | I1. Phase 4 unbuilt until §5.4. Plumbing is tested in paper, period. |
| G2 | **Goalpost moving** | Lowering a §5.4 threshold after seeing it miss. | §9 change protocol logs every threshold edit and **voids the pre-registration**, forcing a fresh holdout. |
| G3 | **Backtest p-hacking** | Trying many subsets/features/windows until one passes. | Locked holdout (§9.1) touched once; backtest-run counter in decision log; the more configs tried, the more the final result is suspect. |
| G4 | **Sunk-cost escalation** | "We've spent weeks; the edge *must* be here somewhere." | K1/K2 are time- and holdout-boxed. Effort spent is not evidence of edge. |
| G5 | **Proxy worship (Goodhart)** | Optimizing graduation AUC instead of post-cost PnL. | I2. The gate metric is realized strategy PnL; AUC is supporting only. |
| G6 | **Survivorship/selection bias** | Training only on tokens the rule would enter. | I3. Outcomes logged for the full population; backtest scores the rule against all. |
| G7 | **Cost softening** | Quietly dropping MEV, using stale low fees, optimistic slippage. | I5 + date-keyed `fee_at`; costs may only move toward *more* conservative. |
| G8 | **Stationarity assumption** | Treating a 2025-fit model as valid in 2026. | I7 + base-rate monitor; re-fit on regime shift; never assume the edge is fixed. |
| G9 | **Scope creep into speed** | Rust rewrite / colocation before proof. | §10 "not now" list with explicit unlock = §5.4 pass. |
| G10 | **Definition drift** | "Edge" quietly redefined to whatever the data showed. | §4 hypothesis is frozen; redefinition requires §9 amendment, new holdout. |

---

## 8. Drift detection and automatic response (runtime)

The system must surveil *itself* for drift, not just the market:

- **Calibration-drift alert** (`monitor.AlertStream`): live predicted vs realized
  graduation diverges past band → alert, and if capital is ever live, **auto-reduce
  size to zero** pending human review.
- **Base-rate monitor:** rolling population graduation rate; on a regime shift,
  flag all fitted models as **stale** and require re-fit before further reliance.
- **Gap-rate monitor:** if `PumpLogsFeed.gaps` exceeds the pre-registered ceiling,
  mark the affected window as **low-integrity** and exclude it from training/eval.
- **Researcher-degrees-of-freedom counter:** every distinct backtest configuration
  is appended to the decision log. This number is reported at the §5.4 gate; a high
  count raises the effective bar (multiple-comparisons honesty).

---

## 9. Governance machinery

### 9.1 The locked holdout
- Reserve the **most recent contiguous time slice** (pre-register the span, e.g.
  the final 2–3 weeks) as the holdout. Fence it by timestamp in code.
- It is **never** used for fitting, feature selection, threshold tuning, or
  "just looking." It is read exactly once, at the §5.4 gate.
- Touching it (any query that informs a modeling decision) **consumes** it. A new
  holdout must then come from a *future* era, which costs calendar time — that
  cost is the deterrent, and it is intentional.

### 9.2 The decision log (`DECISIONS.md`, append-only)
Every gate decision, threshold, and material modeling choice gets an entry:
```
[YYYY-MM-DD] [gate|threshold|model|kill] author=…
data_hash=<git+db snapshot hash>   backtest_runs_to_date=<n>
metrics={post_cost_mean, ci_low, brier, n_grad, n_enter, …}
decision=<pass|fail|defer|kill>   rationale=<one paragraph>
```
No silent decisions. If it isn't in the log, it didn't happen — and it can't be
cited later as evidence the gate was met.

### 9.3 Change protocol (amending this charter or any threshold)
To change a §5.4 threshold, a kill criterion, the hypothesis, or an invariant:
1. Write a decision-log entry: old value, new value, **why now**.
2. Acknowledge in writing that the change **voids the current pre-registration**.
3. If the change touches anything evaluated against the holdout, the **holdout is
   burned** and a fresh one is reserved from future data.
4. Only then edit this file. Charter edits are commits, reviewed like code.

This makes goalpost-moving *expensive and visible*, which is the only thing that
reliably prevents it.

---

## 10. The "not now" list (scope discipline)

These are good ideas. They are forbidden until their unlock condition fires. The
point is not that they're wrong — it's that doing them now optimizes an unproven
edge (G9).

| Item | Unlock condition |
|------|------------------|
| Phase 4 execution engine (any live orders) | §5.4 capital gate passes |
| Rust / yellowstone-grpc hot path, colocation, latency work | §5.4 passes **and** a measured drop-rate or latency problem is shown to hurt classification |
| Position sizing beyond minimum | First **4 weeks** of live capital show realized PnL matching paper within tolerance |
| New exotic features beyond the paper's four signal families | Current feature set is shown insufficient *and* lookahead-audited |
| Multi-venue / Raydium-legacy support | A concrete need appears in logged data |

When tempted, add the idea here with its unlock condition. That honors it without
acting on it.

---

## 11. Pre-registration block (freeze before the final evaluation)

Fill this in, commit it with a git hash, and **do not edit it after looking at the
holdout** (editing = §9.3, burns the holdout). This is the contract with your
future self.

```
PRE-REGISTRATION — frozen YYYY-MM-DD, commit <hash>
Hypothesis (verbatim from §4): …………………………………………………
Entry rule (exact, code-referenced): ………………………………………
Holdout span (timestamps): …………  →  …………   [fenced in config: …]
Primary gate metric: post-cost mean return per ENTER, 95% bootstrap CI low > 0
Thresholds:
  N_grad (graduations in entered subset) = 30
  N_enter (independent signals)          = 300
  calibration reliability slope in        [0.80, 1.20]
  predicted vs realized grad rate within  ±20%
  jackknife: positive after removing best trade = required
  fold stability: positive in every final-window fold = required
Kill box: 12 weeks live paper OR 500 population graduations, whichever first
Backtest runs consumed before freeze: <n>   (report at gate)
Signed: ………………
```

---

## 12. Operating cadence

- **Weekly:** review AlertStream output, base-rate monitor, gap rate; append a
  one-line decision-log note even if "no change." Confirm no invariant slipped.
- **At each phase gate:** full decision-log entry with data hash and metrics; a
  second person (or a 48-hour cooling-off if solo) signs off before advancing.
- **At the capital gate (§5.4):** maximum ceremony. Holdout opened once, all seven
  criteria checked, pass/fail recorded irrevocably. No same-day reversal.
- **If a kill criterion fires:** stop, write the post-mortem, keep the data,
  celebrate having avoided lighting money on fire efficiently.

---

### One paragraph to remember if you read nothing else

This project exists to answer a yes/no question without lying to yourself. The
edge is ~1% rare, decays, and is exactly the size that wishful tuning can fake.
The only defense is to decide *what would convince you* before you look, write it
down, and let a single untouched slice of data deliver the verdict. If it says no,
the right move — the move this whole system was built to make easy — is to not
trade.
