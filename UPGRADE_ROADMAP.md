# Pump.fun Conviction System — Upgrade Roadmap

**Purpose of this document.** This is the build plan for turning the current
three-file prototype (`conviction_engine.py`, `live_monitor.py`,
`test_scenarios.py`) into a system that can be trusted with real capital — or,
just as importantly, that can prove to you it *shouldn't* be. Each upgrade below
states what it does and *why it matters*, ordered so that the cheapest,
highest-information work comes first. The guiding principle: **validate that the
edge exists before optimizing anything.** A fast system trading a non-existent
edge just loses money efficiently.

---

## Status (updated 2026-06-27)

> **Phases 1 and 2 are built, and the edge-validation step they exist to enable
> has run. The verdict so far is NO demonstrated edge.** See `DECISIONS.md`
> [kill] 2026-06-27 and [[conviction-engine-inverted]]. This is a *successful*
> outcome per `CHARTER.md` §1 — the project is here to answer the question, and
> the current answer is "no." The remaining work is narrow (§"Where this leaves
> us" below): find a feature with real *at-fixed-vSol* signal under strict
> anti-confound guards, or invoke a kill criterion (Charter §6). Do **not** treat
> the items below as a to-do list to keep grinding — most are done, and the ones
> left are gated on a real signal existing.

| Component | Status | Honest limitation / finding |
|---|---|---|
| Conviction scoring math | Working, tested | Sound — implements the paper directly |
| Breakeven check `p > (vSol/115)²` | Working | Exact, from the paper |
| On-chain reserves / constants | **Done** | `config.py` derives vSol_grad, fees date-keyed (§5.1) |
| Bot classification (`isBot`) | **Done, but discredited** | `is_direct` (invoke-depth) parsed at 0 RPC, but it *conflates* website humans with naive bots; the per-program route taxonomy (the correct version) shows **no at-fixed-vSol edge** (DECISIONS [model] 2026-06-27) |
| Graduation probability | **Fitted (logistic/GBM), calibrated** | The model is a **vSol-level detector**: vSol alone scores OOS AUC 0.86; behavioural features *degrade* it. Entry-zone AUC ≤ 0.5 → live gate neutered to WATCH-only |
| Persistent logging | **Done** | `storage.py` SQLite; full-population outcomes (Charter I3) |
| Walk-forward backtest | **Done** | `backtest.py`, vSol-matched baseline + entry-zone AUC; now censoring-guarded |
| Outcome labels | **Right-censoring known & guarded** | `stalled` only written by the paper broker at entry + `time_stop_sec`; recent tail is fake-graduate-heavy → maturity guard (`--maturity-sec`, default 3600) added to backtest/refit/route test |
| Paper trading | Built | Costs (fees/slippage/MEV) modelled; not yet run for the §5.3 6-week window |
| Execution | **None — gated** | Charter §10 / Phase 4; stays unbuilt until §5.4 passes |

---

## Phase 1 — Make the signal real ✅ DONE (with one caveat that changed the conclusion)

Goal: every input the model sees should be the *true* value, not an approximation.
Until this is done, no backtest or PnL number means anything. **All three items
below shipped.** The surprise: doing 1.1 *correctly* (real instruction-stack
routing) is what revealed the bot signal doesn't separate graduates — see the
caveat under 1.1.

### 1.1 RPC-based bot classifier — **DONE, and it falsified the premise**
> **Done at zero RPC** (`helius.is_direct_pump_call` / `entry_program`, parsed
> from the `logsSubscribe` logs we already receive). But the on-chain audit
> showed the paper's `isBot = direct call` definition **does not map cleanly to
> pump.fun**: the website frontend itself submits a *top-level* pump call, so
> `is_direct=1` lumps website humans with naive bots, while the wrapped routes
> (`is_direct=0`) are dominated by the sophisticated bots (flashx, mayhem,
> maestro). The correct per-program taxonomy was tested (`route_feature_test.py`)
> and shows **no at-fixed-vSol edge** — the apparent signal was a right-censoring
> artifact. This was the highest-priority item precisely because it could
> falsify the premise, and it did. See DECISIONS.md [model] 2026-06-27.

The original plan text is kept below for provenance.

### 1.1 (original) RPC-based bot classifier — **highest priority**
- **What:** Replace the size heuristic in `_approx_via_frontend` with the paper's
  actual method. Fetch each trade's transaction via an RPC provider and inspect
  the instruction / program-id stack: trades routed through the pump.fun frontend
  program are human; trades calling the bonding-curve program directly are bots.
- **Why:** Bot share is one of the strongest signals in the model, and the paper's
  central finding (high bot share predicts *failure*) only holds if you measure it
  correctly. A wrong bot flag corrupts the one variable that most separates
  winners from churn traps.
- **Effort:** Medium. Needs an RPC provider (Helius, Triton, QuickNode) and
  per-transaction parsing. Watch rate limits — batch or use a streaming geyser feed.

### 1.2 True bonding-curve reserve state — **DONE**
- **What:** Read `vSolInBondingCurve` / real reserves directly from the feed or
  on-chain account state, instead of reconstructing it from running trade history.
- **Status:** Done — the Helius `TradeEvent` carries true on-chain reserves;
  `config.decode_bonding_curve` decodes account state; USDC-paired tokens excluded.
- **Why:** Every downstream number — % to graduation, breakeven, conviction — is a
  function of vSol. Drift in the reconstructed value silently biases every decision.

### 1.3 Persistent logging of everything — **DONE**
- **What:** Write every observed token's full trade history + final outcome
  (graduated / stalled / rugged) to a local store.
- **Status:** Done — `storage.py` (SQLite). Outcomes logged for the full
  population (Charter I3). ~848k swaps / 33k tokens at time of writing.
- **Why:** This is your training set. Highest return-on-effort item, and it paid
  off: the data is what produced the honest no-edge verdict.

### 1.4 Outcome-maturity / right-censoring guard — **DONE (added 2026-06-27)**
- **What:** Exclude tokens whose outcome is not yet settled from training/eval.
- **Why:** A `stalled` label is only written by the paper broker at
  `entry + time_stop_sec` (30 min); graduation labels are real-time. So the most
  recent cohort is **right-censored** — its resolved tokens are
  fake-graduate-heavy (graduation-rate-by-launch-hour climbs to a spurious 100%
  in the final hours). Any feature correlated with "launched late" then rides a
  fake edge. This is what nearly fooled the route-feature test.
- **Status:** `--maturity-sec` (default 3600 = `time_stop_sec` 1800 + entry/settle
  cushion) in `backtest.py`, `refit_model.py`, `route_feature_test.py`. Widen if
  `time_stop_sec` grows. **Caveat still open:** `stalled` is only written for
  *paper-entered* tokens, so the stalled set is a selected subset — a separate
  bias from censoring, worth closing (mark stalled for the full population).

---

## Phase 2 — Replace guesswork with a fitted model ✅ DONE — verdict: NO EDGE

Goal: turn the hand-built `_estimate_graduation_prob` into something calibrated on
your own data. **All three items shipped, and the moment of truth (2.3) returned
a negative answer**, which is the result that matters.

### 2.1 Fit a real graduation model — **DONE**
- **Status:** `model.py` / `refit_model.py` fit logistic + GBM, isotonic/Platt
  calibrated, no-lookahead featurize, GroupKFold-by-mint, time-ordered OOS.
- **What we learned:** the model is a **vSol-level detector**. vSol *alone* scores
  time-ordered OOS AUC ≈ 0.86; the full model scores ~0.65 — the behavioural
  features *degrade* the trivial signal. In the entry zone [33,50] vSol the model
  is anti-predictive (AUC < 0.5). The deploy guard now **refuses to ship** a model
  with entry-zone AUC ≤ 0.5; the live gate is WATCH-only.

### 2.2 New predictive features — **DONE; none earned a place**
- Creator history (`creator_graduated_count` / `creator_rugged_count`): wired, no
  lift. `top_holder_share`: reconstructed as-of, has univariate signal but ~0
  marginal OOS AUC (redundant) → computed but excluded. `smart_money_present`:
  dropped (no honest data source; would be target leakage). Route-category shares
  (`route_feature_test.py`): no robust at-fixed-vSol lift (censoring artifact).
- **The bar (keep it):** a feature must show OOS lift over a **vSol- *and*
  launch-time-matched** baseline, in **both** model classes, before it joins
  `FEATURE_COLUMNS`. A GBM-only lift is an overfit/confound signature.

### 2.3 Walk-forward backtest harness — **DONE — the moment of truth**
- **Status:** `backtest.py` — vSol-matched baseline, entry-zone AUC, bootstrap CI,
  jackknife, fold stability, censoring guard. Reports the §5.4 criteria.
- **Result:** no demonstrated post-cost edge. The earlier "+edge" was an artifact
  of entering at already-high vSol vs an un-matched baseline; vSol-matched, the
  lift is ~0 (DECISIONS [kill] 2026-06-27). This is the single most important
  deliverable, and it did its job: it found the absence of an edge before capital.

---

## Where this leaves us (the actual decision point)

The build plan is essentially complete through Phase 2, and the answer it
produced is **no demonstrated edge**. Per Charter §1 and §6 that is a *result*,
not a failure. There are exactly two honest forward paths — and the trap to avoid
(Charter §7 G3/G4) is grinding new features to make the verdict change.

1. **Find a feature with real *at-fixed-vSol* signal.** The bar is fixed and
   strict: lift over a **vSol- and launch-time-matched** baseline, in **both**
   logistic and GBM, on censoring-guarded data. Candidates not yet tested that
   are *not* obvious vSol/time proxies: trade-size *distribution* shape, inter-
   trade timing microstructure, first-buyer wallet age/funding. Each must clear
   `route_feature_test.py`-style scrutiny before touching `FEATURE_COLUMNS`. If
   one clears, it re-opens Phase 2.3 → 3.
2. **Invoke a kill criterion (§6) and stop.** K1 (time/data box) or the standing
   evidence that the paper's premise doesn't hold here. Write the post-mortem,
   keep the data. This is a clean win for the project's stated purpose.

Either way: **collect more data.** The current window is ~1 day with only ~74
mature graduates in the clean route window — Charter §4's binding constraint is
statistical power, and most features can't be confirmed *or* killed at this n.
Letting the monitor run accumulates the settled graduations that any verdict
needs. This is the one "keep going" activity that isn't drift.

Do **not**, in this state: wire execution, add capital, reach for Rust, or lower
a §5.4 threshold. All are forbidden until §5.4 passes (Charter §10).

---

## Phase 3 — Paper trade before anything touches money

> **Gated on Phase 2 finding a real edge first.** The harness is built, but the
> §5.3 6-week live-paper run is only meaningful once a model exists that clears
> the entry-zone bar. Today the live gate is WATCH-only, so a paper run would be
> measuring an abstaining strategy. Run it for real once item ★ produces a
> deployable model.

### 3.1 Simulation / paper-trading mode
- **What:** Run the live system end-to-end, "entering" and "exiting" positions on
  paper, tracking simulated PnL with realistic assumptions: fees (1.25%/swap),
  slippage, and an MEV/sandwich penalty on entry.
- **Why:** A backtest can hide execution reality. Paper trading on *live* data for
  several weeks reveals whether the edge survives contact with fees, slippage, and
  the bots that will trade against you. If it doesn't survive paper trading, it
  won't survive real money.
- **Effort:** Medium.

### 3.2 Monitoring and alerting
- **What:** A dashboard or simple alert stream: tokens flagged, decisions made,
  running paper PnL, model calibration drift over time.
- **Why:** The edge decays. You need to *see* it decaying so you can retrain or stop,
  rather than discovering it via losses.
- **Effort:** Low–Medium.

---

## Phase 4 — Execution (only if Phases 1–3 prove an edge)

> Do not start this phase until paper trading shows a positive, fee-and-slippage-
> adjusted edge over weeks of live data. This ordering is the whole risk-management
> strategy.

### 4.1 Executor with MEV protection
- **What:** Wire a trade executor (Jupiter, PumpPortal trade API, or direct program
  calls) with slippage limits and, ideally, sandwich-resistant submission.
- **Why:** Entry and exit quality determines whether a real edge becomes real
  profit. MEV bots will try to sandwich you; unprotected execution can erase the
  edge entirely.

### 4.2 Position sizing and risk limits
- **What:** Fractional-Kelly or fixed-fraction sizing, a hard daily loss limit, and
  a per-token max. Size *down* — thin markets punish size.
- **Why:** Even a positive-edge strategy with bad sizing goes to zero on a losing
  streak. Survival first.

---

## The Rust question

**Short answer: not yet, and probably as a hybrid rather than a rewrite.**

Rust buys two things — raw throughput and memory-safe concurrency. Ask where this
project actually needs them:

- **Scoring engine:** trivial math, runs in microseconds in Python. Rewriting it is
  polishing the one part that was never slow.
- **The model:** Rust doesn't make a model more accurate. The hard, valuable work
  (Phases 1–2) is language-agnostic.
- **The ingestion firehose:** *this* is the one place Rust may earn its place. Pump.fun
  produces millions of trades/day; sustained tens per second, far more in bursts. If
  Python+asyncio can't keep up parsing that stream across all live tokens, that's a
  real, measurable wall.

### Decision rule
1. Stay in Python until you hit a *measured* performance wall (dropped messages,
   GC pauses hurting timing, can't process the full firehose).
2. When you hit it, move **only the hot path** — typically stream ingestion and
   transaction parsing — to Rust. Keep scoring and ML in Python.
3. The natural architecture is a **hybrid**: a Rust ingestion/parsing service
   feeding a Python scoring + ML layer, either as a separate process (via a queue)
   or through PyO3 bindings.
4. Only consider a deeper Rust footprint (e.g. a Geyser plugin for lowest-latency
   on-chain data) if you move toward latency-sensitive execution — which, for a
   *selectivity* strategy rather than a sniping one, you likely won't need.

### What NOT to do
- Don't rewrite the whole system in Rust up front. It's the classic way to spend
  three months building infrastructure for an edge you haven't yet proven exists.
- Don't let tooling choices become procrastination. The edge lives in data quality
  and the model, not in the language.

---

## Priority summary

| # | Upgrade | Phase | Priority | Status |
|---|---|---|---|---|
| 1 | Persistent logging | 1 | Critical | ✅ Done |
| 2 | RPC bot classifier | 1 | Critical | ✅ Done (0-RPC; premise falsified) |
| 3 | True reserve state | 1 | High | ✅ Done |
| 3b | Censoring / maturity guard | 1 | High | ✅ Done (2026-06-27) |
| 4 | Fitted graduation model | 2 | High | ✅ Done (vSol-detector; no skill) |
| 5 | New features (creator/holder/route) | 2 | High | ✅ Done (none earned a place) |
| 6 | Walk-forward backtest | 2 | Critical | ✅ Done → **no edge** |
| 7 | Paper-trading mode | 3 | Critical | Built; 6-week run not started |
| 8 | Monitoring/alerting | 3 | Medium | Built (`monitor.py` AlertStream) |
| 9 | Executor + MEV protection | 4 | Gated | 🔒 Unbuilt (Charter §10) |
| 10 | Position sizing / risk limits | 4 | Gated | 🔒 Unbuilt (Charter §10) |
| — | Rust hot-path (ingestion only) | as needed | Conditional | Not needed (no measured wall) |
| ★ | **Find a real at-fixed-vSol feature, or invoke kill (§6)** | 2→6 | **THE open question** | In progress |

---

## The one-paragraph version (updated 2026-06-27)

We did the plan. Logging, real reserve state, the bot classifier (at zero RPC),
the fitted+calibrated model, and the honest walk-forward backtest are all built.
The backtest did exactly the job it was for: it found that **there is no
demonstrated post-cost edge.** The model is a vSol-level detector; every
behavioural signal we tried — bot share, conviction, creator history, holder
concentration, and the correct per-program route taxonomy — fails to separate
graduates *at the same vSol*, and the one result that looked promising was a
right-censoring artifact (now guarded against). Per the charter that is a
*successful* answer, not a failure. The two honest moves now are: keep collecting
data until there's enough statistical power to either confirm a genuinely new
at-fixed-vSol feature or to invoke a kill criterion — and in the meantime build
nothing that increases trade frequency, capital, or speed. The hard part was
never the code; it was being willing to let the data say "no," and it did.

*This is an engineering plan, not financial advice. Memecoin trading is a
negative-sum game for most participants; treat any capital deployed as money you
can fully lose.*
