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

## Where the prototype stands today

| Component | Status | Honest limitation |
|---|---|---|
| Conviction scoring math | Working, tested | Sound — implements the paper directly |
| Breakeven check `p > (vSol/115)²` | Working | Exact, from the paper |
| Bot classification (`isBot`) | **Approximated** | Uses a size heuristic, not the real method |
| Graduation probability | **Hand-built weights** | Not fitted to data; qualitative only |
| Live data feed | Working (PumpPortal) | No reserve state, no real bot flag, higher latency |
| Execution | **None** | Scores only — deliberately |

The two **bold** items are the difference between "a plausible-looking demo" and
"a tool measuring what the researchers actually measured." They come first.

---

## Phase 1 — Make the signal real (do this first)

Goal: every input the model sees should be the *true* value, not an approximation.
Until this is done, no backtest or PnL number means anything.

### 1.1 RPC-based bot classifier — **highest priority**
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

### 1.2 True bonding-curve reserve state
- **What:** Read `vSolInBondingCurve` / real reserves directly from the feed or
  on-chain account state, instead of reconstructing it from running trade history.
- **Why:** Every downstream number — % to graduation, breakeven, conviction — is a
  function of vSol. Drift in the reconstructed value silently biases every decision.
- **Effort:** Low–Medium. Bitquery exposes this directly; raw RPC requires decoding
  the bonding-curve account.

### 1.3 Persistent logging of everything
- **What:** Write every observed token's full trade history + final outcome
  (graduated / stalled / rugged) to a local store (SQLite or Parquet files).
- **Why:** This is your training set. Without it you can never fit a real model or
  run an honest backtest. Start collecting *now* — data is the one thing you can't
  buy back later, and the edge decays, so fresh data is what keeps the model alive.
- **Effort:** Low. Highest return-on-effort item in the whole roadmap.

---

## Phase 2 — Replace guesswork with a fitted model

Goal: turn the hand-built `_estimate_graduation_prob` into something calibrated on
your own data.

### 2.1 Fit a real graduation model
- **What:** On the logged data, fit a logistic regression (interpretable, fast) or
  gradient-boosted trees (more accurate) predicting graduation from: vSol,
  bot_share, sol_per_human_wallet, net_sol_per_swap, momentum, and new features below.
- **Why:** The current weights encode the paper's *direction* but not the right
  *magnitudes* for current market conditions. A fitted model tells you the actual
  probability — which is what the breakeven comparison needs to be meaningful.
- **Effort:** Medium. The modeling is easy; the discipline (time-consistent
  features, no lookahead) is the hard part.

### 2.2 New predictive features
- **Creator wallet history:** Has this creator launched graduating tokens before?
  Prolific serial-ruggers are detectable on-chain. The paper studied this directly.
- **Successful-trader presence:** Is a wallet with a strong recent track record
  buying early? Modest but real signal.
- **Early-holder concentration:** What fraction of supply does the top holder /
  creator control? High concentration = dump risk.
- **Why:** These are the conditioning variables that lift the probability curve
  *above* breakeven. Raw vSol alone is provably below it.

### 2.3 Walk-forward backtest harness
- **What:** Replay logged history chronologically, score each token using only
  information available at that moment, and measure: hit rate of ENTER calls,
  realized return per call, and whether the model's claimed probabilities are
  *calibrated* (when it says 40%, do ~40% actually graduate?).
- **Why:** This is the moment of truth. It either shows a real edge or it doesn't.
  Either answer saves you money — one tells you to proceed, the other tells you to
  stop before risking capital.
- **Effort:** Medium–High. The single most important deliverable in this document.

---

## Phase 3 — Paper trade before anything touches money

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

| # | Upgrade | Phase | Priority | Effort |
|---|---|---|---|---|
| 1 | Persistent logging | 1 | Critical | Low |
| 2 | RPC bot classifier | 1 | Critical | Medium |
| 3 | True reserve state | 1 | High | Low–Med |
| 4 | Fitted graduation model | 2 | High | Medium |
| 5 | New features (creator/holder) | 2 | High | Medium |
| 6 | Walk-forward backtest | 2 | Critical | Med–High |
| 7 | Paper-trading mode | 3 | Critical | Medium |
| 8 | Monitoring/alerting | 3 | Medium | Low–Med |
| 9 | Executor + MEV protection | 4 | Gated | High |
| 10 | Position sizing / risk limits | 4 | Gated | Medium |
| — | Rust hot-path (ingestion only) | as needed | Conditional | High |

---

## The one-paragraph version

Log everything now. Fix the bot classifier and reserve state so your inputs are
real. Fit a model on the logged data and backtest it honestly — that step either
finds an edge or saves you from a fake one. Paper-trade for weeks against live
data with fees and slippage included. Only then wire execution, and size small.
Stay in Python; reach for Rust only when you measure a real throughput wall, and
only for the ingestion hot path. The hard part was never the code — it's proving
the edge is real and surviving fees, MEV, and decay. Most of the value is in being
selective enough to skip the 99% of tokens that are designed to take your money.

*This is an engineering plan, not financial advice. Memecoin trading is a
negative-sum game for most participants; treat any capital deployed as money you
can fully lose.*
