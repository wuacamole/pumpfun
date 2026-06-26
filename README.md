# Pump.fun Conviction System

A pipeline that scores pump.fun token launches in real time, paper-trades the
signals with realistic frictions, and gives you an honest, backtestable answer
to the only question that matters: **does the edge exist, or doesn't it?**

The system implements the findings from Marino, Naviglio, Tarantelli & Lillo,
*"Predicting the success of new crypto-tokens: the pump.fun case"*. The core
insight: graduation is driven by **fast accumulation via few human trades**
(conviction), not by volume — and **high bot share predicts failure**. A
buy-and-hold entry is only +EV when `P(graduation) > (vSol / 115)²`.

## What it does

```
Helius logsSubscribe (Pump program) ─┐
                                     ├──> live_monitor ─┐
PumpPortal websocket (token creates)┘                   │
                                                         ▼
                                              conviction_engine (score)
                                                         │
              ┌──────────────────────────────────────────┼───────────────────────────┐
              ▼                ▼                ▼          ▼                ▼          ▼
         storage          helius            model      paper_trade      monitor     (alerts.log)
         (SQLite)     (real bot flag    (fitted P(grad))  (sim broker    (dashboard)
                       + reserves)                          with fees)
```

- **Scores every trade** on the pump.fun bonding curve using the paper's
  breakeven formula + a graduation-probability estimate.
- **Logs everything** — every swap, every token, every paper fill — to SQLite,
  so you build a training set just by running it.
- **Paper-trades** ENTER signals with realistic frictions (1.25% fee + 2%
  slippage + 1% MEV penalty on entry) and tracks PnL across restarts.
- **Fits a model** on your logged data (logistic or gradient-boosted trees)
  to replace the hand-built graduation-probability heuristic.
- **Backtests** walk-forward: retrains on a rolling fold, first fold uses the
  hand-built heuristic (zero data), reports hit rate, mean return post-cost,
  and Brier score for calibration.
- **Dashboard** — a live `rich` panel showing bankroll, open positions, a
  per-token watch list, recent decisions, and recent closes.

## Why this design

The guiding principle from the [upgrade roadmap](UPGRADE_ROADMAP.md): **validate
that the edge exists before optimizing anything.** A fast system trading a
non-existent edge just loses money efficiently.

1. **Make the signal real first.** The bot flag (`via_frontend`) comes from
   the real on-chain instruction stack, not a size heuristic. Reserve state
   comes from the bonding-curve account, not reconstructed from trade history.
2. **Log everything now.** Data is the one thing you can't buy back later, and
   the edge decays, so fresh data is what keeps the model alive.
3. **Fit a model on your own data** with no-lookahead features, then backtest it
   honestly walk-forward. That step either finds an edge or saves you from a
   fake one.
4. **Paper-trade for weeks** on live data with fees and slippage included.
   Only then wire execution — and size small. Phase 4 (execution) is
   deliberately not built; it's gated on proving the edge first.

## Quick start

```bash
uv sync
cp .env.example .env   # add HELIUS_API_KEY=<your key>
uv run python live_monitor.py --paper --dashboard
```

Without a Helius key, the system runs on the free PumpPortal feed (token
creates only — no trades). With a Helius key, it subscribes to the Pump program
logs via `logsSubscribe` and receives every trade with the real bot flag parsed
from the instruction stack.

## Commands

| Command | Purpose |
|---|---|
| `uv run python live_monitor.py` | Score + log only (no paper trading) |
| `uv run python live_monitor.py --paper --dashboard` | Paper trade with live dashboard |
| `uv run python cli.py fit --kind gbm` | Fit graduation model on logged data |
| `uv run python cli.py backtest` | Walk-forward backtest on logged data |
| `uv run python cli.py export --out-dir snapshots` | SQLite → Parquet snapshot |
| `uv run python scripts/check_helius.py` | Verify Helius RPC connectivity |
| `uv run python test_scenarios.py` | Sanity check the scoring engine |

Quality gates (must all pass after any change):
```bash
uv run ruff check .
uv run mypy .
uv run pytest tests/ -q
```

## Data storage

Everything lives in the project directory and is gitignored:

| Path | What | Format |
|---|---|---|
| `pumpfun.db` | Live store: every swap + per-token outcome + paper trades/fills/state | SQLite (WAL) |
| `snapshots/*.parquet` | Analytical snapshots for model fitting / backtest | Parquet |
| `models/*.pkl` | Fitted graduation models | pickle + JSON |
| `alerts.log` | Alert stream (ENTER, closes, drawdown, calibration drift) | text |

Paper-trade state (bankroll, open positions) is snapshotted to SQLite on every
fill and reloaded on restart, so a crash never wipes your PnL track record.

## Architecture

| File | Role |
|---|---|
| `conviction_engine.py` | Pure scoring logic — breakeven + hand-built grad-prob heuristic. No I/O. |
| `storage.py` | SQLite (live) + Parquet export. The one source of truth. |
| `helius.py` | Solana RPC client: real `via_frontend` from instruction stack, real reserves from the bonding-curve account, `PumpLogsFeed` streaming trades via `logsSubscribe`. |
| `model.py` | No-lookahead featurize, fit (logistic/GBM), predict, save/load. Drop-in replacement for the hand-built heuristic. |
| `backtest.py` | Walk-forward harness: rolling retrain, first fold uses the heuristic, reports hit rate / mean return (post-cost) / Brier. |
| `paper_trade.py` | Sim broker: fractional sizing, time/loss stops, graduation auto-sell, SQLite-persisted. |
| `monitor.py` | `rich` Dashboard + AlertStream (ENTER prints, drawdown, calibration drift). |
| `live_monitor.py` | The runtime — wires all of the above together. CLI flags for `--paper`, `--dashboard`, `--db`, `--model`. |
| `cli.py` | `backtest`, `fit`, `export` subcommands for the offline workflow. |

## Constants that need verification before production

`helius.py` contains the Pump program ID (`6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P`,
verified from a live `CreateV2` transaction), a set of known human-router program IDs
(built from observed trades), and the bonding-curve account layout. The PDA derivation
is a `NotImplementedError` stub. Confirm these against the current pump.fun program IDL
and add a `solders`-based PDA derivation before trusting the reserve reader with capital.
The fallback path (reconstructed vSol from trade history) keeps the system runnable.

## Phase 2.2 join data (not yet wired)

The model exposes `CreatorHistory` (creator graduated/rugged counts, top holder
share, smart-money presence) as optional NaN-tolerant features. They are emitted
as NaN until you build the creator-history index from logged create events and a
holder-snapshot source. The GBM imputer handles them gracefully until then.

## The Rust question

Not yet, and probably as a hybrid rather than a rewrite. The hot path is the
websocket + per-tx `getTransaction` fetch, which is I/O-bound (network
latency), not CPU-bound. Python+asyncio handles pump.fun's volume fine. If you
later measure dropped log events or RPC fetch latency hurting classification,
move only the logs-feed parser to Rust — keep scoring and ML in Python. See the
[Rust section of the roadmap](UPGRADE_ROADMAP.md#the-rust-question).

## Disclaimer

This is an engineering project, not financial advice. Memecoin trading is a
negative-sum game for most participants; treat any capital deployed as money
you can fully lose. The system is designed to tell you whether to *not* trade —
most of the value is in being selective enough to skip the 99% of tokens that
are designed to take your money.