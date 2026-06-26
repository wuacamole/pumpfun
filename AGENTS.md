# AGENTS.md

## Commands

- **Run tests:** `uv run pytest tests/ -q`
- **Lint:** `uv run ruff check .`
- **Typecheck:** `uv run mypy .`
- **Sanity check the engine:** `uv run python test_scenarios.py`
- **Live monitor (score + log):** `uv run python live_monitor.py`
- **Live monitor (paper trading + dashboard):** `uv run python live_monitor.py --paper --dashboard`
- **Fit graduation model on logged data:** `uv run python cli.py fit --kind gbm`
- **Walk-forward backtest on logged data:** `uv run python cli.py backtest`
- **Export SQLite -> Parquet snapshot:** `uv run python cli.py export --out-dir snapshots`

All three of `ruff check`, `mypy`, and `pytest` must pass after any change.

## Architecture

The system is a pipeline:

```
PumpPortal websocket ─> live_monitor ─> conviction_engine (score)
                                  ├──> storage        (SQLite + Parquet snapshot)
                                  ├──> helius         (real bot flag + on-chain reserves, optional)
                                  ├──> model          (fitted P(graduation), falls back to hand-built)
                                  ├──> paper_trade    (simulated broker with fees/slippage/MEV)
                                  └──> monitor        (rich dashboard + alerts.log)
```

- `conviction_engine.py` — pure logic, the paper's breakeven + hand-built grad-prob heuristic. Never touch I/O.
- `storage.py`           — SQLite (live) + Parquet export. The one source of truth. Single-writer WAL.
- `helius.py`            — Solana RPC client. Real `via_frontend` from instruction stack; real reserves from the bonding-curve account. Falls back gracefully when `HELIUS_API_KEY` is unset.
- `model.py`             — featurize (no-lookahead, `as_of`-bounded), fit (logistic / GBM), predict, save/load. Drop-in replacement for `_estimate_graduation_prob`.
- `backtest.py`          — walk-forward harness. Retrains on a rolling fold; first fold uses the hand-built heuristic (zero data). Reports hit rate, mean return (post fees/slippage/MEV), Brier.
- `paper_trade.py`       — sim broker. Fractional sizing, time stop, loss stop, graduation auto-sell.
- `monitor.py`           — Dashboard (rich) + AlertStream (ENTER prints, drawdown, calibration drift).
- `live_monitor.py`      — the runtime. Wires all of the above together. CLI flags for `--paper`, `--dashboard`, `--db`, `--model`.
- `cli.py`               — `backtest`, `fit`, `export` subcommands for the offline workflow.

## Constants that need verification before production

`helius.py` contains placeholder constants (`PUMP_PROGRAM_ID`, `PUMP_FRONTEND_PROGRAM_ID`, `_BONDING_CURVE_DISCRIMINATOR`, `BONDING_CURVE_DATA_LEN`) and `bonding_curve_pda` is intentionally a `NotImplementedError` stub. These must be confirmed against the current pump.fun program IDL (and a `solders`-based PDA derivation added) before the bot classifier and reserve reader can be trusted with capital. The fallback path (size heuristic + reconstructed vSol) keeps the system runnable in the meantime.

## Phase 2.2 join data (not yet wired)

The model exposes `CreatorHistory` (creator graduated/rugged counts, top holder share, smart-money presence) as optional NaN-tolerant features. They are emitted as NaN until you build the creator-history index from logged create events and a holder-snapshot source. Add the join in `backtest.load_examples_from_store` and the live monitor's score path once the data exists. The GBM imputer handles them gracefully until then (sklearn warns; that's expected).