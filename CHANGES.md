# Changelog

All notable changes to the pumpfun conviction scoring system.

## Phase 1 — Data Integrity & Fee Accuracy

### On-chain constant derivation (`config.py`)

- **New module**: `config.py` — date-keyed fee schedule, on-chain Global account decoder, bonding-curve decoder.
- `ProtocolConstants` dataclass: `initial_virtual_sol_reserves`, `v_sol_graduation`, `real_sol_target`, `fee_recipient`, `withdraw_authority`, `pool_migration_fee`, `creator_fee`, `enable_migration`.
- Graduation vSol derived via constant-product formula: `real_sol = ivsr * irtr / (ivtr - irtr)`. Live values: `v_sol_graduation=115.01 SOL`, `fee=1.00%`.
- `fee_at(ts)` — returns the `FeeSchedule` active at a given timestamp. Pre-2026-05-21 = 1.25% curve fee; post = 1.00% curve fee. PumpSwap always 25 bps.
- `load_protocol_constants(rpc_url)` — fetches the Global account (`4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf`) and populates the cache.
- `decode_global_account(data)` — packed struct unpacking (no alignment padding): `initialized(1) + authority(32) + fee_recipient(32) + 5×u64 + withdraw_auth(32) + enable_migrate(1) + pool_migration_fee(u64) + creator_fee(u64)`.
- `BondingCurve` dataclass with `quote_mint` field for USDC detection.
- `decode_bonding_curve(data)` — decodes bonding-curve account; `quote_mint` at offset 41 (32 bytes; all zeros = SOL-paired).
- `bonding_curve_pda(mint)` — PDA derivation using `solders` (`["bonding-curve", mint]`).
- `is_sol_paired(mint, rpc_url)` — one-shot bonding-curve read, cached in `NON_SOL_MINTS`.
- `get_cached_constants()` — returns live constants if loaded, else historical defaults (30/115/85).

### Fee schedule wiring

- `paper_trade.py` — per-fill `fee_at(ts)` for entry/exit; PumpSwap fee on graduated exits; derived `graduation_vsol()`.
- `backtest.py` — per-trade `fee_at(ts)` for date-aware cost; `entry_cost()` and `exit_cost_pumpswap()` methods.
- `conviction_engine.py` — module-level `__getattr__` (PEP 562) resolves `GRADUATION_VSOL`, `VIRTUAL_SOL_START`, `REAL_SOL_TARGET`, `FEE_RATE` from on-chain cache at runtime.

### USDC-denominated token detection

- `config.py:is_sol_paired(mint)` — reads bonding-curve `quote_mint`; returns `False` for USDC-paired tokens.
- `live_monitor.py:_is_sol_paired(mint)` — one-shot check, cached in `NON_SOL_MINTS`; excludes USDC pairs from scoring.
- `helius.py:decode_bonding_curve` — extracts `quote_mint` from raw account data.

### PumpSwap graduation detection

- `helius.py:PumpLogsFeed` — dual subscription: Pump program (`6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P`) + PumpSwap program (`pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA`).
- `("migrate", mint, ts)` event emitted on PumpSwap `create_pool` log instruction.
- `live_monitor.py` — handles migration events: marks outcome as `graduated`, triggers paper broker auto-sell.

### Helius bot classifier

- `helius.py:classify_via_frontend(tx)` — inspects instruction stack for `KNOWN_HUMAN_ROUTERS` (4 programs: `m9obQHAPyZeZ88w7XUY8Te6a8rBmhXsfKco7u8G8tnB`, `term9YPb9mzAsABaqN71A4xdbxHmpBNZavpBiQKZzN3`, `GMgnVFR8Jb39LoXsEVzb3DvBy3ywCmdmJquHUy1Lrkqb`, Jupiter).
- Graceful fallback: catches RPC errors → returns `None` → size heuristic.
- `parse_trade_tx(sig)` — extracts mint/trader/is_buy/sol_amount/via_frontend from token balance diffs.

### Data gap detection

- `helius.py:PumpLogsFeed` — slot-continuity tracking per `logsNotification.context.slot`; flags gaps > 50 slots.
- `PumpLogsFeed.gaps` — records `(last_slot, reconnect_slot)` tuples for audit.
- `PumpLogsFeed.backfill_mint(mint)` — best-effort `getSignaturesForAddress` recovery for open positions.
- `live_monitor.py` — on gap event, backfills open positions and replays missed swaps.

### Atomic writes

- `storage.py:log_swap` — wrapped two INSERTs (swaps + tokens) in explicit `BEGIN`/`COMMIT`/`ROLLBACK` transaction.
- `storage.py:export_parquet` — manifest written to temp file then atomically renamed.

---

## Phase 2 — Model & Backtest

### ML model (`model.py`)

- `featurize(swaps, as_of)` — no-lookahead feature extraction (all inputs strictly `≤ as_of`).
- Features: `vsol`, `real_sol_bonded`, `pct_to_graduation`, `bot_trade_share`, `sol_per_human_wallet`, `net_sol_per_swap`, `momentum`, `creator_graduated_count`, `creator_rugged_count`, `top_holder_share`, `smart_money_present`.
- `fit(df, kind)` — `GradientBoostingClassifier` with `SimpleImputer` pipeline (handles NaN creator features).
- `predict_grad_prob(fitted, swaps)` — returns calibrated probability.
- `FittedModel` / `ModelStore` — save/load via pickle with metadata.
- `CreatorHistory` dataclass — NaN-tolerant creator features (not yet wired to data).

### Walk-forward backtest (`backtest.py`)

- Rolling fold retrain: first fold uses hand-built heuristic (zero data), subsequent folds retrain on accumulated labeled data.
- Per-trade `fee_at(ts)` for date-aware cost deduction.
- OOS Brier score, hit rate, mean return (post fees/slippage/MEV), edge-over-time tracking.
- `BacktestReport` dataclass with `write_report()` producing valid JSON (NaN → None).

### Paper broker (`paper_trade.py`)

- Per-fill `fee_at(ts)` fee schedule (not static constants).
- Derived `graduation_vsol()` from on-chain cache.
- PumpSwap fee on graduated exits.
- SQLite persistence: `paper_trades`, `paper_fills`, `paper_state` (KV bag).
- `reload_state()` — restart-safe: reloads positions + bankroll from DB.
- `check_exits(time_stop, loss_stop)` — time stop and loss stop logic.
- `on_graduated(mint, ts)` — auto-sell at graduation.

---

## Phase 3 — Dashboard & Monitoring

### Dashboard (`monitor.py`)

- Rich `Live(screen=True)` with 1s `_tick` task.
- Watch list + open positions + recent decisions + recent closes tables.
- Feed stats: tokens seen, swaps/sec.
- `AlertStream` with quiet mode for dashboard.
- `write_metrics_snapshot()` — JSONL append for offline analysis.

### Live monitor (`live_monitor.py`)

- Two concurrent feeds: PumpPortal (creates) + Helius (logs for swaps + migration).
- USDC exclusion via `_is_sol_paired` (one-shot bonding curve read, cached).
- Per-ts fee via `fee_at(ts)` in vSol reconstruction.
- `load_protocol_constants(rpc_url)` on startup.
- Gap detection + backfill for open positions.
- Paper trading + dashboard integration.

### CLI (`cli.py`)

- `backtest` — walk-forward backtest on logged data.
- `fit` — fit graduation model (logistic/GBM).
- `export` — SQLite → Parquet snapshot.

---

## Bug Fixes (code review)

| # | Severity | File | Fix |
|---|----------|------|-----|
| 1 | CRITICAL | `conviction_engine.py` | `__getattr__` shadowed by module-level constants — removed assignments, added `_virtual_sol_start()`, `_graduation_vsol()`, `_real_sol_target()` accessors |
| 2 | CRITICAL | `storage.py` | `log_swap` two auto-commits — wrapped in `BEGIN`/`COMMIT`/`ROLLBACK` |
| 3 | HIGH | `helius.py` | `backfill_mint` invalid `minSlot` RPC param — removed |
| 4 | HIGH | `scripts/check_helius.py` | API key leaked in print — redacted |
| 5 | HIGH | `storage.py` | Non-atomic Parquet manifest — temp file + rename |
| 6 | HIGH | `config.py` | `BondingCurve.is_sol_paired` missed wrapped SOL — now checks both mints |
| 7 | HIGH | `backtest.py` | `suppress(Exception)` swallowed model fit failures — logged try/except |
| 8 | HIGH | `backtest.py` | `json.dumps` invalid JSON for NaN — `_safe()` helper |
| 9 | MEDIUM | `conviction_engine.py` | `score_token` missing seconds clamp — added `max(seconds, 1e-6)` |
| 10 | MEDIUM | `backtest.py` | First-fold heuristic `n_human=1` diverged from `featurize` — uses `0.0` fallback |
| 11 | CRITICAL | `conviction_engine.py` | Negative `vsol` (reconstructed-vSol fallback drifting below 0 on a large sell) raised to the fractional power `** 1.6` produced a Python `complex`, crashing the probability clamp with `'<' not supported between 'float' and 'complex'` and killing the live monitor. Floor the bonding-progress ratio at 0 before every power in `_breakeven_probability` / `_estimate_graduation_prob`. |

### Removed

- `main.py` — dead placeholder (unused by any module or CLI entry point).

---

## Storage durability & scale (code review)

`storage.py` / `cli.py` — addressing the SQLite hot-path, WAL growth, and retention findings.

- **Composite `(mint, ts)` index** — `ix_swaps_mint_ts` replaces the standalone `ix_swaps_mint`. The hot query (`swaps_for`, backtest, live scoring) filters by `mint` then reads in `ts` order; the composite lets SQLite seek + walk pre-sorted instead of merging two single-column index scans. The redundant `ix_swaps_mint` is dropped on open (its leftmost column is covered by the composite). `ix_swaps_ts` kept for the retention scan. Migration is automatic on next `Store(...)` open.
- **WAL checkpoint strategy** — `Store.checkpoint()` runs `PRAGMA wal_checkpoint(TRUNCATE)`; called on every open (startup) and after a prune. Without it the WAL grew unbounded under the long-lived `live_monitor` writer (live db had a 4 MB `-wal`; now truncated to 0).
- **`n_swaps` drift** — extracted `Store._recompute_n_swaps()`, the single re-derivation from actual rows. Every delete path (`dedupe_swaps`, new `prune_swaps_before`) now calls it, so the denormalized counter can't desync after a delete.
- **Dedupe key documented** — `dedupe_swaps` deliberately keys on `(mint, ts, wallet, is_buy, sol_amount, vsol_after)`: two genuinely distinct trades move the curve, so their `vsol_after` differs and they survive; only byte-identical replay rows collapse. `id`/`ingested_at` excluded by design. Added a test asserting distinct same-`ts` swaps are not collapsed.
- **Data retention** — `Store.prune_swaps_before(cutoff_ts)` deletes old swaps, re-derives `n_swaps`, and checkpoints. Wired into `cli.py export --prune-days N`: exports the full-history Parquet snapshot first (the analytical source of truth), then prunes SQLite to a rolling window so `swaps` doesn't grow without bound.
- **Atomic delete + recompute** — `dedupe_swaps` / `prune_swaps_before` wrap their `DELETE` and `_recompute_n_swaps()` in a single transaction (same pattern as `log_swap`); a crash between them previously left `n_swaps` stale. Prune's WAL checkpoint runs after `COMMIT` (can't checkpoint inside a transaction).
- **Streaming Parquet export** — `export_parquet` streams the (unbounded) `swaps` table through `ParquetWriter` in `_PARQUET_EXPORT_BATCH`-row pages (`Store._write_swaps_parquet`) instead of `fetchall` + one `from_pylist`, so peak memory is one batch regardless of table size. The bounded `tokens`/`graduation_prices` tables stay in-memory. Schema pulled out to `_SWAPS_PARQUET_SCHEMA`. Empty table still yields a valid readable file.
- **Periodic WAL checkpoint on long runs** — `live_monitor` calls `store.checkpoint()` every `WAL_CHECKPOINT_EVERY_SEC` (6 h), gated behind swap activity (only writes grow the WAL). Closes the gap left by the open-only checkpoint for a monitor that runs for days without restart.

---

## Dependencies added

- `solders` — PDA derivation for bonding-curve addresses.
- `python-dotenv` — `.env` loading for `HELIUS_API_KEY`.
- `pyarrow` — Parquet snapshot export.
- `pandas` — training frame construction.
- `scikit-learn` — GradientBoostingClassifier, imputer, metrics.
- `rich` — dashboard rendering.
- `websockets` — PumpPortal + Helius log feeds.

---

## Constants (verified from on-chain)

| Constant | Value | Source |
|----------|-------|--------|
| Pump program ID | `6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P` | Live `CreateV2` tx |
| PumpSwap program ID | `pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA` | Official SDK IDL |
| Global config account | `4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf` | On-chain |
| Initial virtual SOL | 30.0 | On-chain Global account |
| Real SOL graduation | 85.0054 | Derived: `30 * 85 / (115 - 85)` |
| vSol graduation | 115.0054 | On-chain + derived |
| Curve fee (post 2026-05-21) | 1.00% (95 bps protocol + 5 bps creator) | On-chain Global account |
| PumpSwap fee | 0.25% (20 + 5 bps) | Official SDK IDL |

---

## Test coverage (37 tests, all passing)

### Config decoders

| Test | What it covers |
|------|----------------|
| `test_decode_global_account_roundtrip` | Packed struct decode: 5 u64 fields + derived vSol/real-SOL graduation |
| `test_decode_global_account_too_short` | Rejects truncated data with `ValueError` |
| `test_decode_bonding_curve_sol_paired` | All-zeros quote_mint → `is_sol_paired=True`, reserves in SOL |
| `test_decode_bonding_curve_usdc_paired` | USDC pubkey as quote_mint → `is_sol_paired=False`, `graduated=True` |
| `test_decode_bonding_curve_wrapped_sol_paired` | Wrapped-SOL mint → `is_sol_paired=True` |
| `test_decode_bonding_curve_no_quote_mint` | Short body without V2 quote_mint → defaults to SOL-paired |
| `test_decode_bonding_curve_too_short` | Returns `None` for truncated data |

### Paper broker exits

| Test | What it covers |
|------|----------------|
| `test_paper_broker_time_stop` | `check_exits` fires `on_stop` with `status="timed_out"` after `time_stop_sec` |
| `test_paper_broker_loss_stop` | `check_exits` fires `on_stop` with `status="stopped"` when vSol drops >50% |
| `test_paper_broker_no_stop_within_bounds` | Position stays open within time + loss limits |
| `test_paper_broker_graduation_via_check_exits` | `check_exits` triggers `on_graduated` when vSol crosses threshold |
| `test_on_stop_floor_at_5pct` | `on_stop` floors exit value at 5% of entry for extreme rugs |
| `test_on_stop_returns_none_for_missing_mint` | `on_stop` returns `None` for unknown mint |

### Resolution logic (live_monitor)

| Test | What it covers |
|------|----------------|
| `test_resolve_vsol_from_event` | Uses `vSolInBondingCurve` directly when present |
| `test_resolve_vsol_fallback_to_history` | Reconstructs from `HISTORY` when no event field and no Helius |
| `test_resolve_via_frontend_fallback_heuristic` | Falls back to size heuristic (small→bot, large→human) |
| `test_resolve_via_frontend_source_based` | Uses `source` string (`pump.fun`→human, `api`→bot) |

### Store edge cases

| Test | What it covers |
|------|----------------|
| `test_store_duplicate_swap` | Same swap logged twice → no crash, both appended |
| `test_store_set_outcome_idempotent` | Setting outcome twice → updated, no crash |
| `test_store_empty_swaps_for` | `swaps_for` returns `[]` for unknown mint |

### Additional bug fix

- Fixed `decode_global_account` minimum length check: was double-counting `authority` + `fee_recipient` (226 bytes), corrected to 162 bytes to match actual layout.
