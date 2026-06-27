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
| 11 | CRITICAL | `conviction_engine.py`, `live_monitor.py` | Negative `vsol` (reconstructed-vSol fallback drifting below 0 on a large sell) raised to the fractional power `** 1.6` produced a Python `complex`, crashing the probability clamp with `'<' not supported between 'float' and 'complex'` and killing the live monitor. Two-part fix: (a) floor the bonding-progress ratio at 0 before every power in `_breakeven_probability` / `_estimate_graduation_prob` (the crash); (b) floor the reconstruction in `live_monitor._vsol_for` at the virtual seed `v_start` — `vSol = virtual reserves (constant) + real reserves (≥0)` so it can never fall below `v_start`, keeping impossible values out of the logged training set at the source. |

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

## Duplicate-writer corruption fix (double-spawned monitor)

### Root cause

The harness launching `live_monitor.py` periodically double-spawns the
process. The existing `_acquire_singleton_lock` (`fcntl.flock` on
`/tmp/pumpfun_monitor_<hash>.lock`) guards one host, but the launcher
re-parented the second instance in a way that bypassed the lock check in
some restart paths. Two concurrent monitors each subscribe to the
logsSubscribe firehose and log every on-chain trade, so the `swaps` table
accumulated ~49% duplicate rows in recent windows. The corruption was
invisible to `dedupe_swaps` because that method keyed on
`(mint, ts, wallet, is_buy, sol_amount, vsol_after)` and the two monitors
stamped **different** `ts` values: `helius.py` set `ts = time.time()`
(receipt/dispatch time, since `logsSubscribe` carries no `blockTime`), so
the same on-chain event produced two rows with different `ts` and survived
the 6-col dedupe. Every per-token feature (`bot_trade_share`,
`sol_per_human_wallet`, `net_sol_per_swap`, `n_swaps`) was then inflated,
silently corrupting the training set.

### Fix — three layers, defense in depth

- **`helius.py` — on-chain block_time as `ts`.** `logsSubscribe` doesn't
  carry `blockTime`, but the decoded `TradeEvent` embeds the program's own
  unix timestamp (`block_time`). Both swap-emit paths (zero-RPC log-decode
  and the `HELIUS_CLASSIFY_TRADES=1` RPC path) now stamp the `Swap` with
  `float(te.block_time) when te.block_time else ts`. Two monitors logging
  the same on-chain trade now produce an identical `ts`, so the existing
  6-col dedupe collapses them. Falls back to receipt `time.time()` only
  when the event didn't decode (PumpPortal / RPC-backfill paths).
- **`storage.py` — `signature` column + partial unique index.** New
  nullable `signature` column on `swaps` (the on-chain tx signature from
  `logsSubscribe`). A partial unique index `ux_swaps_signature` rejects a
  second insert of the same signature at insert time — so even if two
  monitors somehow produce different `ts` (e.g. the block_time was None on
  one path), the second insert is a silent no-op. `log_swap` /
  `log_swaps` swallow the `IntegrityError` on a duplicate signature and
  return; other `IntegrityError`s still propagate. SQLite treats NULL
  signatures as distinct (no constraint), so signature-less rows
  (PumpPortal feed) still rely on the 6-col dedupe. Migration adds the
  column via `ALTER TABLE` and creates the index on existing DBs.
- **`storage.py` — `dedupe_swaps` two-pass + startup run.** `dedupe_swaps`
  now runs two passes inside one transaction: a signature pass first
  (collapses any same-signature rows that pre-date the insert-time guard),
  then the original 6-col event-key pass for signature-less rows. `Store`
  calls `dedupe_swaps()` on every open (`contextlib.suppress(
  OperationalError)` — a locked DB skips silently and heals on the
  writer's next restart), so the store self-heals legacy corruption
  without a manual maintenance step.

### Wiring

- `live_monitor.py:_handle_swap` accepts a `signature` kwarg and threads
  it to `store.log_swap`. The logs-feed swap path (`_, sw, mint, sig =
  evt`) and the gap-backfill path both pass the signature; the PumpPortal
  path (no signature in the event) leaves it `None`.

### Parquet

- `swaps_<ts>.parquet` now includes `is_direct`, `entry_program`, and
  `signature` columns (previously dropped). `_SWAPS_PARQUET_SCHEMA` and
  the streaming writer updated.

### Tests

- `test_store_signature_guard_blocks_duplicate_insert` — the same
  signature logged twice (different receipt `ts`) lands exactly once.
- `test_store_dedupe_swaps_collapses_same_on_chain_trade_across_receipt_ts`
  — two rows that agree on the 6-col key (incl. on-chain block_time) are
  collapsed; pins the `ts = block_time` contract.
- `test_store_signature_none_rows_fall_back_to_event_dedupe` — NULL-sig
  rows are not blocked by the partial index but are still caught by the
  6-col pass.
- `test_store_dedupe_keeps_distinct_same_ts_swaps` (existing) still
  passes — distinct genuine same-ts trades (different `vsol_after`) are
  preserved.

### Known limitation

Legacy rows written before this fix have `signature = NULL` (the column
didn't exist), so their receipt-time duplicates cannot be recovered via
the signature pass. The 6-col pass still collapses the ~0.16% byte-
identical dupes; the remaining receipt-time dupes in the legacy window
must be excluded at analysis time (export a snapshot from a clean
post-fix `ts` boundary). New data from this point forward is protected by
all three layers.

## Tier 1 backtest correctness fixes

### #1 — no-model fold no longer ENTERs on the anti-predictive heuristic

- **Root cause:** `run_backtest` fell back to `conviction_engine._estimate_graduation_prob`
  on fold 0 (no fitted model yet) and on any fold where `fit` threw.
  `validate_conviction.py` showed that heuristic is **anti-predictive**
  (AUC 0.366 — worse than random; ENTER hits 5.6% vs a 28.6% base rate, i.e.
  it systematically picks losers). The walk-forward was therefore seeding
  `train_buf` with wrong ENTER labels that poisoned fold 1's fit.
- **Fix:** the no-model fold now uses a **running base-rate prior**
  (`n_grad_so_far / n_resolved_so_far`, seeded at 0.5) as `est_grad_prob`.
  With a base-rate prior, `edge = est - brk ≈ 0`, so the ENTER gate (which
  requires `edge > entry_edge_threshold > 0`) can't fire. ENTER is also
  explicitly gated off when `model is None`. The no-model fold records
  WATCH/SKIP only. The Brier/calibration metrics are no longer inverted by
  the broken heuristic. The `_estimate_graduation_prob` import is removed
  from `backtest.py`.
- **Test:** `test_backtest_no_model_fold_records_no_enters`.

### #2 — real loss-side exit prices (charter I2)

- **Root cause:** `_realized_return` used a flat `default_loss_fraction=0.90`
  for every non-graduation — a fiction that violates charter I2 (post-cost
  realized PnL of the *actual* strategy) on the loss side. The paper broker
  logs the real exit economics in `paper_trades`/`paper_fills`: stopped
  trades average 0.40x entry, timed_out 0.96x. The backtest's
  `enter_mean_return` (the number the §5.4 gate judges) was computed from
  the 0.10x fiction, so it couldn't be compared to the broker's actual
  realized returns — losses were overstated by up to ~25x for timed-out
  trades.
- **Fix:** `_realized_return` takes a `loss_multiple` (the real
  `exit_sol / entry_sol` the broker logged at its stop/time-out) and uses
  `mult * (1 - exit_cost) - (1 + entry_cost)` — symmetric with the
  graduation path. Falls back to `default_loss_multiple=0.10` (the
  rug-floor) only when no logged exit exists. New
  `Store.stop_exit_multiples()` returns the per-mint real exit multiples;
  `run_backtest` accepts a `stop_exit_multiples` map (parallel to
  `graduation_prices`); `load_examples_from_store` returns a 3-tuple
  `(examples, graduation_prices, stop_exit_multiples)`; `cli.py backtest`
  threads it through.
- **Test:** `test_realized_return_uses_real_loss_multiple`,
  `test_store_stop_exit_multiples`.

### #3 — `Store(auto_dedupe=False)` for analytical opens

- **Root cause:** `Store.__init__` ran `dedupe_swaps()` on *every* open,
  including read-only analytical opens (`backtest.load_examples_from_store`,
  `base_rate`, `cli fit`/`export`). That triggered a full-table DELETE +
  `n_swaps` recompute (O(rows), ~500k) for no reason, and on a live DB it
  would mutate the writer's table out from under it.
- **Fix:** new `auto_dedupe: bool = True` kwarg. The writer
  (`live_monitor`) keeps the default so the store self-heals on restart.
  Analytical opens pass `auto_dedupe=False` and skip the dedupe + the
  `n_swaps` recompute entirely. Wired into `backtest.load_examples_from_store`,
  `base_rate.BaseRateMonitor.snapshot`, `cli.cmd_fit`, `cli.cmd_export`.
- **Test:** `test_store_auto_dedupe_false_skips_dedupe_on_open`.

## Tier 2 — model quality improvements

### #4 — creator-history index from existing data (Phase 2.2 wired)

- **Data:** `tokens.created_meta` already carries `traderPublicKey` (the
  creator) on all 25,618 tokens. 5,834 distinct creators; 1,609 have >1 token
  (max 639 by one creator). No new data source was needed — the join was
  always available, just not wired.
- **`storage.py`:** new `Store.creator_of(mint)` extracts the creator from
  `created_meta`; `Store.creator_history_index(as_of=)` builds a per-creator
  `(n_graduated, n_non_graduated)` index from resolved token outcomes, with
  no-lookahead (only outcomes with `outcome_ts <= as_of` count — a token that
  graduated at ts=200 doesn't contribute to a token scored at ts=100).
- **`backtest.py`:** `load_examples_from_store` now builds a
  `CreatorHistory(creator_graduated_count, creator_rugged_count)` per token
  from the creator's *prior* outcomes (before that token's first swap) and
  passes it through the examples list. `top_holder_share` and
  `smart_money_present` remain NaN (need a holder-snapshot source).
- **`live_monitor.py`:** builds a `creator_index` cache at startup, refreshed
  every 5 min on the `_gc` cycle; `_maybe_score` passes the mint's
  `CreatorHistory` into `predict_grad_prob` via a `creator_lookup` callback.
- **`storage.py:swaps_for`:** now reconstructs `is_direct` and
  `entry_program` on the `Swap` objects (previously dropped), so the
  backtest/refit paths get the new features.

### #5 — `direct_call_share` + `n_entry_programs` features (Phase 2.2a)

- **Root cause:** `is_direct` (the real instruction-stack bot flag — pump
  invoked at top level = autonomous bot) and `entry_program` (the raw
  top-level program driving the trade) are logged on every swap at zero RPC
  cost but were not in `FEATURE_COLUMNS`. `validate_conviction.py` showed the
  size-heuristic `via_frontend` proxy is **inverted vs reality** (humans
  trade small, bots graduate); `direct_call_share` is the signal the paper
  actually measured. Adding it is the single biggest unused-data leverage.
- **`model.py`:** `FEATURE_COLUMNS` extended with `direct_call_share` and
  `n_entry_programs` (appended at the end so existing saved models stay
  loadable — the imputer fills NaN for columns a model was trained without).
  `featurize` computes them from `Swap.is_direct` / `Swap.entry_program`;
  NaN when the feed didn't carry them (PumpPortal/RPC-backfill paths).

### #6 — time-ordered expanding-window CV in `refit_model.py`

- **Root cause:** `group_cv` used `GroupKFold` (no per-token leakage) but
  folds were random in time — a token from today could train a model that
  "predicts" a token from last week. The live system only ever predicts the
  future, so random-time OOS metrics over-estimate live performance. The
  saved `latest.pkl` was optimistically calibrated.
- **Fix:** new `time_ordered_cv` sorts mints by first-appearance rank
  (chronological), splits into `n_splits` contiguous time folds, and for
  fold `k` trains on folds `0..k-1` (earlier mints) testing on fold `k`.
  Fold 0 (no training data) is excluded — same contract as the walk-forward
  backtest's no-model fold. `main()` now reports both the random-GroupKFold
  (optimistic upper bound) and the time-ordered CV (honest metric) and
  picks the model on the time-ordered AUC.
- **`refit_model._load_tokens`:** now reads `is_direct` and `entry_program`
  from the DB so the new features populate in the refit frame.

### #7 — Platt (logistic) calibration below 1000 positives

- **Root cause:** `fit` used isotonic regression for calibration. Isotonic is
  non-parametric (a step function fit to the data) and overfits the
  calibration curve when positives are scarce — the in-sample Brier 0.0236
  is suspiciously low for a 28.6%-base-rate problem with 423 positives, a
  symptom of isotonic overfitting the calibration step.
- **Fix:** `fit` now chooses `method="sigmoid"` (Platt — parametric, 2
  parameters: slope + intercept) when `n_positives < 1000`, and
  `method="isotonic"` only once the positive class is large enough for the
  step function to be well-constrained. Charter §5.2 allows either method.

## Tier 3 — reproducibility & audit

### `--min-first-seen-ts` flag for the clean-window backtest

- **Problem:** the gate-relevant clean-window backtests (DECISIONS runs #6/#7
  — pre-dedup-fix contaminated tokens excluded) were run via ad-hoc in-memory
  filter scripts because `load_examples_from_store` only had a
  `holdout_start_ts` (which *excludes* recent tokens), not a post-fix *floor*.
  That violated CHARTER §9.2: the runs that decide the §5.4 gate weren't
  reproducible and weren't auto-logged/counted.
- **Fix:** added a `min_first_seen_ts` parameter to `load_examples_from_store`
  (excludes tokens whose first swap ts is below the floor) and a
  `--min-first-seen-ts` flag to `cli.py backtest`. The clean-window run is now
  a first-class CLI command that auto-logs to DECISIONS.md and increments
  `backtest_runs_to_date`. The floor is also recorded in the run config.
- Run #6 (cli, reproducible): n_tokens=1346, ENTER hit 32.1% vs 10.9% base
  (2.9x lift), +111.2% mean return, reliability slope 0.857 (now in the
  [0.8,1.2] band), bootstrap CI low +76.2%, jackknife positive, fold stable.
  4 of 5 §5.4 criteria pass; n_grad_in_enter=25 (<30) is the lone blocker.

### Dead-feature cleanup — `smart_money_present` dropped, `top_holder_share` measured

- **`smart_money_present`:** removed. It was 100% NaN (no data source) and a
  curated "profitable wallet" set would be target leakage if labelled from
  outcomes. Pure imputation noise + false completeness — gone from
  `FEATURE_COLUMNS`, `CreatorHistory`, and `featurize`.
- **`top_holder_share`:** was 100% NaN; now **reconstructed as-of from the swap
  stream** in `model.featurize` (zero new RPC) — per swap, token amount inferred
  from `sol_amount / curve_price_per_token(vsol_after)`, summed per wallet
  (buys add / sells subtract / negatives clamped), share = max / total. No
  lookahead (only `ts <= as_of`).
- **Measured, then excluded from the active set.** Univariate AUC 0.667
  (graduates less concentrated: median 0.44 vs 0.73). But a controlled A/B on
  identical data shows ~0 marginal OOS AUC (±0.005–0.009; slightly negative for
  gbm) — the signal is redundant with `n_unique_wallets` /
  `sol_per_human_wallet` / `bot_trade_share`. Per the "must prove OOS lift"
  rule it's kept computed (tested, documented) but not in `FEATURE_COLUMNS`.
- Model refit on the cleaned 15-feature set (auto-selected gbm). See DECISIONS.md.
