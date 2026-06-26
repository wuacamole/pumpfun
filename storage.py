"""
storage.py
==========
Persistent logging of every observed token's full trade history + final outcome.

This is Phase 1.3 of the upgrade roadmap — the single highest return-on-effort
item: data is the one thing you can't buy back later, and the edge decays, so
fresh data is what keeps the model alive.

Layout
------
SQLite (``pumpfun.db`` by default) is the live, append-only store. Two tables:

  * ``swaps``    — one row per observed swap event (the raw training data).
  * ``tokens``  — one row per token mint, with final outcome (graduated /
                   stalled / rugged) filled in when known.

Parquet snapshots (``snapshots/``) are produced by :func:`export_parquet` and
are the analytical format used by :mod:`model` and :mod:`backtest`. Single
source of truth is SQLite; Parquet is a derived, immutable snapshot.

The store is safe for concurrent append-from-one-process reads-from-another.
SQLite's WAL mode handles that. Don't share a write connection across processes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from conviction_engine import Swap

SCHEMA_SWAPS = """
CREATE TABLE IF NOT EXISTS swaps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mint        TEXT    NOT NULL,
    ts          REAL    NOT NULL,
    wallet      TEXT    NOT NULL,
    is_buy      INTEGER NOT NULL,            -- 1 buy, 0 sell
    sol_amount  REAL    NOT NULL,
    vsol_after  REAL    NOT NULL,
    via_frontend INTEGER NOT NULL,           -- 1 human (frontend-routed), 0 bot (direct)
    source      TEXT    NOT NULL DEFAULT 'pumpportal',  -- feed provenance
    ingested_at REAL    NOT NULL
);
-- Composite (mint, ts) is the hot path: swaps_for / the backtest / live
-- scoring all filter by mint then read in ts order. A single covering index
-- lets SQLite seek the mint and walk it pre-sorted instead of scanning the
-- mint index, scanning the ts index, and merging. ix_swaps_ts is kept for the
-- retention/prune scan (WHERE ts < cutoff across all mints); ix_swaps_mint is
-- redundant given the composite (its leftmost column is mint) and is dropped.
CREATE INDEX IF NOT EXISTS ix_swaps_mint_ts ON swaps(mint, ts);
CREATE INDEX IF NOT EXISTS ix_swaps_ts      ON swaps(ts);
DROP INDEX IF EXISTS ix_swaps_mint;
"""

SCHEMA_TOKENS = """
CREATE TABLE IF NOT EXISTS tokens (
    mint            TEXT PRIMARY KEY,
    first_seen_ts   REAL NOT NULL,
    last_seen_ts    REAL NOT NULL,
    peak_vsol       REAL NOT NULL DEFAULT 0,
    n_swaps         INTEGER NOT NULL DEFAULT 0,
    outcome         TEXT NOT NULL DEFAULT 'unknown',  -- graduated|stalled|rugged|unknown
    outcome_ts      REAL,
    graduated_at_ts REAL,
    created_meta    TEXT,                              -- JSON blob of the create event
    updated_at      REAL NOT NULL
);
"""

SCHEMA_PAPER = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mint            TEXT    NOT NULL,
    entry_ts        REAL    NOT NULL,
    entry_sol       REAL    NOT NULL,
    entry_cost_sol  REAL    NOT NULL,
    units_held      REAL    NOT NULL,
    entry_vsol      REAL    NOT NULL,
    exit_ts         REAL,
    exit_sol        REAL,
    exit_cost_sol   REAL,
    realized_return REAL,
    status          TEXT    NOT NULL DEFAULT 'open',   -- open|graduated|stopped|timed_out
    opened_at       REAL    NOT NULL,
    closed_at       REAL
);
CREATE INDEX IF NOT EXISTS ix_paper_mint   ON paper_trades(mint);
CREATE INDEX IF NOT EXISTS ix_paper_status ON paper_trades(status);
CREATE INDEX IF NOT EXISTS ix_paper_entry ON paper_trades(entry_ts);

CREATE TABLE IF NOT EXISTS paper_fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        INTEGER NOT NULL REFERENCES paper_trades(id),
    mint            TEXT    NOT NULL,
    side            TEXT    NOT NULL,                 -- buy|sell
    ts              REAL    NOT NULL,
    sol_amount      REAL    NOT NULL,
    effective_price REAL    NOT NULL,
    fee_sol         REAL    NOT NULL,
    slippage_sol    REAL    NOT NULL,
    mev_sol         REAL    NOT NULL,
    net_sol         REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_fills_trade ON paper_fills(trade_id);
CREATE INDEX IF NOT EXISTS ix_fills_mint  ON paper_fills(mint);

-- Key-value bag for the broker's in-flight state (bankroll, open positions
-- snapshot). Written as a single row that's updated on every fill; reloaded on
-- startup so a restart picks up where it left off. The full per-fill history
-- lives in paper_fills/paper_trades; this is just the fast-restore view.
CREATE TABLE IF NOT EXISTS paper_state (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

-- Post-graduation price observations. PumpSwap first-trade price for each
-- graduated token, so the backtest and paper broker can value the exit at the
-- *real* market price instead of a fictional flat 5x. Charter invariant I2
-- (post-cost realized PnL of the actual strategy) requires this — a payoff
-- model that assumes a constant multiple hasn't actually proven the edge.
-- Written once per token at graduation; the backtest reads it to resolve
-- realized returns. Falls back to the conservative 5x heuristic when no
-- observation is logged (the pre-data-collection state).
CREATE TABLE IF NOT EXISTS graduation_prices (
    mint              TEXT PRIMARY KEY,
    graduation_ts     REAL NOT NULL,
    price_sol_per_unit REAL NOT NULL,   -- PumpSwap first-trade price (SOL per token unit)
    entry_vsol        REAL,             -- vSol at the typical entry point (for ratio)
    payoff_multiple   REAL,             -- implied exit multiple vs a 1-SOL entry at entry_vsol
    source            TEXT NOT NULL DEFAULT 'pumpswap_first_trade',
    recorded_at       REAL NOT NULL
);
"""

# Outcome vocabulary. ``unknown`` means we haven't classified it yet.
OUTCOMES = {"graduated", "stalled", "rugged", "unknown"}


@dataclass
class TokenRow:
    mint: str
    first_seen_ts: float
    last_seen_ts: float
    peak_vsol: float
    n_swaps: int
    outcome: str
    outcome_ts: float | None
    graduated_at_ts: float | None
    created_meta: str | None
    updated_at: float


class Store:
    """SQLite-backed append-only log of swaps + per-token outcome.

    Open one instance per writer process. Readers (backtest/model) should open
    their own read-only connection or use :func:`export_parquet` snapshots.
    """

    def __init__(self, path: str | Path = "pumpfun.db") -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        # WAL = readers don't block the writer; sane for a single writer.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA_SWAPS)
        self.conn.executescript(SCHEMA_TOKENS)
        self.conn.executescript(SCHEMA_PAPER)
        # Truncate the WAL on open. Without an explicit checkpoint the WAL grows
        # unbounded under a long-lived writer (SQLite only auto-checkpoints when
        # the WAL crosses wal_autocheckpoint pages, and never truncates the file
        # back down), so a multi-day live_monitor leaves a multi-GB -wal behind.
        # Startup is the natural point: no readers are mid-query yet.
        self.checkpoint()

    def checkpoint(self) -> None:
        """Flush the WAL into the main db and truncate it back to zero.

        Safe to call any time; TRUNCATE blocks only if a reader is mid-read on
        an old snapshot. Call periodically (startup, and e.g. after each daily
        export) to keep ``pumpfun.db-wal`` from growing without bound.
        """
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # ------------------------------------------------------------------ swaps
    def log_swap(
        self,
        swap: Swap,
        mint: str,
        source: str = "pumpportal",
        ingested_at: float | None = None,
    ) -> None:
        """Append one swap. Idempotent-ish: we don't dedupe, so don't replay."""
        # Wrap both statements in a single transaction so a crash between them
        # doesn't leave the tokens table out of sync with the swaps table.
        self.conn.execute("BEGIN")
        try:
            self._log_swap_inner(swap, mint, source, ingested_at)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def _log_swap_inner(
        self,
        swap: Swap,
        mint: str,
        source: str = "pumpportal",
        ingested_at: float | None = None,
    ) -> None:
        """Insert a swap + update the token row. Caller manages the transaction.

        Split out from :meth:`log_swap` so :meth:`log_swaps` can wrap a batch
        in a *single* outer transaction without nesting BEGIN/COMMIT (SQLite
        raises "cannot start a transaction within a transaction").
        """
        self.conn.execute(
            """INSERT INTO swaps
                 (mint, ts, wallet, is_buy, sol_amount, vsol_after, via_frontend,
                  source, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                mint,
                swap.ts,
                swap.wallet,
                1 if swap.is_buy else 0,
                swap.sol_amount,
                swap.vsol_after,
                1 if swap.via_frontend else 0,
                source,
                ingested_at if ingested_at is not None else time.time(),
            ),
        )
        # Update the per-token summary row.
        now = time.time()
        self.conn.execute(
            """INSERT INTO tokens (mint, first_seen_ts, last_seen_ts, peak_vsol,
                                   n_swaps, outcome, updated_at)
               VALUES (?,?,?,?,?, 'unknown', ?)
               ON CONFLICT(mint) DO UPDATE SET
                 last_seen_ts = excluded.last_seen_ts,
                 peak_vsol     = MAX(tokens.peak_vsol, excluded.peak_vsol),
                 n_swaps       = tokens.n_swaps + 1,
                 updated_at    = excluded.updated_at""",
            (mint, swap.ts, swap.ts, swap.vsol_after, 1, now),
        )

    def log_swaps(self, mint: str, swaps: Iterable[Swap], source: str = "pumpportal") -> None:
        """Bulk append. Wraps the batch in a single transaction."""
        self.conn.execute("BEGIN")
        try:
            for s in swaps:
                self._log_swap_inner(s, mint, source)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def _recompute_n_swaps(self) -> None:
        """Re-derive ``tokens.n_swaps`` from the actual swap rows.

        ``n_swaps`` is a denormalized counter maintained incrementally in
        :meth:`_log_swap_inner`. Any path that *removes* rows from ``swaps``
        (dedupe, retention pruning, or a manual DELETE) desyncs it, which then
        skews every per-swap feature in the training set. Every delete path must
        call this so the counter stays the single derived truth. It's a full
        re-count, so it's O(rows) — cheap relative to the delete that triggered
        it, and only run on maintenance paths, never on the hot insert path.
        """
        self.conn.execute(
            """UPDATE tokens SET n_swaps = (
                 SELECT COUNT(*) FROM swaps WHERE swaps.mint = tokens.mint
               )"""
        )

    def dedupe_swaps(self) -> int:
        """Remove duplicate swap rows (same mint, ts, wallet, is_buy, sol_amount,
        vsol_after). Returns the number of rows deleted.

        Replays and backfills can insert the same swap twice (the live feed has
        no exactly-once guarantee). Duplicates silently inflate ``n_swaps`` and
        skew ``bot_trade_share`` / ``net_sol_per_swap`` / ``sol_per_human_wallet``
        — a correctness bug for the training set. Run this periodically (e.g. on
        startup) to keep the record honest. The first occurrence (lowest rowid)
        is kept; later duplicates are deleted.

        The key deliberately includes ``vsol_after``: two *genuinely distinct*
        trades by the same wallet at the same ``ts`` for the same ``sol_amount``
        still move the bonding curve, so their ``vsol_after`` differs and they
        are not collapsed. Only byte-identical event rows (i.e. a true replay of
        the same on-chain event) share all six columns. ``id``/``ingested_at``
        are excluded on purpose — they differ across replays of the same event,
        so including them would defeat the dedupe.
        """
        # DELETE + recompute must be atomic: a crash between them would leave
        # n_swaps stale — the exact drift this guards against. Same single-txn
        # pattern as log_swap/log_swaps.
        self.conn.execute("BEGIN")
        try:
            cur = self.conn.execute(
                """DELETE FROM swaps WHERE id NOT IN (
                     SELECT MIN(id) FROM swaps
                     GROUP BY mint, ts, wallet, is_buy, sol_amount, vsol_after
                   )"""
            )
            deleted = cur.rowcount or 0
            if deleted:
                self._recompute_n_swaps()
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return deleted

    def prune_swaps_before(self, cutoff_ts: float) -> int:
        """Delete swaps with ``ts < cutoff_ts`` and re-derive ``n_swaps``.

        The ``swaps`` table is append-only and grows without bound; past ~10M
        rows the Parquet export and every ``swaps_for`` scan slow down. Keep a
        rolling window in SQLite (the live/training-hot data) and let the older
        rows live only in the immutable Parquet snapshots, which are the
        analytical source anyway. **Export a snapshot covering the window first**
        (:meth:`export_parquet`) — pruning is destructive to SQLite. Returns the
        number of rows deleted; runs the WAL checkpoint so the freed pages don't
        sit in an ever-growing WAL after a large delete.
        """
        # DELETE + recompute atomic (see dedupe_swaps). The checkpoint runs
        # *after* COMMIT — wal_checkpoint can't run inside an open transaction.
        self.conn.execute("BEGIN")
        try:
            cur = self.conn.execute("DELETE FROM swaps WHERE ts < ?", (cutoff_ts,))
            deleted = cur.rowcount or 0
            if deleted:
                self._recompute_n_swaps()
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        if deleted:
            self.checkpoint()
        return deleted

    # --------------------------------------------------------------- tokens
    def register_token(self, mint: str, created_meta: dict[str, object] | None = None) -> None:
        """Create a token row at first sight (the 'create' event)."""
        meta_json = json.dumps(created_meta) if created_meta else None
        now = time.time()
        self.conn.execute(
            """INSERT OR IGNORE INTO tokens
                 (mint, first_seen_ts, last_seen_ts, peak_vsol, n_swaps,
                  outcome, created_meta, updated_at)
               VALUES (?,?,?,?,0,'unknown',?,?)""",
            (mint, now, now, 0.0, meta_json, now),
        )

    def set_outcome(
        self,
        mint: str,
        outcome: str,
        outcome_ts: float | None = None,
        graduated_at_ts: float | None = None,
    ) -> None:
        """Record the terminal state of a token. ``outcome`` must be in OUTCOMES."""
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        self.conn.execute(
            """UPDATE tokens
                 SET outcome = ?, outcome_ts = ?, graduated_at_ts = ?,
                     updated_at = ?
               WHERE mint = ?""",
            (outcome, outcome_ts, graduated_at_ts, time.time(), mint),
        )

    def get_token(self, mint: str) -> TokenRow | None:
        row = self.conn.execute("SELECT * FROM tokens WHERE mint = ?", (mint,)).fetchone()
        if row is None:
            return None
        return TokenRow(
            mint=row["mint"],
            first_seen_ts=row["first_seen_ts"],
            last_seen_ts=row["last_seen_ts"],
            peak_vsol=row["peak_vsol"],
            n_swaps=row["n_swaps"],
            outcome=row["outcome"],
            outcome_ts=row["outcome_ts"],
            graduated_at_ts=row["graduated_at_ts"],
            created_meta=row["created_meta"],
            updated_at=row["updated_at"],
        )

    def swaps_for(self, mint: str) -> list[Swap]:
        """Reconstruct Swap objects from the log. Used by backtest/replay."""
        rows = self.conn.execute(
            "SELECT * FROM swaps WHERE mint = ? ORDER BY ts ASC", (mint,)
        ).fetchall()
        return [
            Swap(
                ts=r["ts"],
                wallet=r["wallet"],
                is_buy=bool(r["is_buy"]),
                sol_amount=r["sol_amount"],
                vsol_after=r["vsol_after"],
                via_frontend=bool(r["via_frontend"]),
            )
            for r in rows
        ]

    def all_mints(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT mint FROM tokens ORDER BY first_seen_ts")]

    def labelled_mints(self) -> list[tuple[str, str]]:
        """Tokens whose outcome is known — the training set."""
        return [
            (r[0], r[1])
            for r in self.conn.execute(
                "SELECT mint, outcome FROM tokens WHERE outcome != 'unknown'"
            )
        ]

    # ------------------------------------------------------------ parquet
    def export_parquet(self, out_dir: str | Path = "snapshots") -> Path:
        """Snapshot the whole store to Parquet. Returns the snapshot path.

        Two files: ``swaps_<ts>.parquet`` and ``tokens_<ts>.parquet``.
        These are what :mod:`model` and :mod:`backtest` read from.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())

        swap_rows = self.conn.execute(
            "SELECT mint, ts, wallet, is_buy, sol_amount, vsol_after, via_frontend, "
            "source, ingested_at FROM swaps"
        ).fetchall()
        token_rows = self.conn.execute("SELECT * FROM tokens").fetchall()

        swaps_table = pa.Table.from_pylist(
            [
                {
                    "mint": r["mint"],
                    "ts": r["ts"],
                    "wallet": r["wallet"],
                    "is_buy": bool(r["is_buy"]),
                    "sol_amount": r["sol_amount"],
                    "vsol_after": r["vsol_after"],
                    "via_frontend": bool(r["via_frontend"]),
                    "source": r["source"],
                    "ingested_at": r["ingested_at"],
                }
                for r in swap_rows
            ],
            schema=pa.schema([
                ("mint", pa.string()),
                ("ts", pa.float64()),
                ("wallet", pa.string()),
                ("is_buy", pa.bool_()),
                ("sol_amount", pa.float64()),
                ("vsol_after", pa.float64()),
                ("via_frontend", pa.bool_()),
                ("source", pa.string()),
                ("ingested_at", pa.float64()),
            ]),
        )
        tokens_table = pa.Table.from_pylist(
            [
                {
                    "mint": r["mint"],
                    "first_seen_ts": r["first_seen_ts"],
                    "last_seen_ts": r["last_seen_ts"],
                    "peak_vsol": r["peak_vsol"],
                    "n_swaps": r["n_swaps"],
                    "outcome": r["outcome"],
                    "outcome_ts": r["outcome_ts"],
                    "graduated_at_ts": r["graduated_at_ts"],
                    "created_meta": r["created_meta"],
                    "updated_at": r["updated_at"],
                }
                for r in token_rows
            ],
            schema=pa.schema([
                ("mint", pa.string()),
                ("first_seen_ts", pa.float64()),
                ("last_seen_ts", pa.float64()),
                ("peak_vsol", pa.float64()),
                ("n_swaps", pa.int64()),
                ("outcome", pa.string()),
                ("outcome_ts", pa.float64()),
                ("graduated_at_ts", pa.float64()),
                ("created_meta", pa.string()),
                ("updated_at", pa.float64()),
            ]),
        )

        s_path = out_dir / f"swaps_{ts}.parquet"
        t_path = out_dir / f"tokens_{ts}.parquet"
        g_path = out_dir / f"graduation_prices_{ts}.parquet"
        pq.write_table(swaps_table, s_path)  # type: ignore[no-untyped-call]
        pq.write_table(tokens_table, t_path)  # type: ignore[no-untyped-call]

        # Graduation prices (post-graduation PumpSwap first-trade price).
        # Empty until tokens start graduating on the live feed; the backtest
        # falls back to the conservative 5x heuristic when no row is present.
        gp_rows = self.conn.execute("SELECT * FROM graduation_prices").fetchall()
        gp_table = pa.Table.from_pylist(
            [
                {
                    "mint": r["mint"],
                    "graduation_ts": r["graduation_ts"],
                    "price_sol_per_unit": r["price_sol_per_unit"],
                    "entry_vsol": r["entry_vsol"],
                    "payoff_multiple": r["payoff_multiple"],
                    "source": r["source"],
                    "recorded_at": r["recorded_at"],
                }
                for r in gp_rows
            ],
            schema=pa.schema([
                ("mint", pa.string()),
                ("graduation_ts", pa.float64()),
                ("price_sol_per_unit", pa.float64()),
                ("entry_vsol", pa.float64()),
                ("payoff_multiple", pa.float64()),
                ("source", pa.string()),
                ("recorded_at", pa.float64()),
            ]),
        )
        pq.write_table(gp_table, g_path)  # type: ignore[no-untyped-call]
        # A manifest with all three, so model/backtest can find the latest snapshot.
        # Write to a temp file then atomically rename to avoid a crash between
        # writing the parquet files and updating the manifest.
        manifest = out_dir / "latest.json"
        tmp = manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "swaps": str(s_path), "tokens": str(t_path),
            "graduation_prices": str(g_path), "ts": ts,
        }))
        tmp.rename(manifest)
        return s_path

    # ------------------------------------------------------------ paper trades
    def open_paper_trade(
        self,
        mint: str,
        entry_ts: float,
        entry_sol: float,
        entry_cost_sol: float,
        units_held: float,
        entry_vsol: float,
    ) -> int:
        """Insert a new open paper position. Returns its trade id."""
        now = time.time()
        c = self.conn.execute(
            """INSERT INTO paper_trades
                 (mint, entry_ts, entry_sol, entry_cost_sol, units_held, entry_vsol,
                  status, opened_at)
               VALUES (?,?,?,?,?,?, 'open', ?)""",
            (mint, entry_ts, entry_sol, entry_cost_sol, units_held, entry_vsol, now),
        )
        assert c.lastrowid is not None
        return int(c.lastrowid)

    def close_paper_trade(
        self,
        trade_id: int,
        exit_ts: float,
        exit_sol: float,
        exit_cost_sol: float,
        realized_return: float,
        status: str,
    ) -> None:
        """Mark a paper position closed with its exit economics."""
        self.conn.execute(
            """UPDATE paper_trades
                 SET exit_ts = ?, exit_sol = ?, exit_cost_sol = ?,
                     realized_return = ?, status = ?, closed_at = ?
               WHERE id = ?""",
            (exit_ts, exit_sol, exit_cost_sol, realized_return, status, time.time(), trade_id),
        )

    def log_paper_fill(
        self,
        trade_id: int,
        mint: str,
        side: str,
        ts: float,
        sol_amount: float,
        effective_price: float,
        fee_sol: float,
        slippage_sol: float,
        mev_sol: float,
        net_sol: float,
    ) -> None:
        self.conn.execute(
            """INSERT INTO paper_fills
                 (trade_id, mint, side, ts, sol_amount, effective_price,
                  fee_sol, slippage_sol, mev_sol, net_sol)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (trade_id, mint, side, ts, sol_amount, effective_price,
             fee_sol, slippage_sol, mev_sol, net_sol),
        )

    def open_paper_trade_ids(self) -> dict[str, int]:
        """mint -> trade_id for every position still open. Used on restart to
        reattach the broker to its in-flight positions."""
        return {
            r[0]: int(r[1])
            for r in self.conn.execute(
                "SELECT mint, id FROM paper_trades WHERE status = 'open'"
            )
        }

    def closed_paper_trades(self) -> list[dict[str, object]]:
        """All closed positions, newest first. For the dashboard/backtest."""
        rows = self.conn.execute(
            "SELECT * FROM paper_trades WHERE status != 'open' ORDER BY closed_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------- paper state (KV)
    def set_paper_state(self, key: str, value: object) -> None:
        """Upsert a JSON-serialisable value into the paper_state KV bag."""
        self.conn.execute(
            "INSERT INTO paper_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )

    def get_paper_state(self, key: str, default: object = None) -> object:
        row = self.conn.execute(
            "SELECT value FROM paper_state WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def close(self) -> None:
        self.conn.close()

    # -------------------------------------------------- graduation prices
    def record_graduation_price(
        self,
        mint: str,
        graduation_ts: float,
        price_sol_per_unit: float,
        entry_vsol: float | None = None,
        payoff_multiple: float | None = None,
        source: str = "pumpswap_first_trade",
    ) -> None:
        """Record the post-graduation PumpSwap price for a token.

        Called once when a token graduates (the migrate event). The backtest and
        paper broker use this to value exits at the real market price rather than
        a fictional flat multiple. Idempotent: re-recording overwrites.
        """
        self.conn.execute(
            """INSERT INTO graduation_prices
                 (mint, graduation_ts, price_sol_per_unit, entry_vsol,
                  payoff_multiple, source, recorded_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(mint) DO UPDATE SET
                 graduation_ts = excluded.graduation_ts,
                 price_sol_per_unit = excluded.price_sol_per_unit,
                 entry_vsol = excluded.entry_vsol,
                 payoff_multiple = excluded.payoff_multiple,
                 source = excluded.source,
                 recorded_at = excluded.recorded_at""",
            (
                mint, graduation_ts, price_sol_per_unit, entry_vsol,
                payoff_multiple, source, time.time(),
            ),
        )

    def get_graduation_price(self, mint: str) -> dict[str, object] | None:
        """Return the logged graduation price row for a mint, or None."""
        row = self.conn.execute(
            "SELECT * FROM graduation_prices WHERE mint = ?", (mint,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def all_graduation_prices(self) -> dict[str, dict[str, object]]:
        """All logged graduation prices, keyed by mint. For the backtest."""
        rows = self.conn.execute("SELECT * FROM graduation_prices").fetchall()
        return {r["mint"]: dict(r) for r in rows}


def load_latest_snapshot(snap_dir: str | Path = "snapshots") -> tuple[Path, Path, Path]:
    """Return (swaps_parquet, tokens_parquet, graduation_prices_parquet) for the
    latest snapshot. The graduation_prices path may point to a file that doesn't
    exist for very old snapshots (pre-Phase-2.3); callers should check.
    """
    snap_dir = Path(snap_dir)
    manifest = snap_dir / "latest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"no snapshot manifest at {manifest} — run store.export_parquet() first"
        )
    m = json.loads(manifest.read_text())
    sp = Path(m["swaps"])
    tp = Path(m["tokens"])
    gp = Path(m.get("graduation_prices", ""))
    return sp, tp, gp