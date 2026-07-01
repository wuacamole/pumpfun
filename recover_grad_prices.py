"""Recover real PumpSwap graduation (exit) prices for graduated tokens that are
missing one — the forward-collection fix (Charter I2).

WHY THIS EXISTS
The live hook `live_monitor._record_graduation_price` fires *at the migrate
event*, when the PumpSwap pool has had no trades yet, so `_fetch_pumpswap_price`
finds no swap and silently returns None (its `except: pass`). Result: of 1,211
historical graduations only 1 logged a live price; the 65 others were filled by
an earlier (now-lost) recovery pass. Without real exit prices the backtest values
wins at a FLAT placeholder multiple, so no economic edge can be proven.

THE FIX
Run the *same* extraction the live hook uses, but LATER — against tokens that
graduated a while ago and therefore now have real PumpSwap swaps in their recent
signatures. The timing that dooms the live path is exactly what makes recovery
work. Idempotent (ON CONFLICT upsert), best-effort per token, rate-limited.

This is a backfill/repair tool, not a substitute for fixing the live path — but
it makes forward collection real today and can be re-run on a schedule so every
graduation eventually gets a true exit price.

Run (preview):  uv run python recover_grad_prices.py --limit 20
Run (apply):    uv run python recover_grad_prices.py --limit 500 --apply
Sweep all:      uv run python recover_grad_prices.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
import time

from dotenv import load_dotenv

from config import get_cached_constants
from helius import HeliusClient
from live_monitor import (
    _fetch_pumpswap_price,
    _price_is_plausible,
    _representative_entry_vsol,
)
from model import curve_price_per_token
from storage import Store

load_dotenv()

RECOVER_SOURCE = "pumpswap_first_swap_recovered"
RPC_SLEEP_SEC = 0.15  # gentle pacing so we don't trip Helius rate limits


def _unpriced_graduated(store: Store, limit: int | None) -> list[str]:
    """Mints that graduated but have no graduation_prices row, newest first."""
    sql = (
        "SELECT t.mint FROM tokens t "
        "LEFT JOIN graduation_prices g ON g.mint = t.mint "
        "WHERE t.outcome = 'graduated' AND g.mint IS NULL "
        "ORDER BY t.outcome_ts DESC"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [r[0] for r in store.conn.execute(sql).fetchall()]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="pumpfun.db")
    ap.add_argument("--limit", type=int, default=None,
                    help="max tokens to attempt (default: all unpriced)")
    ap.add_argument("--apply", action="store_true",
                    help="write recovered prices (default: preview only)")
    args = ap.parse_args()

    pc = get_cached_constants()
    store = Store(args.db)
    mints = _unpriced_graduated(store, args.limit)
    print(f"DB: {args.db}   graduated-but-unpriced: {len(mints)}"
          + (f" (capped at --limit {args.limit})" if args.limit else ""))
    if not mints:
        print("nothing to recover.")
        return

    helius = HeliusClient()
    rep_vsol = _representative_entry_vsol(pc)
    rep_entry_price = curve_price_per_token(rep_vsol, pc)

    found = 0
    written = 0
    no_swap = 0
    implausible = 0
    try:
        for i, mint in enumerate(mints, 1):
            try:
                price = await _fetch_pumpswap_price(mint, helius)
            except Exception:
                price = None
            await asyncio.sleep(RPC_SLEEP_SEC)
            if price is None:
                no_swap += 1
                continue
            if not _price_is_plausible(price, pc):
                implausible += 1
                continue
            found += 1
            mult = price / rep_entry_price if rep_entry_price > 0 else None
            if args.apply:
                grad_ts = store.conn.execute(
                    "SELECT outcome_ts FROM tokens WHERE mint=?", (mint,)
                ).fetchone()
                store.record_graduation_price(
                    mint=mint,
                    graduation_ts=float(grad_ts[0]) if grad_ts and grad_ts[0] else time.time(),
                    price_sol_per_unit=price,
                    entry_vsol=rep_vsol,
                    payoff_multiple=mult,
                    source=RECOVER_SOURCE,
                )
                written += 1
            if i % 25 == 0 or i == len(mints):
                print(f"  [{i}/{len(mints)}] priced={found} written={written} "
                      f"no_swap={no_swap} implausible={implausible}")
    finally:
        await helius.close() if hasattr(helius, "close") else None

    print(f"\ndone. attempted={len(mints)} priced={found} "
          f"({found/len(mints):.0%}) no_swap={no_swap} implausible={implausible}")
    if args.apply:
        total = store.conn.execute(
            "SELECT COUNT(*) FROM graduation_prices").fetchone()[0]
        print(f"written={written}   graduation_prices total now: {total}")
    else:
        print("[dry-run] no writes. Re-run with --apply to commit.")


if __name__ == "__main__":
    asyncio.run(main())
