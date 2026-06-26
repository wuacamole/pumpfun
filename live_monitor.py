"""
live_monitor.py
===============
Live Pump.fun swap feed -> conviction scoring -> paper trading + logging.

This is the integrated runtime that wires together every Phase 1–3 component:

  * storage.Store          — append every swap + token outcome to SQLite
  * helius.HeliusClient    — real bot classifier (instruction-stack) + true
                             reserves; falls back to the size heuristic when
                             no API key / RPC unavailable
  * model.FittedModel      — graduation probability from a fitted model if one
                             is on disk; else the hand-built heuristic
  * paper_trade.PaperBroker — simulated entries/exits with fees+slippage+MEV
  * monitor.Dashboard / AlertStream — live terminal + alerts.log

Run modes
---------
  python live_monitor.py             # score + log only (default)
  python live_monitor.py --paper     # also paper-trade with realistic frictions
  python live_monitor.py --paper --dashboard

Env
---
  HELIUS_API_KEY   — Helius key. Without it, we use the size-heuristic bot flag
                     and reconstruct vSol from trade history (Phase 1 fallbacks).
  PUMPFUN_DB       — SQLite path (default pumpfun.db)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # reads .env in cwd; harmless if absent

from rich.live import Live  # noqa: E402

from config import fee_at, get_cached_constants  # noqa: E402
from conviction_engine import (  # noqa: E402
    Swap,
    TokenScore,
    _breakeven_probability,
    score_token,
)
from helius import HeliusClient, PumpLogsFeed  # noqa: E402
from model import FittedModel, ModelStore, predict_grad_prob  # noqa: E402
from monitor import AlertStream, Dashboard, DecisionEvent, OpenPositionView, PnlEvent  # noqa: E402
from paper_trade import PaperBroker  # noqa: E402
from storage import Store  # noqa: E402

try:
    import websockets
except ImportError:
    websockets = None  # type: ignore[assignment]

PUMPPORTAL_WS = "wss://pumpportal.fun/api/data"

# Per-token rolling swap history. In production cap memory: drop tokens that
# stalled (>15 min, low SOL) and tokens that already graduated.
HISTORY: dict[str, list[Swap]] = defaultdict(list)
LAUNCH_TS: dict[str, float] = {}
LAST_SCORE_AT: dict[str, float] = {}
# Cached bonding-curve PDA -> reserves address (so we don't re-derive per swap).
BC_ADDR: dict[str, str] = {}
# Token mints we've subscribed to trade updates for (so we can unsubscribe
# when GC drops them — PumpPortal bans clients that open many connections,
# and per-token subscriptions are metered, so keep the set bounded).
SUBSCRIBED: set[str] = set()

# Tunables
MIN_SWAPS_TO_SCORE = 8
RESCORE_EVERY_SEC  = 2.0
STALE_AFTER_SEC    = 900
PRINT_DECISIONS    = {"ENTER", "WATCH"}
# Cap on concurrent per-token trade subscriptions. PumpPortal charges
# 0.01 SOL per 10k events; dropping stale subscriptions keeps cost bounded.
MAX_SUBSCRIPTIONS = 200
# Cap on the in-memory per-token swap buffer. When exceeded, the oldest tokens
# are evicted by _gc(force=True) so the buffer can't grow unbounded under load
# (a memory leak that drops the feed is a correctness bug, not just speed).
MAX_TOKENS = 1000


def _approx_via_frontend(evt: dict[str, Any]) -> bool:
    """Fallback bot flag — used when no RPC key is configured.

    See helius.classify_via_frontend for the real method.
    """
    src = (evt.get("source") or evt.get("txSource") or "").lower()
    if "pump.fun" in src or "frontend" in src or "website" in src:
        return True
    if "bot" in src or "api" in src or "direct" in src:
        return False
    sol = abs(float(evt.get("solAmount", evt.get("sol_amount", 0)) or 0))
    return sol >= 0.05


async def _resolve_via_frontend(evt: dict[str, Any], helius: HeliusClient | None) -> bool:
    """Phase 1.1 — real bot flag via tx instruction stack if Helius is configured,
    else the size heuristic. The signature is read from the event; if absent we
    fall back. Errors fall back too — the feed should never block on RPC."""
    sig = evt.get("signature") or evt.get("txSignature")
    if helius and sig:
        try:
            v = await helius.classify_via_frontend(sig)
            if v is not None:
                return v
        except Exception:
            pass
    return _approx_via_frontend(evt)


async def _resolve_vsol(
    evt: dict[str, Any],
    mint: str,
    helius: HeliusClient | None,
    ts: float | None = None,
) -> float:
    """Phase 1.2 — true reserve state from the bonding-curve account if Helius
    is configured, else reconstruct from running trade history.

    ``ts`` is the swap's own timestamp (the on-chain block time when available,
    else wall-clock now). The reconstruction path uses ``fee_at(ts)`` so a
    backfilled swap from a prior fee regime is costed at the schedule that was
    live *then*, not the current one. Charter invariant I5 (date-keyed costs)
    applies to the reconstruction path too.
    """
    vsol = evt.get("vSolInBondingCurve") or evt.get("vsol_after")
    if vsol is not None:
        return float(vsol)
    # Try on-chain read (only if we know the bonding-curve address for this mint).
    if helius and mint in BC_ADDR:
        try:
            r = await helius.read_reserves(BC_ADDR[mint])
            if r is not None:
                return r.vsol
        except Exception:
            pass
    # Fallback: reconstruct from history.
    prev = HISTORY[mint][-1].vsol_after if HISTORY[mint] else get_cached_constants().virtual_sol_start
    txtype = (evt.get("txType") or evt.get("type") or "").lower()
    sol = abs(float(evt.get("solAmount", evt.get("sol_amount", 0)) or 0))
    # Use the swap's own ts for the fee schedule, not wall-clock now — a
    # backfilled swap from a prior fee regime must be costed at that regime.
    fee_rate = fee_at(ts if ts is not None else time.time()).curve_fee_rate
    return prev + sol * (1 - fee_rate) * (1 if txtype == "buy" else -1)


async def _to_swap(evt: dict[str, Any], mint: str, helius: HeliusClient | None) -> Swap | None:
    txtype = (evt.get("txType") or evt.get("type") or "").lower()
    if txtype not in ("buy", "sell"):
        return None
    sol = abs(float(evt.get("solAmount", evt.get("sol_amount", 0)) or 0))
    # Stamp the swap with the on-chain block time when available so temporal
    # order is preserved; fall back to wall-clock now for the live-feed path.
    ts = float(evt["blockTime"]) if evt.get("blockTime") else time.time()
    via_frontend = await _resolve_via_frontend(evt, helius)
    vsol = await _resolve_vsol(evt, mint, helius, ts=ts)
    return Swap(
        ts=ts,
        wallet=evt.get("traderPublicKey") or evt.get("trader") or evt.get("wallet") or "?",
        is_buy=(txtype == "buy"),
        sol_amount=sol,
        vsol_after=vsol,
        via_frontend=via_frontend,
    )


def _broker_graduation_price(mint: str, store: Store) -> tuple[float, float] | None:
    """Lookup hook for PaperBroker.on_graduated: return (price, payoff_multiple)
    from the logged graduation_prices table, or None to fall back to the default.

    The multiple is derived from the bonding-curve terminal price: at graduation
    the bonding-curve's vSol is ~115, so a 1-SOL entry at vSol V buys
    (1 / price_at_v) units; the exit value at the PumpSwap price P is
    (units * P) = P / price_at_v. We approximate price_at_v as the vSol-ratio
    implied entry price (entry_vsol / total_supply_at_graduation), which for the
    flat default case yields ~5x. When a logged entry_vsol is present we compute
    the per-position multiple exactly; otherwise we return the stored
    payoff_multiple directly.
    """
    row = store.get_graduation_price(mint)
    if row is None:
        return None
    price = row.get("price_sol_per_unit")
    pm = row.get("payoff_multiple")
    if isinstance(price, (int, float)) and price > 0:
        # If we have a logged payoff_multiple, use it; else return the price
        # and let the broker compute a rough multiple from the default 5x.
        if isinstance(pm, (int, float)) and pm > 0:
            return float(price), float(pm)
        return float(price), 5.0
    return None


async def _record_graduation_price(
    mint: str,
    graduation_ts: float,
    helius: HeliusClient | None,
    store: Store,
) -> None:
    """Best-effort: fetch the PumpSwap first-trade price for a freshly
    graduated token and log it to SQLite. The backtest and paper broker use
    this to value exits at the real market price (charter I2) instead of a
    flat 5x placeholder. Non-blocking on failure — the broker falls back.
    """
    if helius is None:
        return
    try:
        price = await _fetch_pumpswap_first_price(mint, helius)
        if price is None:
            return
        # The payoff multiple vs a 1-SOL entry at the bonding-curve terminal
        # price. We don't know each position's exact entry vSol here; the
        # backtest computes per-trade multiples from the entry_vsol column.
        # For the live paper broker, a single representative multiple is fine
        # since all open positions in the same token exit at the same price.
        store.record_graduation_price(
            mint=mint,
            graduation_ts=graduation_ts,
            price_sol_per_unit=price,
        )
    except Exception:
        pass  # never block graduation handling on a price fetch failure


async def _fetch_pumpswap_first_price(mint: str, helius: HeliusClient) -> float | None:
    """Fetch the first PumpSwap swap price for a graduated token (SOL per unit).

    Uses getSignaturesForAddress on the PumpSwap pool account to find the first
    swap, then reads the token/SOL balance deltas from that tx. Returns None if
    the pool or first swap can't be found. This is a best-effort price discovery
    — the bonding-curve terminal price is the fallback when this fails.
    """
    # The PumpSwap pool PDA is derived from the base mint. We don't have the
    # exact PDA seeds here without the PumpSwap IDL, so fall back to scanning
    # the token's recent signatures for a PumpSwap swap instruction.
    try:
        sigs = await helius._rpc(
            "getSignaturesForAddress", [mint, {"limit": 5}]
        )
        for s in (sigs or []):
            if s.get("err"):
                continue
            tx = await helius.get_transaction(s["signature"])
            if not tx or "transaction" not in tx:
                continue
            # Look for a PumpSwap swap: token balance delta on the pool + SOL
            # delta. The first swap after graduation gives the market price.
            meta = tx.get("meta", {}) or {}
            post_bal = meta.get("postTokenBalances") or []
            pre_bal = meta.get("preTokenBalances") or []
            for pb in post_bal:
                if pb.get("mint") != mint:
                    continue
                idx = pb.get("accountIndex")
                pre_match = next((b for b in pre_bal if b.get("accountIndex") == idx), None)
                pre_amt = (pre_match or {}).get("uiTokenAmount", {}).get("uiAmount") or 0.0
                post_amt = pb.get("uiTokenAmount", {}).get("uiAmount") or 0.0
                token_delta = abs(post_amt - pre_amt)
                if token_delta <= 0:
                    continue
                pre_sol = meta.get("preBalances") or []
                post_sol = meta.get("postBalances") or []
                if not pre_sol or not post_sol:
                    continue
                sol_delta = abs(post_sol[0] - pre_sol[0]) / 1e9
                if sol_delta <= 0:
                    continue
                price = sol_delta / token_delta
                return float(price) if price > 0 else None
    except Exception:
        return None
    return None


def _gc(now: float, store: Store | None = None, force: bool = False) -> list[str]:
    """Forget stalled/graduated tokens. Returns the dropped mints so the caller
    can unsubscribe them from PumpPortal to bound metered cost.

    If ``store`` is provided, graduated tokens have their outcome recorded before
    being dropped — a safety net for when the PumpPortal "migrate" event is missed.
    Note: the outcome is only set to "graduated" when the vSol came from an
    on-chain reserves read; a *reconstructed* vSol crossing the threshold is
    marked "unknown" (charter I3 — the label must be the on-chain truth, not a
    heuristic that can drift on a fee-rate mismatch).

    ``force``: when True, also evict the oldest stalled tokens to bound memory
    even if they haven't hit STALE_AFTER_SEC yet — called when len(HISTORY) >
    MAX_TOKENS so the in-memory buffer can't grow unbounded under load.
    """
    dropped: list[str] = []
    grad_vsol = get_cached_constants().v_sol_graduation
    # If over the memory cap, evict the oldest tokens first (by last swap ts).
    if force and len(HISTORY) > MAX_TOKENS:
        sorted_mints = sorted(HISTORY.keys(), key=lambda m: HISTORY[m][-1].ts if HISTORY[m] else 0)
        for mint in sorted_mints[: len(HISTORY) - MAX_TOKENS]:
            HISTORY.pop(mint, None)
            LAUNCH_TS.pop(mint, None)
            LAST_SCORE_AT.pop(mint, None)
            SUBSCRIBED.discard(mint)
            dropped.append(mint)
    for mint in list(HISTORY.keys()):
        last = HISTORY[mint][-1].ts if HISTORY[mint] else 0
        graduated = bool(HISTORY[mint]) and HISTORY[mint][-1].vsol_after >= grad_vsol
        if now - last > STALE_AFTER_SEC or graduated:
            if graduated and store is not None:
                store.set_outcome(mint, "graduated", outcome_ts=now)
            HISTORY.pop(mint, None)
            LAUNCH_TS.pop(mint, None)
            LAST_SCORE_AT.pop(mint, None)
            SUBSCRIBED.discard(mint)
            dropped.append(mint)
    return dropped


async def run(
    paper: bool = False,
    dashboard: bool = False,
    db_path: str = "pumpfun.db",
    model_name: str = "latest",
) -> None:
    if websockets is None:
        raise SystemExit("pip install websockets first")

    store = Store(db_path)
    model_store = ModelStore()
    fitted: FittedModel | None = None
    try:
        fitted = model_store.load(model_name)
        print(f"loaded fitted model: kind={fitted.kind} brier={fitted.brier_score:.4f}")
    except FileNotFoundError:
        print("no fitted model on disk — using hand-built heuristic (Phase 1 mode)")

    helius: HeliusClient | None = None
    use_logs_feed = False
    rpc_url = None
    if os.environ.get("HELIUS_API_KEY"):
        helius = HeliusClient()
        use_logs_feed = True
        rpc_url = helius.url
        print("Helius RPC configured — using real bot classifier + on-chain reserves")
        print("Trade data source: Helius logsSubscribe on the Pump program (every trade, no metered cost)")
    else:
        print("HELIUS_API_KEY unset — using size heuristic bot flag + reconstructed vSol")
        print("Trade data source: PumpPortal free feed (creates only; no trades without a metered key)")

    # Load the on-chain protocol constants (graduation threshold, virtual SOL
    # seed, fee schedule) so the breakeven formula uses live values, not the
    # hardcoded 115/30. Falls back to historical defaults if the read fails.
    if rpc_url:
        from config import load_protocol_constants
        pc = await load_protocol_constants(rpc_url)
        print(f"protocol constants: v_sol_graduation={pc.v_sol_graduation:.2f} "
              f"real_sol_grad={pc.real_sol_graduation:.2f} fee={(pc.fee_basis_points+pc.creator_fee)/100:.2f}%")
    else:
        pc = get_cached_constants()
        print(f"protocol constants (cached fallback): v_sol_graduation={pc.v_sol_graduation:.2f}")

    # Charter I5: warn if the fee schedule is stale (a regime change may have
    # happened since the last entry in _REGIMES). Backtests across a fee-regime
    # change use the wrong schedule if this isn't updated.
    from config import warn_if_stale_regime
    stale = warn_if_stale_regime()
    if stale:
        print(stale)

    broker = PaperBroker() if paper else None
    if broker is not None:
        # Attach the store so every entry/exit/fill is persisted to SQLite and
        # the bankroll + open positions reload on restart.
        broker.attach_store(store, reload=True)
        # Wire the graduation-price lookup so exits value at the real PumpSwap
        # first-trade price (charter I2), not the flat 5x placeholder.
        broker.graduation_price_lookup = lambda m: _broker_graduation_price(m, store)
        print(f"paper broker restored: bankroll={broker.bankroll:.3f} SOL, "
              f"{len(broker.positions)} open positions carried over")
    # In dashboard mode, the rich panel owns stdout — silence the alert stream's
    # competing prints (it still writes to alerts.log). Wire the auto-de-risk
    # callback so calibration drift halts the paper broker (charter §8).
    drift_action = broker.halt_trading if broker is not None else None
    alert = AlertStream(quiet=dashboard, drift_action=drift_action)
    dash = Dashboard()
    dash_ctx: Live | None = None
    if dashboard:
        dash_ctx = dash.live()
        dash_ctx.__enter__()

    # Counters for the feed-stats panel.
    class _Stats:
        n_tokens_seen = 0
        n_swaps_total = 0
        last_swaps_window = 0
        last_stats_ts = time.time()
    stats = _Stats()

    def _vsol_for(mint: str, sw: Swap) -> float:
        """Fill in vsol_after: prefer on-chain reserves, else reconstruct from history."""
        # Fee in effect at this swap's ts — was hardcoded 0.0125; now date-keyed.
        v_start = get_cached_constants().virtual_sol_start
        fee_rate = fee_at(sw.ts).curve_fee_rate
        prev = HISTORY[mint][-1].vsol_after if HISTORY[mint] else v_start
        delta = sw.sol_amount * (1 - fee_rate) * (1 if sw.is_buy else -1)
        # Floor at the virtual seed: vSol = virtual reserves (constant ~30) + real
        # reserves (>= 0), so it can never fall below v_start. This reconstruction
        # is approximate and can undershoot on a big sell; without the floor a
        # negative vsol_after gets logged (polluting the training set) and later
        # crashes the fractional-power grad-prob math downstream.
        return max(prev + delta, v_start)

    # USDC-paired tokens have a different breakeven geometry; exclude them from
    # scoring + the training set until a USDC path is built. Detected once per
    # mint via a one-shot bonding-curve reserve read (cached in NON_SOL_MINTS).
    NON_SOL_MINTS: set[str] = set()
    async def _is_sol_paired(mint: str) -> bool:
        if mint in NON_SOL_MINTS:
            return False
        if mint in _SOL_CONFIRMED:
            return True
        if helius is None:
            return True  # can't check without RPC; assume SOL (the common case)
        try:
            from config import bonding_curve_pda, decode_bonding_curve
            addr = bonding_curve_pda(mint)
            info = await helius.get_account_info(addr)
            if info and info.get("data"):
                raw_b64 = info["data"][0] if isinstance(info["data"], list) else info["data"]
                if isinstance(raw_b64, list):
                    raw_b64 = raw_b64[0]
                import base64 as _b64
                bc = decode_bonding_curve(_b64.b64decode(raw_b64))
                if bc is not None and not bc.is_sol_paired:
                    NON_SOL_MINTS.add(mint)
                    return False
            _SOL_CONFIRMED.add(mint)
            return True
        except Exception:
            return True  # on error, don't exclude — score it
    _SOL_CONFIRMED: set[str] = set()

    def _handle_swap(mint: str, sw: Swap, now: float) -> None:
        """Append + log + score + paper-trade a swap from either feed."""
        if mint in NON_SOL_MINTS:
            return  # USDC-paired; excluded from scoring/training
        sw.vsol_after = _vsol_for(mint, sw)
        HISTORY[mint].append(sw)
        store.log_swap(sw, mint)
        stats.n_swaps_total += 1

        # Paper-trade exit checks on every event (cheap).
        if broker is not None:
            vsol_by_mint = {m: h[-1].vsol_after for m, h in HISTORY.items() if h}
            closed = broker.check_exits(now, vsol_by_mint)
            for pos in closed:
                store.set_outcome(
                    pos.mint,
                    "graduated" if pos.status == "graduated" else "stalled",
                    outcome_ts=pos.exit_ts,
                )
                ev = PnlEvent(
                    mint=pos.mint, status=pos.status,
                    realized_return=pos.realized_return or 0.0,
                    bankroll_after=broker.bankroll, ts=now,
                )
                alert.on_pnl(ev)
                dash.on_pnl(ev)
            dash.set_bankroll(broker.bankroll)
            dash.set_open(len(broker.positions))
            dash.set_open_positions(_open_position_views(broker, now))

        # Rolling swaps/sec for the feed-stats panel.
        elapsed = now - stats.last_stats_ts
        if elapsed >= 5.0:
            dash.set_feed_stats(
                stats.n_tokens_seen, stats.n_swaps_total,
                (stats.n_swaps_total - stats.last_swaps_window) / elapsed,
            )
            stats.last_swaps_window = stats.n_swaps_total
            stats.last_stats_ts = now

        _maybe_score(mint, now, store, fitted, broker, alert, dash, dashboard)

    # ------------------------------------------------------------------ PumpPortal
    # PumpPortal gives us create events (token discovery + metadata) for free.
    # Trades come from the Helius logs feed when configured; without Helius we
    # fall back to per-token subscriptions (metered — needs PUMPPORTAL_API_KEY).
    ws_url = PUMPPORTAL_WS
    pp_key = os.environ.get("PUMPPORTAL_API_KEY")
    if pp_key:
        ws_url = f"{PUMPPORTAL_WS}?api-key={pp_key}"

    async def _pumpportal_creates() -> None:
        """Token discovery via PumpPortal's free subscribeNewToken stream.

        Reconnects with exponential backoff on disconnect — PumpPortal's FAQ
        warns websockets drop on network instability / server rebalancing.
        """
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=60) as ws:
                    await ws.send(json.dumps({"method": "subscribeNewToken"}))
                    await ws.send(json.dumps({"method": "subscribeMigration"}))
                    backoff = 1.0
                    async for raw in ws:
                        now = time.time()
                        try:
                            evt = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if (evt.get("txType") or evt.get("type")) == "create":
                            mint = evt.get("mint") or evt.get("token")
                            if mint:
                                LAUNCH_TS[mint] = now
                                store.register_token(mint, created_meta=evt)
                                stats.n_tokens_seen += 1
                                # Fallback path only: per-token trade subscription (metered).
                                if not use_logs_feed and pp_key and len(SUBSCRIBED) < MAX_SUBSCRIPTIONS:
                                    with contextlib.suppress(Exception):
                                        await ws.send(json.dumps({
                                            "method": "subscribeTokenTrade", "keys": [mint],
                                        }))
                                        SUBSCRIBED.add(mint)
                        elif (evt.get("txType") or evt.get("type")) == "migrate":
                            # Migration = graduation. Mark outcome + record the
                            # PumpSwap first-trade price (best-effort) so the
                            # backtest values exits at the real market price.
                            mint = evt.get("mint") or evt.get("token")
                            if mint:
                                store.set_outcome(mint, "graduated", outcome_ts=now, graduated_at_ts=now)
                                await _record_graduation_price(mint, now, helius, store)
                        elif not use_logs_feed:
                            # No Helius — use PumpPortal trade events directly (metered tier).
                            mint = evt.get("mint") or evt.get("token") or evt.get("ca")
                            if not mint:
                                continue
                            sw = await _to_swap(evt, mint, helius)
                            if sw is None:
                                continue
                            _handle_swap(mint, sw, now)
                            if int(now) % 30 == 0 or len(HISTORY) > MAX_TOKENS:
                                _gc(now, store, force=(len(HISTORY) > MAX_TOKENS))
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.InvalidStatus,
                websockets.exceptions.InvalidHandshake,
                ConnectionError,
                OSError,
            ) as e:
                import sys
                print(f"[pumpportal] disconnected ({type(e).__name__}: {e}); "
                      f"reconnecting in {backoff:.0f}s...", file=sys.stderr, flush=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except Exception as e:
                import sys
                print(f"[pumpportal] unexpected error ({type(e).__name__}: {e}); "
                      f"reconnecting in {backoff:.0f}s...", file=sys.stderr, flush=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _helius_logs_feed() -> None:
        """Trade + migration events via Helius logsSubscribe on the Pump + PumpSwap programs."""
        assert helius is not None
        feed = PumpLogsFeed(helius)
        async for evt in feed.stream():
            now = time.time()
            if evt[0] == "create":
                # The logs feed sees creates too, but PumpPortal's create event
                # carries richer metadata. We still register here as a fallback
                # in case PumpPortal is down. The mint isn't in the log event,
                # so we'd need to fetch the tx — skip and let PumpPortal handle it.
                continue
            if evt[0] == "migrate":
                # PumpSwap pool creation = graduation. Mark the outcome + trigger
                # the paper broker's auto-sell. This is the authoritative on-chain
                # graduation signal (more reliable than the vSol heuristic).
                _, mint, _ts = evt
                store.set_outcome(mint, "graduated", outcome_ts=now, graduated_at_ts=now)
                # Best-effort: fetch the PumpSwap first-trade price so the
                # backtest and paper broker value the exit at the real market
                # price (charter I2), not a flat 5x placeholder. Non-blocking
                # on failure — the broker falls back to the default multiple.
                await _record_graduation_price(mint, now, helius, store)
                if broker is not None and mint in broker.positions:
                    closed = broker.on_graduated(mint, now)
                    if closed:
                        ev = PnlEvent(
                            mint=mint, status="graduated",
                            realized_return=closed.realized_return or 0.0,
                            bankroll_after=broker.bankroll, ts=now,
                        )
                        alert.on_pnl(ev)
                        dash.on_pnl(ev)
                        dash.set_bankroll(broker.bankroll)
                        dash.set_open(len(broker.positions))
                continue
            if evt[0] == "gap":
                # logsSubscribe had a disconnect — the training set may have a
                # hole. Log it; backfill high-priority mints (open positions) via
                # feed.backfill_mint. The gap is recorded in feed.gaps for audit.
                _, last_slot, _gap_ts = evt
                import sys
                print(f"[gap] logs feed: data gap after slot {last_slot}; "
                      f"backfilling {len(broker.positions) if broker else 0} open positions...",
                      file=sys.stderr, flush=True)
                if broker is not None and helius is not None:
                    async def _backfill_one(
                        mint: str,
                        *,
                        _slot: int | None = last_slot,
                        _now: float = now,
                    ) -> None:
                        sigs = await feed.backfill_mint(mint)
                        for sig in sigs:
                            parsed = await helius.parse_trade_tx(sig)
                            if parsed and parsed["mint"] == mint:
                                # Stamp the swap with the real on-chain block time
                                # so temporal order inside the token's history is
                                # preserved. Falls back to _now only if blockTime
                                # was unavailable (old tx / pruned node).
                                sw_ts = parsed.get("block_time") or _now
                                sw = Swap(
                                    ts=sw_ts, wallet=parsed["trader"],
                                    is_buy=parsed["is_buy"], sol_amount=parsed["sol_amount"],
                                    vsol_after=0.0, via_frontend=parsed["via_frontend"],
                                )
                                _handle_swap(mint, sw, sw_ts)

                    await asyncio.gather(*(_backfill_one(m) for m in broker.positions))
                continue
            # swap
            _, sw, mint, _sig = evt
            if mint not in LAUNCH_TS:
                # First time we see this mint — register it (logs feed may see
                # trades before PumpPortal's create event arrives).
                LAUNCH_TS[mint] = now
                store.register_token(mint)
                stats.n_tokens_seen += 1
                # USDC-pairing check: exclude non-SOL tokens from scoring/training.
                # Done once per mint on first sight (cached in NON_SOL_MINTS).
                if not await _is_sol_paired(mint):
                    if not dashboard:
                        print(f"  [skip] {mint[:8]}.. USDC-paired — excluded (non-SOL breakeven geometry)")
                    continue
            elif mint in NON_SOL_MINTS:
                continue
            _handle_swap(mint, sw, now)
            if int(now) % 30 == 0 or len(HISTORY) > MAX_TOKENS:
                _gc(now, store, force=(len(HISTORY) > MAX_TOKENS))

    print("connecting to live feeds...")
    try:
        # Background dashboard tick — re-renders even with no events.
        async def _tick() -> None:
            while True:
                await asyncio.sleep(1.0)
                if broker is not None:
                    dash.set_bankroll(broker.bankroll)
                    dash.set_open(len(broker.positions))
                    dash.set_open_positions(_open_position_views(broker, time.time()))
                now = time.time()
                elapsed = max(now - stats.last_stats_ts, 1e-3)
                dash.set_feed_stats(
                    stats.n_tokens_seen, stats.n_swaps_total,
                    (stats.n_swaps_total - stats.last_swaps_window) / elapsed,
                )
                dash.tick()
        tick_task = asyncio.create_task(_tick()) if dashboard else None

        tasks = [asyncio.create_task(_pumpportal_creates())]
        if use_logs_feed:
            tasks.append(asyncio.create_task(_helius_logs_feed()))
        await asyncio.gather(*tasks)
    finally:
        if tick_task is not None:
            tick_task.cancel()
        if dash_ctx is not None:
            dash_ctx.__exit__(None, None, None)
        if helius is not None:
            await helius.close()
        store.close()


def _open_position_views(broker: PaperBroker, now: float) -> list[OpenPositionView]:
    """Snapshot the broker's open positions for the dashboard table."""
    views: list[OpenPositionView] = []
    for mint, p in broker.positions.items():
        # Real current vSol from the swap history; fall back to entry if dropped.
        cur_vsol = HISTORY[mint][-1].vsol_after if HISTORY.get(mint) else p.entry_vsol
        ratio = cur_vsol / max(p.entry_vsol, 1e-6)
        # Cap the rough unrealized at a sane range — the real exit value comes
        # from the bonding-curve payoff at graduation/stop time.
        unreal = max(min(ratio - 1.0, 5.0), -0.95)
        views.append(OpenPositionView(
            mint=mint, entry_ts=p.entry_ts, entry_sol=p.entry_sol,
            entry_vsol=p.entry_vsol, current_vsol=cur_vsol,
            age_sec=now - p.entry_ts, unrealized_return=unreal,
        ))
    return views


def _maybe_score(
    mint: str,
    now: float,
    store: Store,
    fitted: FittedModel | None,
    broker: PaperBroker | None,
    alert: AlertStream,
    dash: Dashboard,
    quiet: bool = False,
) -> None:
    if len(HISTORY[mint]) < MIN_SWAPS_TO_SCORE:
        return
    if now - LAST_SCORE_AT.get(mint, 0) < RESCORE_EVERY_SEC:
        return
    LAST_SCORE_AT[mint] = now

    swaps = HISTORY[mint]
    s = score_token(mint, swaps)

    # If a fitted model exists, recompute the graduation probability with it
    # and re-derive the edge/decision using the same thresholds. Otherwise the
    # hand-built estimate baked into the TokenScore stands.
    if fitted is not None:
        try:
            est = predict_grad_prob(fitted, swaps)
            brk = _breakeven_probability(s.vsol)
            edge = est - brk
            s = _rescore_with_model(s, est, edge)
        except Exception:
            pass

    ev = DecisionEvent(
        mint=mint, decision=s.decision, score=s.conviction_score,
        pct_to_graduation=s.pct_to_graduation, bot_share=s.bot_trade_share,
        sol_per_human=s.sol_per_human_wallet, edge=s.edge, ts=now,
    )
    alert.on_decision(ev)
    dash.on_decision(ev)

    # In dashboard mode the rich panel owns stdout — don't print. The dashboard's
    # "Watching" + "Recent decisions" tables already show this.
    if not quiet and s.decision in PRINT_DECISIONS:
        print(f"[{s.decision:5}] {mint[:8]}.. "
              f"score {s.conviction_score:5.1f} | "
              f"{s.pct_to_graduation:5.1f}% bonded | "
              f"bots {s.bot_trade_share:.0%} | "
              f"SOL/human {s.sol_per_human_wallet:+.3f} | "
              f"edge {s.edge:+.1%}")
        for r in s.reasons[:2]:
            print(f"          - {r}")

    # Paper-trade the ENTER.
    if broker is not None and s.decision == "ENTER":
        pos = broker.on_enter(mint, now, s.vsol)
        if pos is not None and not quiet:
            print(f"          > paper ENTER filled: {pos.entry_sol:.3f} SOL "
                  f"(cost {pos.entry_cost_sol:.4f})")


def _rescore_with_model(s: TokenScore, est: float, edge: float) -> TokenScore:
    """Re-derive the decision using the fitted model's P(graduation).

    Returns a *new* frozen TokenScore (charter discipline: the score object is
    immutable so the dashboard and the persisted DecisionEvent can't drift apart
    from the logged swap record).
    """
    if edge > 0.03 and s.conviction_score >= 55 and s.bot_trade_share < 0.6:
        decision = "ENTER"
    elif edge > 0 and s.conviction_score >= 40:
        decision = "WATCH"
    else:
        decision = "SKIP"
    from dataclasses import replace
    return replace(
        s,
        estimated_grad_prob=round(est, 4),
        edge=round(edge, 4),
        decision=decision,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Pump.fun conviction live monitor")
    ap.add_argument("--paper", action="store_true", help="paper-trade ENTER signals")
    ap.add_argument("--dashboard", action="store_true", help="rich live dashboard")
    ap.add_argument("--db", default=os.environ.get("PUMPFUN_DB", "pumpfun.db"))
    ap.add_argument("--model", default="latest", help="fitted model name (models/<name>.pkl)")
    args = ap.parse_args()
    try:
        asyncio.run(run(paper=args.paper, dashboard=args.dashboard, db_path=args.db, model_name=args.model))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()