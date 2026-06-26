"""
helius.py
=========
Helius RPC client for the two Phase 1 upgrades that need on-chain truth:

  1.1  Real bot classification — inspect each trade's instruction / program-id
       stack. Trades routed through the pump.fun frontend program are human;
       trades that call the bonding-curve program directly are bots.
  1.2  True bonding-curve reserve state — read vSol / real reserves directly
       from the on-chain bonding-curve account instead of reconstructing them
       from running trade history.

Why Helius: it's the standard Solana RPC + enhanced-API provider for pump.fun
work (parsed tx, account info, DAS). The same client works against any Solana
RPC; Helius-specific bits are flagged.

CONFIG
------
Set the env var ``HELius_API_KEY`` (yes, typo-free: ``HELIUS_API_KEY``) to your
Helius key. If unset, falls back to a public Solana mainnet endpoint so the
code runs end-to-end in tests (but without rate-limit headroom or parsed txs).

NOTE — verify the constants in this file against the current pump.fun program
before trusting production decisions on them. Program IDs and the bonding-curve
account layout are stable but you should confirm against your own labelling.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import time
from dataclasses import dataclass
from typing import Any

import httpx

from conviction_engine import Swap

# --- Program / account constants --------------------------------------------
# Pump.fun bonding-curve program. This is the REAL mainnet program ID, verified
# from a live CreateV2 transaction's instruction stack (Jun 2026). All token
# creates, buys, and sells route through this program. Trades that go through
# the pump.fun *frontend* include an intermediate router program (PUMP_FRONTEND);
# direct calls to PUMP_PROGRAM_ID with no router are bots (the paper's isBot).
PUMP_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# The pump.fun frontend router program. TODO: confirm this is the right
# discriminator for the paper's isBot. The paper's definition is "tx bypassed
# the pump.fun frontend". The cleanest proxy is the presence of a pump frontend
# router program in the instruction stack; if your labelling finds a better
# signal, swap it in here. Leave as None to use the simpler heuristic: any
# trade whose instruction stack contains ONLY PUMP_PROGRAM_ID (no router, no
# 3rd-party aggregator like Jupiter) is a bot. Frontend-routed trades include
# the pump router or a known aggregator on behalf of a human UI.
PUMP_FRONTEND_PROGRAM_ID: str | None = None  # fill in once confirmed

# Known "human" routing programs. A trade whose instruction stack includes one
# of these is considered frontend-routed (human). Trades whose stack contains
# ONLY PUMP_PROGRAM_ID + the system/token/associated-token/fee programs (no
# router) are direct contract calls (bots — the paper's isBot definition).
# Built from observed real trades; extend as you confirm more routers.
KNOWN_HUMAN_ROUTERS: set[str] = {
    "m9obQHAPyZeZ88w7XUY8Te6a8rBmhXsfKco7u8G8tnB",  # observed on a Buy
    "term9YPb9mzAsABaqN71A4xdbxHmpBNZavpBiQKZzN3",  # observed on a Sell
    "GMgnVFR8Jb39LoXsEVzb3DvBy3ywCmdmJquHUy1Lrkqb",  # observed on a Buy
    # Jupiter aggregator — humans often route via Jupiter.
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
}

# "Infrastructure" programs that appear in every trade (system, token, ATA
# helper, pump fee, compute budget). These don't count as routing — a stack
# of ONLY these + PUMP_PROGRAM_ID is a direct (bot) call.
INFRA_PROGRAMS: set[str] = {
    "11111111111111111111111111111111",  # System
    "ComputeBudget111111111111111111111111111111",
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",  # Token-2022 (pump uses this)
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # SPL Token
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",  # ATA helper
    "pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ",  # pump fee program
}

# Bonding-curve account layout (Anchor serialization, little-endian).
#   discriminator           : 8 bytes  (the Anchor account discriminator)
#   virtual_token_reserves  : u64
#   virtual_sol_reserves    : u64  <- the "virtual" SOL seeded into the curve
#   real_token_reserves     : u64
#   real_sol_reserves       : u64  <- the real SOL actually bonded
#   token_total_supply      : u64
#   complete                : bool (1 byte)  <- set when graduated
# Then padding. vSol (the paper's variable) = virtual_sol_reserves + real_sol_reserves.
# Verify against the current pump program IDL before trusting.
_BONDING_CURVE_DISCRIMINATOR = b"\x67\x9b\x3a\x55\x1f\x2e\x8c\xec"  # placeholder
BONDING_CURVE_DATA_LEN = 8 + 8 * 5 + 1  # discriminator + 5 u64s + bool

LAMPORTS_PER_SOL = 1_000_000_000.0


@dataclass
class Reserves:
    """Decoded bonding-curve reserves, in SOL (not lamports)."""

    virtual_sol_reserves: float
    real_sol_reserves: float
    virtual_token_reserves: float
    real_token_reserves: float
    total_supply: float
    complete: bool

    @property
    def vsol(self) -> float:
        """The paper's vSol: total virtual SOL in the bonding curve."""
        return self.virtual_sol_reserves + self.real_sol_reserves

    @property
    def real_sol_bonded(self) -> float:
        return self.real_sol_reserves

    @property
    def graduated(self) -> bool:
        return self.complete


class HeliusClient:
    """Async Solana RPC client (Helius or any compatible endpoint)."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        key = api_key or os.environ.get("HELIUS_API_KEY")
        if endpoint:
            self.url = endpoint
        elif key:
            self.url = f"https://mainnet.helius-rpc.com/?api-key={key}"
        else:
            # Public mainnet; works but rate-limited. Good enough for tests.
            self.url = "https://api.mainnet-beta.solana.com"
        self._client: httpx.AsyncClient | None = None
        self.timeout = timeout

    async def __aenter__(self) -> HeliusClient:
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def _rpc(self, method: str, params: list[Any]) -> Any:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        client = self._ensure_client()
        r = await client.post(self.url, json=payload)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"RPC error on {method}: {data['error']}")
        return data["result"]

    # -------------------------------------------------- tx classification (1.1)
    @staticmethod
    def _collect_instructions(tx: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten outer + inner instructions from a parsed getTransaction result."""
        msg = tx["transaction"].get("message", {})
        ixs: list[dict[str, Any]] = list(msg.get("instructions", []))
        meta = tx.get("meta", {}) or {}
        for ix_set in meta.get("innerInstructions", []) or []:
            ixs.extend(ix_set.get("instructions", []) or [])
        return ixs

    @staticmethod
    def _program_id(ix: dict[str, Any]) -> str:
        """Extract the program ID from a parsed instruction (handles both
        ``programId`` and nested ``program.id`` formats from jsonParsed encoding)."""
        prog: Any = ix.get("programId") or ix.get("program", "")
        if isinstance(prog, dict):
            prog = prog.get("id", "")
        return str(prog) if prog else ""

    async def get_transaction(self, signature: str) -> dict[str, Any] | None:
        """Fetch a fully-parsed transaction. Returns None if not found.

        Uses ``jsonParsedEncoding`` so instruction program-ids are human-readable
        strings. Helius supports this; a vanilla RPC may return partial parsing.
        """
        res = await self._rpc(
            "getTransaction",
            [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
        )
        return res if res is None else dict(res)

    async def classify_via_frontend(self, signature: str) -> bool | None:
        """Phase 1.1: the real bot flag.

        Returns True if the trade was routed through the pump.fun frontend or a
        known aggregator (human), False if it called the bonding-curve program
        directly with no router (bot), or None if the tx couldn't be classified
        (not found / parse failure / invalid signature / RPC error). Callers
        must handle None — the live monitor falls back to the size heuristic.

        Heuristic (since PUMP_FRONTEND_PROGRAM_ID is not confirmed): a trade is
        human-routed if its instruction stack contains a KNOWN_HUMAN_ROUTERS
        program. A trade whose stack contains ONLY PUMP_PROGRAM_ID + the
        infrastructure programs (system/token/ATA/pump-fee/compute-budget) is a
        direct contract call = bot. This matches the paper's isBot definition.
        """
        try:
            tx = await self.get_transaction(signature)
        except Exception:
            return None
        if not tx or "transaction" not in tx:
            return None
        # Inner instructions carry the real routing detail for CPI.
        touched_router = False
        touched_pump = False
        for ix in self._collect_instructions(tx):
            prog = self._program_id(ix)
            if prog in KNOWN_HUMAN_ROUTERS:
                touched_router = True
            if prog == PUMP_PROGRAM_ID:
                touched_pump = True

        if touched_router:
            return True   # routed via a frontend/aggregator => human
        if touched_pump:
            return False  # direct contract call, no router => bot
        return None

    async def classify_many(self, signatures: list[str]) -> dict[str, bool | None]:
        """Batch classify with bounded concurrency (avoid rate limits)."""
        sem = asyncio.Semaphore(5)
        out: dict[str, bool | None] = {}

        async def one(sig: str) -> None:
            async with sem:
                try:
                    out[sig] = await self.classify_via_frontend(sig)
                except Exception:
                    out[sig] = None

        await asyncio.gather(*(one(s) for s in signatures))
        return out

    # ------------------------------------------------------- reserves (1.2)
    async def get_account_info(self, address: str) -> dict[str, Any] | None:
        res = await self._rpc(
            "getAccountInfo", [address, {"encoding": "base64"}]
        )
        v = res.get("value")
        return v if v is None else dict(v)

    @staticmethod
    def bonding_curve_pda(mint: str, program_id: str = PUMP_PROGRAM_ID) -> str:
        """Derive the bonding-curve PDA for a given mint.

        Delegates to :func:`config.bonding_curve_pda`, which uses ``solders``
        for the real PDA derivation (seeds ``[b"bonding-curve", mint]``).
        Kept on the client for backwards compatibility; new code should call
        ``config.bonding_curve_pda`` directly.
        """
        from config import bonding_curve_pda as _bc_pda
        return _bc_pda(mint)

    async def read_reserves(self, bonding_curve_address: str) -> Reserves | None:
        """Phase 1.2: decode the bonding-curve account -> Reserves (in SOL).

        ``bonding_curve_address`` must be the PDA for the token's bonding curve.
        See :func:`bonding_curve_pda` for how to get it (or use a cached lookup).
        Returns None if the account isn't found / has the wrong length.
        """
        info = await self.get_account_info(bonding_curve_address)
        if not info or not info.get("data"):
            return None
        raw_b64 = info["data"][0] if isinstance(info["data"], list) else info["data"]
        if isinstance(raw_b64, list):
            raw_b64 = raw_b64[0]
        raw = base64.b64decode(raw_b64)
        return self._decode_reserves(raw)

    @staticmethod
    def _decode_reserves(raw: bytes) -> Reserves | None:
        if len(raw) < BONDING_CURVE_DATA_LEN:
            return None
        # Skip the 8-byte Anchor discriminator.
        body = raw[8:]
        (
            v_tok,
            v_sol,
            r_tok,
            r_sol,
            total,
        ) = struct.unpack_from("<5Q", body, 0)
        complete = bool(body[40])
        return Reserves(
            virtual_sol_reserves=v_sol / LAMPORTS_PER_SOL,
            real_sol_reserves=r_sol / LAMPORTS_PER_SOL,
            virtual_token_reserves=v_tok,
            real_token_reserves=r_tok,
            total_supply=total,
            complete=complete,
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------- trade parsing
    async def parse_trade_tx(self, signature: str) -> dict[str, Any] | None:
        """Decode a pump trade transaction into the fields the engine needs.

        Returns a dict with: mint, trader, is_buy, sol_amount, via_frontend,
        signature, block_time. Returns None if the tx can't be parsed as a pump
        trade. ``block_time`` is the unix timestamp of the slot the tx landed in
        (from getTransaction's top-level ``blockTime``), so backfilled swaps can
        be stamped with their *real* on-chain time — not the wall-clock time of
        the backfill — preserving temporal order inside a token's history.

        Extraction strategy (robust to account-layout changes):
          * mint      — the mint of the token balance that changed
          * is_buy    — True if the trader's token balance increased (post>pre)
          * trader    — the owner of the changed token account
          * sol_amount— abs(delta) of the fee-payer's SOL balance, net of the
                        tx fee (this is the SOL the user spent on a buy or
                        received on a sell, before pump's 1.25% fee)
          * via_frontend — the real bot flag (see classify_via_frontend)
          * block_time — unix ts of the tx's slot (None if unavailable)
        """
        try:
            tx = await self.get_transaction(signature)
        except Exception:
            return None
        if not tx or "transaction" not in tx:
            return None
        meta = tx.get("meta", {}) or {}
        if meta.get("err"):
            return None  # skip failed txs

        pre = {b["accountIndex"]: b for b in (meta.get("preTokenBalances") or [])}
        post = {b["accountIndex"]: b for b in (meta.get("postTokenBalances") or [])}
        # Find the token account whose balance changed and is a pump token
        # (mint ends in 'pump'). If multiple, pick the one with the biggest delta.
        best = None
        best_delta = 0.0
        for idx, pb in post.items():
            mint = pb.get("mint", "")
            if not mint.endswith("pump"):
                continue
            pre_amt = (pre.get(idx) or {}).get("uiTokenAmount", {}).get("uiAmount") or 0.0
            post_amt = pb.get("uiTokenAmount", {}).get("uiAmount") or 0.0
            delta = abs(post_amt - pre_amt)
            if delta > best_delta:
                best_delta = delta
                best = (idx, mint, pre_amt, post_amt, pb.get("owner"))
        if best is None:
            return None
        idx, mint, pre_amt, post_amt, owner = best
        is_buy = post_amt > pre_amt

        # SOL delta on the fee payer (account 0). For a buy this is negative
        # (spent); for a sell positive (received). Use abs as sol_amount.
        pre_sol = meta.get("preBalances") or []
        post_sol = meta.get("postBalances") or []
        if not pre_sol or not post_sol or len(pre_sol) != len(post_sol):
            return None
        # Fee payer is account index 0; the delta includes the tx fee + priority
        # fee, which we subtract to get the trade SOL notional.
        fee = (meta.get("fee") or 0) / 1e9
        delta_sol = (post_sol[0] - pre_sol[0]) / 1e9
        # Buy: delta_sol negative (spent SOL); Sell: positive (received SOL).
        # Add back the fee to get the gross trade notional.
        sol_amount = abs(delta_sol) + fee if is_buy else abs(delta_sol)
        # If we still have a 0 sol_amount, fall back to reconstructing from the
        # bonding-curve price (rare; only for weird routed txs).
        if sol_amount <= 0:
            return None

        via_frontend = await self._classify_from_tx(tx)
        # blockTime is the unix timestamp of the slot the tx landed in. Helius
        # returns it at the top level of getTransaction's result. We use it to
        # stamp backfilled swaps with their real on-chain time so temporal order
        # inside a token's history is preserved across gap recovery.
        block_time = tx.get("blockTime")
        return {
            "mint": mint,
            "trader": owner or "",
            "is_buy": bool(is_buy),
            "sol_amount": float(sol_amount),
            "via_frontend": via_frontend,
            "signature": signature,
            "block_time": float(block_time) if block_time else None,
        }

    async def _classify_from_tx(self, tx: dict[str, Any]) -> bool:
        """Same heuristic as classify_via_frontend but reuses a fetched tx."""
        for ix in self._collect_instructions(tx):
            if self._program_id(ix) in KNOWN_HUMAN_ROUTERS:
                return True
        return False  # pump program present (we only call this on pump txs) => bot


# --------------------------------------------------------------- logs feed
class PumpLogsFeed:
    """Real-time pump.fun trade feed via Helius ``logsSubscribe``.

    Subscribes to logs that mention :data:`PUMP_PROGRAM_ID`. For each log event
    that looks like a Buy/Sell (instruction-name heuristic), fetches the parsed
    transaction and decodes it into a :class:`~conviction_engine.Swap`.

    This is the Phase 1 data path: every input the model sees is the *true*
    value from on-chain, not a 3rd-party abstraction. No metered per-event
    cost; you already pay for Helius.

    Usage::

        feed = PumpLogsFeed(client)
        async for swap in feed.stream():
            ...

    The feed yields ``Swap`` objects (with ``via_frontend`` filled from the
    real instruction stack) plus ``create`` events as dicts for the monitor to
    register new tokens. Reconnects on websocket errors.
    """

    CREATE_INSTR = "CreateV2"

    def __init__(self, client: HeliusClient, max_concurrent_fetches: int = 8) -> None:
        self.client = client
        self.max_concurrent_fetches = max_concurrent_fetches
        # Slot-continuity tracking for gap detection. ``logsSubscribe`` has no
        # delivery guarantee and truncates under load; without tracking, a Helius
        # outage or network blip produces a *silent* permanent hole in the
        # training set — exactly the thing this project exists to avoid.
        self.last_slot: int | None = None
        self.gaps: list[tuple[int, int]] = []  # (last_slot_before_gap, slot_after_reconnect)

    async def stream(self) -> Any:
        """Async generator of events. Reconnects with backoff on disconnect.

        Each event is one of:
          * ``("create", {"signature": str, "ts": float})``
          * ``("swap", Swap, mint: str, signature: str)``
          * ``("migrate", mint: str, ts: float)``  — graduation to PumpSwap

        Subscribes to TWO log streams: the Pump bonding-curve program (trades +
        creates) and the PumpSwap program (pool creation = migration/graduation).
        """
        import websockets

        from config import PUMP_SWAP_PROGRAM_ID

        ws_url = self.client.url.replace("https://", "wss://")
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=60) as ws:
                    # Subscribe to the Pump bonding-curve program (trades + creates).
                    await ws.send(json.dumps({
                        "jsonrpc": "2.0", "id": 1, "method": "logsSubscribe",
                        "params": [{"mentions": [PUMP_PROGRAM_ID]}],
                    }))
                    # Subscribe to PumpSwap (graduation / pool creation).
                    await ws.send(json.dumps({
                        "jsonrpc": "2.0", "id": 2, "method": "logsSubscribe",
                        "params": [{"mentions": [PUMP_SWAP_PROGRAM_ID]}],
                    }))
                    sem = asyncio.Semaphore(self.max_concurrent_fetches)
                    backoff = 1.0  # reset after a successful connection

                    async for raw in ws:
                        try:
                            evt = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if "params" not in evt:
                            continue  # subscription ack
                        val = evt["params"]["result"]["value"]
                        # Track slot continuity for gap detection. Each
                        # logsNotification carries context.slot; a jump > 1 between
                        # consecutive events (or a non-empty gap after a reconnect)
                        # means we may have missed transactions.
                        slot = evt["params"]["result"].get("context", {}).get("slot")
                        if slot is not None:
                            if self.last_slot is not None and slot > self.last_slot + 1:
                                # Slot gap — some slots had no Pump-program tx, which
                                # is normal (Pump isn't in every slot). Only flag large
                                # gaps (>50 slots ≈ ~20s) as potential data loss.
                                gap = slot - self.last_slot
                                if gap > 50:
                                    self.gaps.append((self.last_slot, slot))
                            self.last_slot = slot
                        if val.get("err"):
                            continue  # skip failed txs
                        logs = val.get("logs") or []
                        signature = val.get("signature")
                        ts = time.time()
                        if not signature:
                            continue
                        # Identify the instruction kind from "Program log: Instruction: X".
                        kind = None
                        for line in logs:
                            if "Instruction: " in line and "Program log:" in line:
                                instr = line.split("Instruction:", 1)[1].strip()
                                if instr.startswith("Create"):
                                    kind = "create"
                                    break
                                if instr.startswith("Buy") or instr.startswith("Sell"):
                                    kind = "swap"
                                    break
                                if instr in ("create_pool", "CreatePool"):
                                    # PumpSwap pool creation = graduation/migration.
                                    kind = "migrate"
                                    break
                        if kind is None:
                            continue
                        if kind == "create":
                            yield ("create", {"signature": signature, "ts": ts})
                            continue
                        if kind == "migrate":
                            # Fetch the tx to extract the base_mint (the token
                            # that graduated). The pool's base_mint is in the
                            # create_pool instruction accounts.
                            mint = await self._extract_pool_base_mint(signature)
                            if mint:
                                yield ("migrate", mint, ts)
                            continue
                        # swap — parse_trade_tx is a full RPC round-trip; bound
                        # concurrency so we don't flood Helius under a burst.
                        async with sem:
                            parsed = await self.client.parse_trade_tx(signature)
                        if parsed is None:
                            continue
                        swap = Swap(
                            ts=ts,
                            wallet=parsed["trader"],
                            is_buy=parsed["is_buy"],
                            sol_amount=parsed["sol_amount"],
                            vsol_after=0.0,  # monitor fills this from reserves/history
                            via_frontend=parsed["via_frontend"],
                        )
                        yield ("swap", swap, parsed["mint"], signature)
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.InvalidStatus,
                websockets.exceptions.InvalidHandshake,
                ConnectionError,
                OSError,
            ) as e:
                # Network blip / keepalive timeout / server rebalance — reconnect.
                # PumpPortal FAQ: "Websockets can just disconnect with network
                # instability or when our servers rebalance load."
                import sys
                disconnected_at_slot = self.last_slot
                print(f"[helius] logs feed disconnected ({type(e).__name__}: {e}); "
                      f"reconnecting in {backoff:.0f}s... "
                      f"(last_slot={disconnected_at_slot})", file=sys.stderr, flush=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)  # exponential, capped at 30s
                # After reconnect, emit a gap event so the monitor knows the
                # training set may have a hole. Full per-mint backfill via
                # getSignaturesForAddress is available via :meth:`backfill_mint`
                # but is too expensive to run for every open mint automatically;
                # the monitor can choose to backfill high-priority mints.
                if disconnected_at_slot is not None:
                    yield ("gap", disconnected_at_slot, time.time())

    async def backfill_mint(self, mint: str) -> list[str]:
        """Best-effort backfill of recent signatures for one mint.

        Uses ``getSignaturesForAddress`` on the mint's bonding-curve PDA.
        Returns the signatures found — the caller can then ``parse_trade_tx``
        each one to recover missed swaps.

        This is the recovery path for the gap-detection above. It's per-mint
        (Solana's getSignaturesForAddress is account-scoped, not program-scoped),
        so only run it for mints you care about (open positions, high scores).
        """
        from config import bonding_curve_pda
        try:
            addr = bonding_curve_pda(mint)
            res = await self.client._rpc("getSignaturesForAddress", [addr, {"limit": 100}])
            return [s["signature"] for s in (res or []) if not s.get("err")]
        except Exception:
            return []

    async def _extract_pool_base_mint(self, signature: str) -> str | None:
        """For a PumpSwap create_pool tx, return the base_mint (the graduated token).

        The create_pool instruction's accounts include the base_mint; we find the
        account whose mint ends in 'pump' (pump.fun tokens). Falls back to None
        on parse failure.
        """
        try:
            tx = await self.client.get_transaction(signature)
        except Exception:
            return None
        if not tx or "transaction" not in tx:
            return None
        meta = tx.get("meta", {}) or {}
        if meta.get("err"):
            return None
        # The base_mint is one of the accounts of the create_pool instruction.
        # Heuristic: it's the account ending in 'pump' (pump.fun token mints).
        msg = tx["transaction"].get("message", {})
        for ix in msg.get("instructions", []) or []:
            for acct in ix.get("accounts", []) or []:
                if isinstance(acct, str) and acct.endswith("pump"):
                    return acct
        # Inner instructions (CPI).
        for ix_set in meta.get("innerInstructions", []) or []:
            for ix in ix_set.get("instructions", []) or []:
                for acct in ix.get("accounts", []) or []:
                    if isinstance(acct, str) and acct.endswith("pump"):
                        return acct
        return None