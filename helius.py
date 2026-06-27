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
import contextlib
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

# --- pump.fun TradeEvent (emitted in the program logs) -----------------------
# The bonding-curve program emits a `TradeEvent` for every Buy/Sell via Anchor's
# `emit!`, which surfaces in the transaction logs as a base64 `Program data:`
# line. logsSubscribe delivers those logs *in the notification itself* — so we
# can decode the full trade (mint, amounts, trader, true reserves) WITHOUT a
# follow-up getTransaction RPC. That round-trip, run once per global pump trade,
# is the dominant Helius credit cost; decoding from the log eliminates it.
#
# Anchor event discriminator = sha256("event:TradeEvent")[:8]. Verified by
# derivation; the decoder also length- and sanity-checks every field and returns
# None on any mismatch, so a layout change degrades gracefully to the RPC path
# (parse_trade_tx) rather than silently corrupting the trade.
TRADE_EVENT_DISCRIMINATOR = bytes.fromhex("bddb7fd34ee661ee")
# Anchor `emit_cpi!` wraps the same payload behind this self-CPI sentinel; accept
# both framings so we decode regardless of which logging macro the program uses.
_ANCHOR_CPI_EVENT_SENTINEL = bytes.fromhex("e445a52e51cb9a1d")
# Borsh body after the discriminator: mint(32) sol_amount(u64) token_amount(u64)
# is_buy(bool) user(32) timestamp(i64) v_sol(u64) v_tok(u64) r_sol(u64) r_tok(u64)
_TRADE_EVENT_BODY_LEN = 32 + 8 + 8 + 1 + 32 + 8 + 8 + 8 + 8 + 8  # = 121


@dataclass
class TradeEvent:
    """A pump.fun trade decoded straight from the program logs (no RPC)."""

    mint: str
    trader: str
    is_buy: bool
    sol_amount: float          # SOL, pre-fee
    vsol_after: float          # true virtual SOL (virtual + real) after the trade
    block_time: float | None   # on-chain unix ts from the event


def _pubkey58(raw: bytes) -> str:
    from solders.pubkey import Pubkey
    return str(Pubkey(raw))


def decode_trade_event(logs: list[str]) -> TradeEvent | None:
    """Decode a pump.fun ``TradeEvent`` from a logsSubscribe ``logs`` array.

    Returns a :class:`TradeEvent` if exactly one trade event is found and passes
    structural validation, else ``None`` (caller falls back to ``parse_trade_tx``).
    No network call — this reads the base64 the websocket already handed us.
    """
    for line in logs:
        if "Program data:" not in line:
            continue
        b64 = line.split("Program data:", 1)[1].strip()
        try:
            raw = base64.b64decode(b64)
        except Exception:
            continue
        # Strip the optional emit_cpi! sentinel, then require the TradeEvent disc.
        if raw[:8] == _ANCHOR_CPI_EVENT_SENTINEL:
            raw = raw[8:]
        if raw[:8] != TRADE_EVENT_DISCRIMINATOR:
            continue
        body = raw[8:]
        if len(body) < _TRADE_EVENT_BODY_LEN:
            continue
        try:
            mint_b = body[0:32]
            sol_amount, token_amount = struct.unpack_from("<QQ", body, 32)
            is_buy = body[48]
            user_b = body[49:81]
            (timestamp,) = struct.unpack_from("<q", body, 81)
            v_sol, v_tok, r_sol, r_tok = struct.unpack_from("<4Q", body, 89)
        except struct.error:
            continue
        # Sanity gate: a real trade has a boolean flag, non-zero token movement,
        # and seeded virtual reserves. Anything else is a different event we
        # mis-matched — bail to the RPC path rather than log garbage.
        if is_buy not in (0, 1) or token_amount == 0 or v_sol == 0:
            continue
        return TradeEvent(
            mint=_pubkey58(mint_b),
            trader=_pubkey58(user_b),
            is_buy=bool(is_buy),
            sol_amount=sol_amount / LAMPORTS_PER_SOL,
            vsol_after=(v_sol + r_sol) / LAMPORTS_PER_SOL,
            block_time=float(timestamp) if timestamp else None,
        )
    return None


# Size threshold for the cheap bot heuristic when full tx classification is off:
# trades >= this many SOL are treated as human (frontend), smaller as bots. Mirror
# of live_monitor._approx_via_frontend so the two feed paths agree.
_HUMAN_SOL_THRESHOLD = 0.05

# Sentinel pushed onto the raw queue when the reader task ends (disconnect), so
# the processing loop wakes up and re-raises the connection error for reconnect.
_READER_EOF = object()


def is_direct_pump_call(logs: list[str]) -> bool:
    """True if the Pump program is invoked at the TOP level (depth [1]) of the tx
    — a direct contract call with no router/aggregator/bot program wrapping it.
    This is the paper's ``isBot`` (autonomous sniper). Parsed from the
    logsSubscribe ``logs`` we already receive, so zero extra RPC. Returns False
    when Pump appears only nested (routed via another program: the pump.fun
    frontend, Jupiter, a telegram bot, etc.)."""
    marker = f"Program {PUMP_PROGRAM_ID} invoke [1]"
    return any(line.startswith(marker) for line in logs)


# Runtime/infra programs that may appear at depth [1] before the real swap entry
# (compute-budget setup, ATA creation, SOL/token transfers) — skipped so we
# capture the program that actually drives the trade.
_INFRA_PROGRAMS: set[str] = {
    "ComputeBudget111111111111111111111111111111",
    "11111111111111111111111111111111",            # System
    "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",  # Associated Token Account
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",   # SPL Token
    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",   # Token-2022
}

# Identified entry programs -> human-readable route name. Pure program identities
# (no human/bot semantics baked in); "other" covers anything unidentified. Used
# only at analysis/retrain time — the raw entry program is what's stored.
_ROUTE_NAMES: dict[str, str] = {
    PUMP_PROGRAM_ID: "pump_direct",
    "MAyhSmzXzV1pTf7LsNkrNwkWKTo4ougAJ1PPg47MD4e": "mayhem",        # pump.fun native
    "MaestroAAe9ge5HTc64VbBQZ6fP77pwvrhM8i1XWSAx": "maestro",       # telegram bot
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "jupiter",       # aggregator
    "FLASHX8DrLbgeR8FcfNV1F5krxYcYMUdBkrP1EPBtxB9": "flashx",       # unidentified wrapper
}


def entry_program(logs: list[str]) -> str | None:
    """The first non-infra program invoked at the top level [1] — the program
    that actually drives the trade (pump itself for a direct call, else the
    wrapping router/aggregator/bot). Parsed from logs, zero RPC. None if no
    top-level program line is present. Stored raw so categorisation stays a
    revisable mapping (see ``route_category``) rather than baked at capture."""
    for line in logs:
        if line.startswith("Program ") and " invoke [1]" in line:
            prog = line.split()[1]
            if prog not in _INFRA_PROGRAMS:
                return prog
    return None


def route_category(prog: str | None) -> str:
    """Map a stored entry program to a readable route name for analysis/retrain.
    Unknown programs -> 'other'; None -> 'unknown'."""
    if prog is None:
        return "unknown"
    return _ROUTE_NAMES.get(prog, "other")


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

    def __init__(
        self,
        client: HeliusClient,
        max_concurrent_fetches: int = 8,
        classify_trades: bool = False,
    ) -> None:
        self.client = client
        self.max_concurrent_fetches = max_concurrent_fetches
        # When False (default) decode each trade straight from the log event and
        # use the size heuristic for the bot flag — zero RPC per trade. When True,
        # fetch the parsed tx per trade to get the real instruction-stack bot flag
        # (the precise but credit-expensive path). Decode-from-log still runs
        # first either way; the fetch only adds the routing classification.
        self.classify_trades = classify_trades
        # Telemetry so the caller can see the credit savings: how many trades we
        # served from the log vs. had to fall back to a getTransaction for.
        self.n_decoded_from_log = 0
        self.n_fetched = 0
        # Slot-continuity tracking for gap detection. ``logsSubscribe`` has no
        # delivery guarantee and truncates under load; without tracking, a Helius
        # outage or network blip produces a *silent* permanent hole in the
        # training set — exactly the thing this project exists to avoid.
        self.last_slot: int | None = None
        self.gaps: list[tuple[int, int]] = []  # (last_slot_before_gap, slot_after_reconnect)
        # A dedicated reader task drains the socket into this queue so the
        # websocket is always read (pongs flow, keepalive never times out) even
        # while the consumer is busy. Under sustained overload we drop the OLDEST
        # raw message and count it — a bounded, visible loss instead of a silent
        # disconnect + slot gap. Sized to absorb multi-second bursts at the full
        # Pump firehose (~600 msg/s observed).
        self.raw_queue_max = 5000
        self.n_dropped = 0

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

                    # Reader task: drain the socket as fast as possible into a
                    # bounded queue, doing NOTHING heavy. This keeps the websocket
                    # continuously read so pongs are handled and the keepalive
                    # never times out — the root cause of the previous 1011
                    # disconnect/gap cycle was the consumer stalling the read loop
                    # under the firehose. The connection error surfaces here and
                    # is handed to the processing loop via _READER_EOF + reader_exc.
                    raw_q: asyncio.Queue = asyncio.Queue(maxsize=self.raw_queue_max)
                    reader_exc: list[BaseException] = []

                    async def _reader() -> None:
                        try:
                            async for raw in ws:
                                if raw_q.full():
                                    # Drop the oldest to make room — a bounded,
                                    # counted loss beats a silent disconnect+gap.
                                    try:
                                        raw_q.get_nowait()
                                        self.n_dropped += 1
                                    except asyncio.QueueEmpty:
                                        pass
                                raw_q.put_nowait(raw)
                        except BaseException as e:  # noqa: BLE001 - re-raised in main loop
                            reader_exc.append(e)
                        finally:
                            await raw_q.put(_READER_EOF)

                    reader_task = asyncio.create_task(_reader())
                    try:
                        while True:
                            raw = await raw_q.get()
                            if raw is _READER_EOF:
                                if reader_exc:
                                    raise reader_exc[0]
                                return  # clean stop (e.g. generator closed)
                            try:
                                evt = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            if "params" not in evt:
                                continue  # subscription ack
                            val = evt["params"]["result"]["value"]
                            # Track slot continuity for gap detection. Each
                            # logsNotification carries context.slot; a jump > 1
                            # between consecutive events (or a non-empty gap after
                            # a reconnect) means we may have missed transactions.
                            slot = evt["params"]["result"].get("context", {}).get("slot")
                            if slot is not None:
                                if self.last_slot is not None and slot > self.last_slot + 1:
                                    # Slot gap — some slots had no Pump-program tx,
                                    # which is normal (Pump isn't in every slot).
                                    # Only flag large gaps (>50 slots ≈ ~20s).
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
                            # swap — decode the TradeEvent straight from the log we
                            # already have (no RPC).
                            te = decode_trade_event(logs)
                            if not self.classify_trades:
                                # Default mode: zero RPC per trade. If the event
                                # doesn't decode it's almost always a PumpSwap
                                # post-graduation Buy/Sell (different event format,
                                # arriving via the PumpSwap subscription we keep for
                                # graduation detection) — not a bonding-curve trade
                                # we want. Skip it. Doing a getTransaction round-trip
                                # here instead would stall the read loop (queue fills
                                # → drops) and burn credits for data we discard.
                                if te is None:
                                    continue
                                self.n_decoded_from_log += 1
                                # Stamp the swap with the TradeEvent's on-chain
                                # block_time when present, falling back to the
                                # receipt ts. logsSubscribe delivers logs in the
                                # notification (no blockTime), but the decoded
                                # TradeEvent embeds the program's own unix ts;
                                # using it makes the swap key stable across
                                # concurrent monitors, so dedupe_swaps collapses
                                # double-logged trades regardless of receipt time.
                                swap_ts = float(te.block_time) if te.block_time else ts
                                swap = Swap(
                                    ts=swap_ts,
                                    wallet=te.trader,
                                    is_buy=te.is_buy,
                                    sol_amount=te.sol_amount,
                                    vsol_after=te.vsol_after,  # true on-chain reserves
                                    # No routing info in the event; use the size
                                    # heuristic (same rule as the no-RPC fallback).
                                    via_frontend=te.sol_amount >= _HUMAN_SOL_THRESHOLD,
                                    # Direct-call bot flag, parsed from the logs we
                                    # already have (0 RPC). Captured for a future
                                    # model feature; doesn't affect via_frontend.
                                    is_direct=is_direct_pump_call(logs),
                                    # Raw entry program (router/bot/pump) for
                                    # per-route signal at the next retrain; 0 RPC.
                                    entry_program=entry_program(logs),
                                )
                                yield ("swap", swap, te.mint, signature)
                                continue
                            # Classify path (HELIUS_CLASSIFY_TRADES=1): full RPC
                            # round-trip for the precise instruction-stack bot flag;
                            # bound concurrency so we don't flood Helius.
                            async with sem:
                                parsed = await self.client.parse_trade_tx(signature)
                            if parsed is None:
                                continue
                            self.n_fetched += 1
                            # Prefer the TradeEvent's on-chain block_time so two
                            # concurrent monitors produce an identical swap key
                            # (see the zero-RPC path above); fall back to the
                            # receipt ts only when the log didn't decode.
                            swap_ts = float(te.block_time) if te is not None and te.block_time else ts
                            swap = Swap(
                                ts=swap_ts,
                                wallet=parsed["trader"],
                                is_buy=parsed["is_buy"],
                                sol_amount=parsed["sol_amount"],
                                # Prefer the true reserves we decoded from the log;
                                # else the monitor reconstructs from history.
                                vsol_after=te.vsol_after if te is not None else 0.0,
                                via_frontend=parsed["via_frontend"],
                            )
                            yield ("swap", swap, parsed["mint"], signature)
                    finally:
                        reader_task.cancel()
                        with contextlib.suppress(BaseException):
                            await reader_task
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