"""
config.py
=========
The single source of truth for fee schedules, slippage/MEV assumptions, and
the on-chain-derived protocol constants that the breakeven formula depends on.

Why this module exists
----------------------
Pump.fun's fee schedule is **non-stationary**. As of 2026 the bonding-curve fee
moved from 1.25% to ~1.0% (95 bps protocol + 5 bps creator, live on-chain), and
creators can now pair launches with USDC instead of SOL — a different fee/quote
geometry that isn't comparable to the SOL breakeven. PumpSwap (the in-house
graduation DEX) charges 0.25% (20 bps LP + 5 bps protocol). A fixed 1.25%
applied across a fee-regime change biases post-cost returns — exactly the number
the whole project exists to measure.

This module replaces the hardcoded ``FEE_RATE = 0.0125`` and the ``115`` /
``30.0`` constants with:

  * :class:`FeeSchedule` — a date-keyed table of (protocol_bps, creator_bps,
    pumpswap_lp_bps, pumpswap_protocol_bps, slippage, mev_penalty) so a
    backtest replays the schedule that was live at each trade's timestamp.
  * :func:`fee_at` — look up the schedule in effect at a given unix ts.
  * :class:`ProtocolConstants` — the on-chain-derived values
    ``initial_virtual_sol_reserves`` (was hardcoded 30.0) and the graduation
    threshold ``vSol_graduation`` (was hardcoded 115.0 = 30 virtual + 85 real).
    Pulled from the Pump program's Global account + a bonding-curve sample, so
    a parameter change doesn't quietly invalidate every EV decision.
  * :func:`load_protocol_constants` — async fetch + decode of the Global account
    and a bonding-curve reserve read, returning the live values. Call once on
    startup; the values are cached.

Quote-mint (USDC) handling
--------------------------
The bonding-curve account now carries a ``quote_mint`` field. SOL-paired coins
have ``quote_mint == Pubkey::default()`` (32 zero bytes); USDC-paired coins have
the SPL USDC mint. The two geometries have different breakevens, so
:func:`is_sol_paired` exposes the check and the monitor/backtest exclude
non-SOL coins from scoring and the training set until a USDC path is built.
"""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass

# --- Program IDs (verified from the official pump-public-docs repo) ---------
PUMP_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMP_SWAP_PROGRAM_ID = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
# Global config PDA — ["global"] seeds. Holds initial_virtual_sol_reserves etc.
PUMP_GLOBAL_ACCOUNT = "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf"
PUMP_SWAP_GLOBAL_CONFIG = "ADyA8hdefvWN2dbGGWFotbzWxrAvLW83WG6QCVXvJKqw"
# SOL quote-mint sentinel: the wrapped-SOL mint, used as the "SOL" quote for
# the V2 instructions. The bonding-curve's quote_mint field is Pubkey::default()
# (all zeros) for legacy SOL coins; the V2 instructions pass So111... as quoteMint.
WRAPPED_SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
LAMPORTS_PER_SOL = 1_000_000_000.0

# --- Locked holdout fence (charter §9.1) ------------------------------------
# The most recent contiguous time slice reserved as the holdout. Tokens whose
# first swap ts >= HOLDOUT_START_TS belong to the holdout and MUST NOT enter
# training, feature selection, or threshold tuning. It is read exactly once,
# at the §5.4 capital gate. Pre-register the span before looking at it; set to
# None until the holdout is reserved (the current state — no holdout yet).
#
# To reserve: set this to the unix timestamp of the start of your holdout
# window, commit it, and do NOT run `cli.py backtest` on data that includes
# it. The backtest loader (backtest.load_examples_from_store) filters on this.
HOLDOUT_START_TS: float | None = None


# --- Date-keyed fee schedule ------------------------------------------------
# Each entry is the schedule in effect from `effective_from` (unix ts) onward.
# Values are in basis points for fees; slippage/mev are fractions.
# Sources: pump-public-docs (PUMP_PROGRAM_README.md, PUMP_SWAP_README.md) +
# on-chain reads of the Global account (fee_basis_points/creator_fee) and
# PumpSwap GlobalConfig (lp_fee / protocol_fee). The live on-chain values are
# the truth — this table is the *backtest* fallback for dates we can't re-read.
@dataclass(frozen=True)
class FeeSchedule:
    protocol_bps: int          # bonding-curve protocol fee, bps per side
    creator_bps: int            # bonding-curve creator fee, bps per side
    pumpswap_lp_bps: int        # PumpSwap LP fee, bps per side (post-graduation)
    pumpswap_protocol_bps: int  # PumpSwap protocol fee, bps per side
    slippage: float             # assumed average slippage fraction per side
    mev_penalty: float          # assumed MEV/sandwich penalty on entry (fraction)

    @property
    def curve_fee_rate(self) -> float:
        """Total bonding-curve fee fraction per side = (protocol + creator) / 1e4."""
        return (self.protocol_bps + self.creator_bps) / 10_000.0

    @property
    def pumpswap_fee_rate(self) -> float:
        """Total PumpSwap fee fraction per side (post-graduation)."""
        return (self.pumpswap_lp_bps + self.pumpswap_protocol_bps) / 10_000.0

    def entry_cost(self) -> float:
        """Total entry cost fraction: curve fee + slippage + MEV."""
        return self.curve_fee_rate + self.slippage + self.mev_penalty

    def exit_cost_curve(self) -> float:
        """Exit cost from a bonding-curve position (curve fee + slippage; no MEV)."""
        return self.curve_fee_rate + self.slippage

    def exit_cost_pumpswap(self) -> float:
        """Exit cost from a graduated position on PumpSwap (pumpswap fee + slippage)."""
        return self.pumpswap_fee_rate + self.slippage


# Known historical regimes. Append new entries as the schedule changes; never
# edit an existing entry (it's already baked into logged data). Ordered by ts.
# Pre-2026-05: 1.25% per side (95 bps protocol + 30 bps creator per the original
#   codebase comment). Slippage 2%, MEV 1%.
# 2026-05-21 onward: ~1.0% per side (95 bps protocol + 5 bps creator, live
#   on-chain). PumpSwap 0.30% (20+5 bps). Same slippage/MEV assumptions.
_REGIMES: list[tuple[float, FeeSchedule]] = [
    (0.0, FeeSchedule(
        protocol_bps=95, creator_bps=30, pumpswap_lp_bps=25, pumpswap_protocol_bps=5,
        slippage=0.02, mev_penalty=0.01,
    )),
    # 2026-05-21 00:00 UTC — USDC pairing launch + revised creator fee.
    (1_779_321_600, FeeSchedule(
        protocol_bps=95, creator_bps=5, pumpswap_lp_bps=20, pumpswap_protocol_bps=5,
        slippage=0.02, mev_penalty=0.01,
    )),
]


def fee_at(ts: float) -> FeeSchedule:
    """Return the fee schedule in effect at unix timestamp ``ts``.

    Picks the regime with the largest ``effective_from`` <= ts. If ``ts`` is
    before the first regime, returns the first regime (the earliest known).
    """
    active = _REGIMES[0][1]
    for eff, sched in _REGIMES:
        if ts >= eff:
            active = sched
        else:
            break
    return active


def warn_if_stale_regime(now: float | None = None) -> str | None:
    """Return a warning string if the latest fee regime is older than 90 days,
    else None. Charter I5 (date-keyed costs): when the protocol changes fees
    again (it will), every backtest run *after* the change but on data *before*
    the change silently uses the wrong schedule if ``fee_at(now)`` is called.
    Call this on startup to surface a stale schedule early.
    """
    import time as _time
    now = now if now is not None else _time.time()
    last_eff, _ = _REGIMES[-1]
    age_sec = now - last_eff
    if age_sec > 90 * 86400:
        return (
            f"WARNING: latest fee regime is {age_sec / 86400:.0f} days old "
            f"(effective_from={last_eff}). If pump.fun has changed fees since, "
            f"add a new entry to _REGIMES in config.py — backtests across the "
            f"change will otherwise use the wrong schedule (charter I5)."
        )
    return None


# --- On-chain protocol constants -------------------------------------------
@dataclass(frozen=True)
class ProtocolConstants:
    """Protocol-wide values derived from the Pump Global account.

    These replace the hardcoded ``VIRTUAL_SOL_START = 30.0`` and
    ``GRADUATION_VSOL = 115.0``. Graduation vSol is *derived*: it's the
    initial virtual reserves + the real SOL threshold the program enforces,
    not a magic number. A parameter change on-chain propagates here.
    """

    initial_virtual_sol_reserves: float   # was VIRTUAL_SOL_START (30.0 SOL)
    initial_virtual_token_reserves: float
    initial_real_token_reserves: float
    token_total_supply: float
    fee_basis_points: int                  # protocol fee, bps
    creator_fee: int                       # creator fee, bps
    # The graduation threshold in vSol. Derived as:
    #   virtual_sol_reserves_start + real_sol_graduation_threshold
    # where the real-SOL threshold is initial_real_token_reserves-derived and
    # enforced by the program (the bonding curve completes when real_token_reserves
    # hits 0, which corresponds to ~85 SOL raised at the initial price). We expose
    # it as a separate field so a program change to the threshold propagates.
    v_sol_graduation: float                # was GRADUATION_VSOL (115.0)
    real_sol_graduation: float             # was REAL_SOL_TARGET (85.0)

    @property
    def virtual_sol_start(self) -> float:
        return self.initial_virtual_sol_reserves

    @property
    def real_sol_target(self) -> float:
        return self.real_sol_graduation


def decode_global_account(raw: bytes) -> ProtocolConstants:
    """Decode the Pump Global account bytes into ProtocolConstants.

    Layout (Anchor, packed — no inter-field padding; verified against live data):
      discriminator(8) + initialized(bool,1) + authority(32) + fee_recipient(32) +
      initial_virtual_token_reserves(u64) + initial_virtual_sol_reserves(u64) +
      initial_real_token_reserves(u64) + token_total_supply(u64) + fee_basis_points(u64) +
      withdraw_authority(32) + enable_migrate(bool,1) + pool_migration_fee(u64) +
      creator_fee(u64) + ...
    All u64s are little-endian, in lamports / raw units.
    """
    if len(raw) < 8 + 65 + 8 * 5 + 32 + 1 + 8 + 8:
        raise ValueError(f"Global account too short: {len(raw)} bytes")
    body = raw[8:]  # skip discriminator
    off = 1 + 32 + 32  # initialized + authority + fee_recipient = 65
    ivtr = struct.unpack_from("<Q", body, off)[0]
    off += 8
    ivsr = struct.unpack_from("<Q", body, off)[0]
    off += 8
    irtr = struct.unpack_from("<Q", body, off)[0]
    off += 8
    tts = struct.unpack_from("<Q", body, off)[0]
    off += 8
    fbp = struct.unpack_from("<Q", body, off)[0]
    off += 8
    off += 32  # withdraw_authority
    off += 1   # enable_migrate
    off += 8   # pool_migration_fee
    creator_fee_lamports = struct.unpack_from("<Q", body, off)[0]

    ivsr_sol = ivsr / LAMPORTS_PER_SOL
    # Graduation real-SOL threshold via the constant-product (Uniswap-V2) curve.
    # The bonding curve is constant-product: (ivsr + real_sol) * (ivtr - irtr) = ivsr*ivtr.
    # It completes when all real tokens are bought (real_token_reserves -> 0).
    # Solving: real_sol_raised = ivsr * irtr / (ivtr - irtr). This is the ~85 SOL
    # the paper measured; deriving it on-chain means a parameter change propagates.
    real_sol_grad = (ivsr * irtr / (ivtr - irtr)) / LAMPORTS_PER_SOL if (ivtr - irtr) > 0 else 0.0
    v_sol_grad = ivsr_sol + real_sol_grad

    return ProtocolConstants(
        initial_virtual_sol_reserves=ivsr_sol,
        initial_virtual_token_reserves=float(ivtr),
        initial_real_token_reserves=float(irtr),
        token_total_supply=float(tts),
        fee_basis_points=int(fbp),
        creator_fee=int(creator_fee_lamports),  # on-chain this is lamports; treat as bps for the schedule
        v_sol_graduation=v_sol_grad,
        real_sol_graduation=real_sol_grad,
    )


# --- Bonding-curve account decoder (quote-mint + graduation) ----------------
@dataclass(frozen=True)
class BondingCurve:
    """Decoded bonding-curve account for one mint. All reserves in SOL/units."""

    virtual_sol_reserves: float
    real_sol_reserves: float
    virtual_token_reserves: float
    real_token_reserves: float
    total_supply: float
    complete: bool              # the real graduation flag (set on curve completion)
    quote_mint: str             # Pubkey::default() (all-zeros) = SOL-paired

    @property
    def vsol(self) -> float:
        return self.virtual_sol_reserves + self.real_sol_reserves

    @property
    def graduated(self) -> bool:
        return self.complete

    @property
    def is_sol_paired(self) -> bool:
        """True if this token is SOL-denominated (the only geometry we model)."""
        return self.quote_mint in ("11111111111111111111111111111111", WRAPPED_SOL_MINT)


def decode_bonding_curve(raw: bytes) -> BondingCurve | None:
    """Decode a bonding-curve account. Returns None if the bytes don't fit.

    Layout (Anchor, packed): discriminator(8) + virtual_token_reserves(u64) +
    virtual_sol_reserves(u64) + real_token_reserves(u64) + real_sol_reserves(u64) +
    token_total_supply(u64) + complete(bool,1) + [quote_mint(pubkey,32) if present]
    """
    if len(raw) < 8 + 8 * 5 + 1:
        return None
    body = raw[8:]
    v_tok, v_sol, r_tok, r_sol, total = struct.unpack_from("<5Q", body, 0)
    complete = bool(body[40])
    # quote_mint was added in the V2 update; it sits at offset 41 if present.
    quote_mint = "11111111111111111111111111111111"  # default = SOL-paired
    if len(body) >= 73:
        qm_bytes = body[41:73]
        if qm_bytes == b"\x00" * 32:
            quote_mint = "11111111111111111111111111111111"  # Pubkey::default()
        else:
            try:
                from solders.pubkey import Pubkey
                quote_mint = str(Pubkey(qm_bytes))
            except Exception:
                pass
    return BondingCurve(
        virtual_sol_reserves=v_sol / LAMPORTS_PER_SOL,
        real_sol_reserves=r_sol / LAMPORTS_PER_SOL,
        virtual_token_reserves=float(v_tok),
        real_token_reserves=float(r_tok),
        total_supply=float(total),
        complete=complete,
        quote_mint=quote_mint,
    )


def bonding_curve_pda(mint: str) -> str:
    """Derive the bonding-curve PDA for a mint. Requires ``solders``."""
    from solders.pubkey import Pubkey
    bc, _ = Pubkey.find_program_address(
        [b"bonding-curve", bytes(Pubkey.from_string(mint))],
        Pubkey.from_string(PUMP_PROGRAM_ID),
    )
    return str(bc)


def is_sol_paired(quote_mint: str) -> bool:
    """True if the quote mint is SOL (default or wrapped-SOL)."""
    return quote_mint in ("11111111111111111111111111111111", WRAPPED_SOL_MINT)


# --- Cached protocol constants (loaded once on startup) --------------------
_cached: ProtocolConstants | None = None


def get_cached_constants() -> ProtocolConstants:
    """Return the cached ProtocolConstants, or a safe fallback if not yet loaded.

    The fallback uses the historical defaults (30/85/115) so the system runs
    before the first on-chain read completes. Call :func:`load_protocol_constants`
    on startup to populate the cache with the live values.
    """
    global _cached
    if _cached is not None:
        return _cached
    # Historical fallback — matches the pre-2026 constants.
    _cached = ProtocolConstants(
        initial_virtual_sol_reserves=30.0,
        initial_virtual_token_reserves=1_073_000_000_000_000,
        initial_real_token_reserves=793_100_000_000_000,
        token_total_supply=1_000_000_000_000_000,
        fee_basis_points=95,
        creator_fee=30,
        v_sol_graduation=115.0,
        real_sol_graduation=85.0,
    )
    return _cached


def set_cached_constants(pc: ProtocolConstants) -> None:
    """Populate the cache (called by :func:`load_protocol_constants`)."""
    global _cached
    _cached = pc


async def load_protocol_constants(rpc_url: str) -> ProtocolConstants:
    """Fetch + decode the Pump Global account from the given RPC URL.

    Call once on startup so the breakeven formula uses the live graduation
    threshold rather than the hardcoded 115. Falls back to the cache if the
    read fails (network error / account not found).
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
                    "params": [PUMP_GLOBAL_ACCOUNT, {"encoding": "base64", "commitment": "confirmed"}],
                },
            )
            v = r.json()["result"]["value"]
            if v is None:
                raise RuntimeError("Global account not found")
            raw = base64.b64decode(v["data"][0] if isinstance(v["data"], list) else v["data"])
            pc = decode_global_account(raw)
            set_cached_constants(pc)
            return pc
    except Exception:
        return get_cached_constants()


# ---------------------------------------------------------------------------
# Live fee-regime checker
# ---------------------------------------------------------------------------

def _decode_pumpswap_config(raw: bytes) -> tuple[int, int] | None:
    """Decode PumpSwap GlobalConfig bytes → (lp_fee_bps, protocol_fee_bps).

    Inferred Anchor layout:
      discriminator(8) + admin(Pubkey,32) +
      lp_fee_basis_points(u64) + protocol_fee_basis_points(u64) + …

    Returns None if bytes are too short or values fail sanity range checks
    (0–500 bps each), so a wrong layout produces None rather than garbage.
    """
    if len(raw) < 56:  # 8 + 32 + 8 + 8
        return None
    try:
        lp_fee = struct.unpack_from("<Q", raw, 40)[0]
        proto_fee = struct.unpack_from("<Q", raw, 48)[0]
        if not (0 <= lp_fee <= 500 and 0 <= proto_fee <= 500):
            return None
        return int(lp_fee), int(proto_fee)
    except Exception:
        return None


def _append_regime_entry_to_file(ts: float, sched: FeeSchedule) -> bool:
    """Insert a new _REGIMES entry into config.py on disk.

    Locates the closing ] of the _REGIMES list by finding the rightmost
    "    )),\\n]" in the file (unique in config.py) and inserts the new
    entry immediately before it. Returns True on success.
    """
    import datetime as _dt
    import pathlib as _pl

    config_path = _pl.Path(__file__).with_name("config.py")
    try:
        text = config_path.read_text()
        ts_int = int(ts)
        dt_str = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        new_entry = (
            f"    # {dt_str} — on-chain fee change detected;"
            f" slippage/mev carried from prior entry.\n"
            f"    ({ts_int:_}, FeeSchedule(\n"
            f"        protocol_bps={sched.protocol_bps}, creator_bps={sched.creator_bps},"
            f" pumpswap_lp_bps={sched.pumpswap_lp_bps},"
            f" pumpswap_protocol_bps={sched.pumpswap_protocol_bps},\n"
            f"        slippage={sched.slippage}, mev_penalty={sched.mev_penalty},\n"
            f"    )),\n"
        )
        marker = "    )),\n]"
        idx = text.rfind(marker)
        if idx < 0:
            return False
        insert_pos = idx + len("    )),\n")
        config_path.write_text(text[:insert_pos] + new_entry + text[insert_pos:])
        return True
    except Exception:
        return False


async def check_live_fee_regime(
    rpc_url: str,
    *,
    pump_constants: ProtocolConstants | None = None,
    auto_append: bool = False,
    effective_from: float | None = None,
) -> str | None:
    """Compare on-chain fees to the latest _REGIMES entry; warn on mismatch.

    Reads the Pump Global account and (optionally) the PumpSwap GlobalConfig
    from ``rpc_url`` and compares live fee values to ``_REGIMES[-1]``. Returns
    a warning string on mismatch, None if fees agree or the check itself fails
    (errors are non-blocking — startup must never hang on this).

    pump_constants: pass the already-fetched ProtocolConstants from
        load_protocol_constants() to avoid a redundant RPC round-trip.

    auto_append: if True and fees differ, appends a new FeeSchedule to
        _REGIMES in memory *and* writes config.py on disk. Set effective_from
        to the actual on-chain change date before backtesting across the
        boundary; if omitted it defaults to time.time().
    """
    import time as _time
    import httpx

    # Reuse already-fetched pump constants if provided; otherwise re-fetch.
    pc = pump_constants
    swap_raw: bytes | None = None

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            if pc is None:
                r = await client.post(
                    rpc_url,
                    json={
                        "jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
                        "params": [PUMP_GLOBAL_ACCOUNT, {"encoding": "base64", "commitment": "confirmed"}],
                    },
                )
                v = r.json()["result"]["value"]
                if v is None:
                    return None
                raw = base64.b64decode(v["data"][0] if isinstance(v["data"], list) else v["data"])
                pc = decode_global_account(raw)

            r2 = await client.post(
                rpc_url,
                json={
                    "jsonrpc": "2.0", "id": 2, "method": "getAccountInfo",
                    "params": [PUMP_SWAP_GLOBAL_CONFIG, {"encoding": "base64", "commitment": "confirmed"}],
                },
            )
            v2 = r2.json()["result"]["value"]
            if v2 is not None:
                raw2 = v2["data"][0] if isinstance(v2["data"], list) else v2["data"]
                swap_raw = base64.b64decode(raw2)
    except Exception:
        return None  # non-blocking — don't surface RPC errors at startup

    _, latest = _REGIMES[-1]

    pump_mismatch = (
        pc.fee_basis_points != latest.protocol_bps
        or pc.creator_fee != latest.creator_bps
    )
    swap_lp_bps: int | None = None
    swap_proto_bps: int | None = None
    if swap_raw is not None:
        decoded = _decode_pumpswap_config(swap_raw)
        if decoded is not None:
            swap_lp_bps, swap_proto_bps = decoded
    swap_mismatch = swap_lp_bps is not None and (
        swap_lp_bps != latest.pumpswap_lp_bps
        or swap_proto_bps != latest.pumpswap_protocol_bps
    )

    if not pump_mismatch and not swap_mismatch:
        return None

    parts: list[str] = []
    if pump_mismatch:
        parts.append(
            f"pump: on-chain {pc.fee_basis_points}+{pc.creator_fee} bps, "
            f"_REGIMES has {latest.protocol_bps}+{latest.creator_bps} bps"
        )
    if swap_mismatch:
        parts.append(
            f"pumpswap: on-chain {swap_lp_bps}+{swap_proto_bps} bps, "
            f"_REGIMES has {latest.pumpswap_lp_bps}+{latest.pumpswap_protocol_bps} bps"
        )
    diff = "; ".join(parts)

    if not auto_append:
        return (
            f"WARNING: on-chain fees differ from latest _REGIMES entry: {diff}. "
            f"Append a new (ts, FeeSchedule(...)) to _REGIMES in config.py."
        )

    eff_ts = effective_from if effective_from is not None else _time.time()
    new_sched = FeeSchedule(
        protocol_bps=pc.fee_basis_points,
        creator_bps=pc.creator_fee,
        pumpswap_lp_bps=swap_lp_bps if swap_lp_bps is not None else latest.pumpswap_lp_bps,
        pumpswap_protocol_bps=swap_proto_bps if swap_proto_bps is not None else latest.pumpswap_protocol_bps,
        slippage=latest.slippage,
        mev_penalty=latest.mev_penalty,
    )
    _REGIMES.append((eff_ts, new_sched))
    persisted = _append_regime_entry_to_file(eff_ts, new_sched)

    return (
        f"NOTICE: on-chain fees differ ({diff}); auto-appended new regime "
        f"effective_from={int(eff_ts)} "
        f"({'written to config.py' if persisted else 'config.py write failed — add manually'}). "
        f"Correct effective_from to the actual on-chain change date before backtesting across the boundary."
    )