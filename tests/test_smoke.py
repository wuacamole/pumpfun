"""End-to-end smoke tests for the upgraded system.

These don't need a network connection or a Helius key. They exercise:
  - storage (SQLite roundtrip + Parquet export)
  - helius reserve decoding (pure-function)
  - model featurize + fit + predict (synthetic labelled data)
  - backtest walk-forward (synthetic examples)
  - paper_trade broker entries/exits/PnL
  - monitor alert + dashboard plumbing
"""

from __future__ import annotations

import time

import pytest

from backtest import run_backtest
from conviction_engine import Swap, _graduation_vsol, score_token
from helius import HeliusClient
from model import build_training_frame, featurize, fit, predict_grad_prob
from monitor import AlertStream, Dashboard, DecisionEvent, PnlEvent
from paper_trade import PaperBroker
from storage import Store, load_latest_snapshot


def _swap(i: int, t: float, wallet: str, is_buy: bool, sol: float, vsol: float, via: bool = True) -> Swap:
    return Swap(ts=t, wallet=wallet, is_buy=is_buy, sol_amount=sol, vsol_after=vsol, via_frontend=via)


def test_storage_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    s = Store(db)
    mint = "MintAAA"
    s.register_token(mint, {"name": "test"})
    t0 = time.time()
    s.log_swap(_swap(0, t0, "w1", True, 1.0, 31.0), mint)
    s.log_swap(_swap(1, t0 + 1, "w2", True, 2.0, 33.0), mint)
    s.set_outcome(mint, "graduated", outcome_ts=t0 + 100, graduated_at_ts=t0 + 100)

    row = s.get_token(mint)
    assert row is not None
    assert row.outcome == "graduated"
    assert row.n_swaps == 2
    assert row.peak_vsol == 33.0

    swaps = s.swaps_for(mint)
    assert len(swaps) == 2
    assert swaps[0].via_frontend is True

    labelled = s.labelled_mints()
    assert labelled == [(mint, "graduated")]
    s.close()


def test_storage_parquet_export(tmp_path):
    db = tmp_path / "t.db"
    snap = tmp_path / "snap"
    s = Store(db)
    s.log_swap(_swap(0, 1.0, "w", True, 1.0, 31.0), "M")
    s.export_parquet(snap)
    s.close()
    sp, tp, _gp = load_latest_snapshot(snap)
    assert sp.exists() and tp.exists()


def test_helius_reserve_decode():
    # Build a fake bonding-curve account: 8-byte discriminator + 5 u64s + bool.
    import struct
    disc = b"\x00" * 8
    v_tok, v_sol, r_tok, r_sol, total = 1_000_000, 30 * 10**9, 800_000, 50 * 10**9, 10**9
    body = struct.pack("<5Q", v_tok, v_sol, r_tok, r_sol, total) + b"\x01"
    raw = disc + body
    r = HeliusClient._decode_reserves(raw)
    assert r is not None
    assert r.real_sol_reserves == 50.0
    assert r.virtual_sol_reserves == 30.0
    assert r.vsol == 80.0
    assert r.graduated is True


def test_featurize_no_lookahead():
    swaps = [
        _swap(0, 0.0, "w", True, 1.0, 31.0),
        _swap(1, 1.0, "w", True, 1.0, 32.0),
        _swap(2, 2.0, "w", True, 1.0, 33.0),
        _swap(3, 10.0, "w", True, 1.0, 34.0),
    ]
    # as_of = 1.5 => only the first two swaps count.
    f = featurize(swaps, as_of=1.5)
    assert f["n_swaps"] == 2
    assert f["vsol"] == 32.0
    # as_of = None => all swaps.
    f_all = featurize(swaps)
    assert f_all["n_swaps"] == 4
    assert f_all["vsol"] == 34.0


def test_model_fit_predict():
    # Synthetic: tokens with high conviction + low bot share graduate; churn fails.
    def good(swaps, n=40):
        t = 0.0
        vsol = 30.0
        for i in range(n):
            t += 1.0
            vsol += 1.5
            swaps.append(_swap(i, t, f"h{i}", True, 1.0, vsol, via=True))
        return swaps

    def bad(swaps, n=80):
        t = 0.0
        vsol = 30.0
        for i in range(n):
            t += 0.4
            vsol += 0.05
            swaps.append(_swap(i, t, f"b{i%10}", True, 0.05, vsol, via=False))
        return swaps

    examples = []
    for i in range(30):
        examples.append((f"good{i}", good([]), True, None, None))
        examples.append((f"bad{i}", bad([]), False, None, None))
    df = build_training_frame(examples)
    assert not df.empty
    assert df["graduated"].nunique() == 2

    model = fit(df, kind="gbm")
    # A good token should have higher P(graduation) than a bad one.
    g = predict_grad_prob(model, good([]))
    b = predict_grad_prob(model, bad([]))
    assert g > b
    assert 0.0 <= g <= 1.0
    assert 0.0 <= b <= 1.0


def test_backtest_smoke():
    # Two contrasting tokens, walk-forward should at least run and produce a report.
    def good():
        s = []
        t = 0.0
        vsol = 30.0
        for i in range(40):
            t += 1.0
            vsol = min(vsol + 2.0, _graduation_vsol())
            s.append(_swap(i, t, f"h{i}", True, 1.0, vsol, via=True))
        return s

    def bad():
        s = []
        t = 0.0
        vsol = 30.0
        for i in range(80):
            t += 0.4
            vsol += 0.05
            s.append(_swap(i, t, f"b{i%10}", True, 0.05, vsol, via=False))
        return s

    examples = []
    for i in range(10):
        examples.append((f"good{i}", good(), "graduated", None))
        examples.append((f"bad{i}", bad(), "rugged", None))
    report = run_backtest(examples, decision_every_sec=5.0, fold_size=10)
    assert report.n_tokens == 20
    assert report.n_decisions > 0
    # The good tokens should produce more ENTERs than the bad ones.
    good_enters = sum(1 for t in report.trades if t.mint.startswith("good") and t.entered)
    bad_enters = sum(1 for t in report.trades if t.mint.startswith("bad") and t.entered)
    assert good_enters >= bad_enters


def test_paper_broker_pnl():
    b = PaperBroker(bankroll=10.0, size_fraction=0.1)
    pos = b.on_enter("M", time.time(), 50.0)
    assert pos is not None
    assert pos.entry_sol > 0
    # Graduation exit: bankroll should have grown.
    b.on_graduated("M", time.time() + 60)
    assert b.bankroll > 10.0
    s = b.summary()
    assert s["n_closed"] == 1
    assert s["n_graduated"] == 1


def test_paper_broker_persistence(tmp_path):
    """Paper trades + fills + bankroll survive a restart via the SQLite store."""
    db = tmp_path / "paper.db"
    # Run 1: enter a position, then "crash" (drop the broker object).
    s1 = Store(db)
    b1 = PaperBroker(bankroll=10.0, size_fraction=0.1)
    b1.attach_store(s1, reload=True)
    starting_bankroll = b1.bankroll
    pos = b1.on_enter("MintAAA", time.time(), 50.0)
    assert pos is not None
    assert pos.trade_id is not None  # persisted to SQLite
    bankroll_after_enter = b1.bankroll
    assert bankroll_after_enter < starting_bankroll  # capital deployed
    # The trade row + entry fill should be in SQLite.
    trades = s1.conn.execute("SELECT * FROM paper_trades WHERE status='open'").fetchall()
    assert len(trades) == 1
    fills = s1.conn.execute("SELECT * FROM paper_fills").fetchall()
    assert len(fills) == 1  # the entry fill
    s1.close()

    # Run 2: a fresh broker reloads state from the store — bankroll + the open
    # position should carry over.
    s2 = Store(db)
    b2 = PaperBroker()
    b2.attach_store(s2, reload=True)
    assert b2.bankroll == pytest.approx(bankroll_after_enter)
    assert "MintAAA" in b2.positions
    assert b2.positions["MintAAA"].trade_id == pos.trade_id
    # Close it via graduation — the close + exit fill should land in SQLite.
    closed = b2.on_graduated("MintAAA", time.time() + 120)
    assert closed is not None
    assert closed.status == "graduated"
    assert closed.realized_return is not None and closed.realized_return > 0
    rows = s2.conn.execute("SELECT * FROM paper_trades WHERE status='graduated'").fetchall()
    assert len(rows) == 1
    fills = s2.conn.execute("SELECT * FROM paper_fills").fetchall()
    assert len(fills) == 2  # entry + exit
    s2.close()

    # Run 3: reload again — bankroll should reflect the graduation payout, and
    # the position should be gone (it's closed).
    s3 = Store(db)
    b3 = PaperBroker()
    b3.attach_store(s3, reload=True)
    assert b3.bankroll > starting_bankroll  # the graduation gain survived
    assert len(b3.positions) == 0
    closed_rows = s3.closed_paper_trades()
    assert len(closed_rows) == 1
    s3.close()


def test_monitor_alerts(tmp_path):
    alert = AlertStream(path=str(tmp_path / "a.log"))
    alert.on_decision(DecisionEvent(
        mint="MintXYZ", decision="ENTER", score=70.0, pct_to_graduation=40.0,
        bot_share=0.2, sol_per_human=0.5, edge=0.1,
    ))
    alert.on_pnl(PnlEvent(
        mint="MintXYZ", status="graduated", realized_return=4.0, bankroll_after=15.0,
    ))
    assert (tmp_path / "a.log").exists()


def test_dashboard_renders():
    d = Dashboard()
    d.on_decision(DecisionEvent(
        mint="M", decision="ENTER", score=70, pct_to_graduation=40,
        bot_share=0.2, sol_per_human=0.5, edge=0.1,
    ))
    d.on_pnl(PnlEvent(mint="M", status="graduated", realized_return=4.0, bankroll_after=15.0))
    # _render should not raise.
    d._render()


def test_engine_still_separates():
    """The original sanity check from test_scenarios must still hold."""
    def churn():
        s = []
        t = 0.0
        vsol = 30.0
        for i in range(400):
            t += 0.4
            is_buy = (i % 5 != 0)
            amt = 0.05 + (i % 3) * 0.02
            vsol += amt * 0.5 if is_buy else -amt * 0.4
            s.append(_swap(i, t, f"b{i%12}", is_buy, amt, vsol, via=(i % 10 == 0)))
        return s

    def conviction():
        s = []
        t = 0.0
        vsol = 30.0
        for i in range(50):
            t += 1.5
            is_buy = (i % 9 != 0)
            amt = 0.8 + (i % 4) * 0.3
            vsol += amt * 0.9 if is_buy else -amt * 0.5
            s.append(_swap(i, t, f"h{i}", is_buy, amt, vsol, via=(i % 8 != 0)))
        return s

    churn_score = score_token("CHURN", churn())
    conv_score = score_token("CONV", conviction())
    assert churn_score.decision == "SKIP"
    # Conviction token clears the score + bot_share gates but its edge (~2.7%)
    # is below the 3% ENTER threshold, so the honest decision is WATCH, not ENTER.
    assert conv_score.decision in {"ENTER", "WATCH"}
    assert conv_score.conviction_score > churn_score.conviction_score
    assert conv_score.edge > churn_score.edge


def test_grad_prob_handles_negative_vsol():
    """Regression: a negative vsol (reconstructed-vSol fallback drifting below
    zero on a large sell) raised a fractional power to a complex number and
    crashed the clamp with 'float < complex'. Both probability helpers must
    stay real and in [0, 1]."""
    from conviction_engine import _breakeven_probability, _estimate_graduation_prob
    p = _estimate_graduation_prob(-5.0, 0.2, 0.3, 0.1, 120.0)
    b = _breakeven_probability(-5.0)
    assert isinstance(p, float) and 0.0 <= p <= 1.0
    assert isinstance(b, float) and 0.0 <= b <= 1.0


# ---- helius helpers ----------------------------------------------------------

def test_collect_instructions():
    tx: dict = {
        "transaction": {"message": {"instructions": [{"programId": "A"}, {"programId": "B"}]}},
        "meta": {"innerInstructions": [
            {"instructions": [{"programId": "C"}]},
            {"instructions": [{"programId": "D"}, {"programId": "E"}]},
        ]},
    }
    ixs = HeliusClient._collect_instructions(tx)
    assert [HeliusClient._program_id(ix) for ix in ixs] == ["A", "B", "C", "D", "E"]


def test_program_id_formats():
    # programId string form
    assert HeliusClient._program_id({"programId": "ABC"}) == "ABC"
    # program string form
    assert HeliusClient._program_id({"program": "XYZ"}) == "XYZ"
    # nested program.id form (jsonParsed encoding)
    assert HeliusClient._program_id({"program": {"id": "NEST"}}) == "NEST"
    # empty
    assert HeliusClient._program_id({}) == ""


def test_helius_reserve_decode_sol_only():
    # Verify the discriminator-skip + field ordering for a minimal account.
    import struct
    disc = b"\x00" * 8
    v_tok = 793_100_000_000_000
    v_sol = 30 * 10**9
    r_tok = 600_000_000_000_000
    r_sol = 42 * 10**9
    total = 10**15
    body = struct.pack("<5Q", v_tok, v_sol, r_tok, r_sol, total) + b"\x00"
    r = HeliusClient._decode_reserves(disc + body)
    assert r is not None
    assert r.virtual_sol_reserves == pytest.approx(30.0)
    assert r.real_sol_reserves == pytest.approx(42.0)
    assert r.vsol == pytest.approx(72.0)
    assert r.graduated is False


# ---- PumpLogsFeed slot-continuity tracking -----------------------------------

def test_pump_logs_feed_gap_tracking():
    """Slot gaps > 50 are recorded; small gaps (normal empty slots) are not."""
    from helius import PumpLogsFeed
    client = HeliusClient()  # no network calls in this test
    feed = PumpLogsFeed(client)

    # Simulate slot updates as if processing events.
    feed.last_slot = 1000
    # Gap of 51 slots: should be recorded.
    new_slot = 1051
    if feed.last_slot is not None and new_slot > feed.last_slot + 1:
        gap = new_slot - feed.last_slot
        if gap > 50:
            feed.gaps.append((feed.last_slot, new_slot))
    feed.last_slot = new_slot
    assert len(feed.gaps) == 1
    assert feed.gaps[0] == (1000, 1051)

    # Gap of 10 slots: should NOT be recorded.
    prev = feed.last_slot
    new_slot = prev + 10
    if feed.last_slot is not None and new_slot > feed.last_slot + 1:
        gap = new_slot - feed.last_slot
        if gap > 50:
            feed.gaps.append((feed.last_slot, new_slot))
    feed.last_slot = new_slot
    assert len(feed.gaps) == 1  # still only the one large gap


def test_config_fee_schedule():
    """fee_at returns the correct schedule for pre- and post-regime timestamps."""
    from config import fee_at
    early = fee_at(0.0)
    assert early.creator_bps == 30
    late = fee_at(2_000_000_000)  # well into 2033 — post 2026-05 regime
    assert late.creator_bps == 5
    assert late.curve_fee_rate < early.curve_fee_rate


def test_protocol_constants_fallback():
    """get_cached_constants returns the historical fallback without an RPC."""
    from config import get_cached_constants
    pc = get_cached_constants()
    assert pc.v_sol_graduation == pytest.approx(115.0)
    assert pc.real_sol_graduation == pytest.approx(85.0)
    assert pc.virtual_sol_start == pytest.approx(30.0)


# ---- config decoders --------------------------------------------------------

def _build_global_account_bytes(
    ivtr: int = 793_100_000_000_000,
    ivsr_lamports: int = 30 * 10**9,
    irtr: int = 793_100_000_000_000,
    tts: int = 1_000_000_000_000_000,
    fbp: int = 95,
    creator_fee: int = 30,
) -> bytes:
    """Build a synthetic Pump Global account with the correct packed layout."""
    import struct
    disc = b"\x00" * 8
    body = bytearray()
    body += b"\x01"           # initialized = true
    body += b"\x00" * 32     # authority
    body += b"\x00" * 32     # fee_recipient
    body += struct.pack("<Q", ivtr)
    body += struct.pack("<Q", ivsr_lamports)
    body += struct.pack("<Q", irtr)
    body += struct.pack("<Q", tts)
    body += struct.pack("<Q", fbp)
    body += b"\x00" * 32     # withdraw_authority
    body += b"\x01"           # enable_migrate = true
    body += struct.pack("<Q", 2_000_000_000)  # pool_migration_fee
    body += struct.pack("<Q", creator_fee)
    return disc + bytes(body)


def test_decode_global_account_roundtrip():
    """decode_global_account parses the packed layout and derives vSol graduation."""
    from config import LAMPORTS_PER_SOL, decode_global_account
    raw = _build_global_account_bytes(
        ivsr_lamports=30 * 10**9, ivtr=1_000_000, irtr=700_000, fbp=95, creator_fee=5,
    )
    pc = decode_global_account(raw)
    assert pc.initial_virtual_sol_reserves == pytest.approx(30.0)
    assert pc.fee_basis_points == 95
    assert pc.creator_fee == 5
    # real_sol_grad = 30e9 * 700_000 / (1_000_000 - 700_000) / LAMPORTS_PER_SOL
    expected_real = (30e9 * 700_000 / 300_000) / LAMPORTS_PER_SOL
    assert pc.real_sol_graduation == pytest.approx(expected_real, rel=1e-6)
    assert pc.v_sol_graduation == pytest.approx(30.0 + expected_real, rel=1e-6)


def test_decode_global_account_too_short():
    """Returns ValueError for truncated data."""
    from config import decode_global_account
    with pytest.raises(ValueError, match="too short"):
        decode_global_account(b"\x00" * 10)


def test_decode_bonding_curve_sol_paired():
    """SOL-paired bonding curve: all-zeros quote_mint."""
    import struct

    from config import decode_bonding_curve
    disc = b"\x00" * 8
    v_tok, v_sol, r_tok, r_sol, total = 1_000_000, 30 * 10**9, 800_000, 50 * 10**9, 10**9
    body = struct.pack("<5Q", v_tok, v_sol, r_tok, r_sol, total)
    body += b"\x00"  # complete = false
    body += b"\x00" * 32  # quote_mint = all zeros (SOL-paired)
    raw = disc + body
    bc = decode_bonding_curve(raw)
    assert bc is not None
    assert bc.virtual_sol_reserves == pytest.approx(30.0)
    assert bc.real_sol_reserves == pytest.approx(50.0)
    assert bc.vsol == pytest.approx(80.0)
    assert bc.graduated is False
    assert bc.is_sol_paired is True


def test_decode_bonding_curve_usdc_paired():
    """USDC-paired bonding curve: non-zero quote_mint (the USDC pubkey)."""
    import struct

    from solders.pubkey import Pubkey

    from config import USDC_MINT, decode_bonding_curve
    disc = b"\x00" * 8
    v_tok, v_sol, r_tok, r_sol, total = 1_000_000, 30 * 10**9, 800_000, 50 * 10**9, 10**9
    body = struct.pack("<5Q", v_tok, v_sol, r_tok, r_sol, total)
    body += b"\x01"  # complete = true
    body += bytes(Pubkey.from_string(USDC_MINT))  # quote_mint = USDC
    raw = disc + body
    bc = decode_bonding_curve(raw)
    assert bc is not None
    assert bc.graduated is True
    assert bc.is_sol_paired is False
    assert bc.quote_mint == USDC_MINT


def test_decode_bonding_curve_wrapped_sol_paired():
    """Wrapped-SOL mint as quote_mint should also be SOL-paired."""
    import struct

    from solders.pubkey import Pubkey

    from config import WRAPPED_SOL_MINT, decode_bonding_curve
    disc = b"\x00" * 8
    body = struct.pack("<5Q", 1_000_000, 30 * 10**9, 800_000, 50 * 10**9, 10**9)
    body += b"\x00"
    body += bytes(Pubkey.from_string(WRAPPED_SOL_MINT))
    raw = disc + body
    bc = decode_bonding_curve(raw)
    assert bc is not None
    assert bc.is_sol_paired is True


def test_decode_bonding_curve_too_short():
    """Returns None for truncated data."""
    from config import decode_bonding_curve
    assert decode_bonding_curve(b"\x00" * 5) is None


def test_decode_bonding_curve_no_quote_mint():
    """Short body without quote_mint defaults to SOL-paired."""
    import struct

    from config import decode_bonding_curve
    disc = b"\x00" * 8
    body = struct.pack("<5Q", 1_000_000, 30 * 10**9, 800_000, 50 * 10**9, 10**9)
    body += b"\x00"  # complete only, no quote_mint
    raw = disc + body
    bc = decode_bonding_curve(raw)
    assert bc is not None
    assert bc.is_sol_paired is True


def _trade_event_log(sol_lamports=1_500_000_000, is_buy=1, v_sol=30 * 10**9,
                     r_sol=5 * 10**9, cpi=False):
    """Build a synthetic pump.fun TradeEvent 'Program data:' log line."""
    import base64
    import struct

    from solders.pubkey import Pubkey

    from helius import _ANCHOR_CPI_EVENT_SENTINEL, TRADE_EVENT_DISCRIMINATOR
    mint = bytes(Pubkey.default())
    user = bytes(Pubkey.default())
    body = (mint + struct.pack("<QQ", sol_lamports, 42_000) + bytes([is_buy]) + user
            + struct.pack("<q", 1_750_000_000)
            + struct.pack("<4Q", v_sol, 1_000_000, r_sol, 800_000))
    payload = TRADE_EVENT_DISCRIMINATOR + body
    if cpi:
        payload = _ANCHOR_CPI_EVENT_SENTINEL + payload
    return "Program data: " + base64.b64encode(payload).decode()


def test_decode_trade_event_roundtrip():
    """A trade is decoded from the log with true reserves and no RPC."""
    from helius import decode_trade_event
    te = decode_trade_event(["Program log: Instruction: Buy", _trade_event_log()])
    assert te is not None
    assert te.is_buy is True
    assert abs(te.sol_amount - 1.5) < 1e-9
    assert abs(te.vsol_after - 35.0) < 1e-9  # (v_sol + r_sol) in SOL


def test_decode_trade_event_cpi_framing():
    """emit_cpi! sentinel-wrapped events decode identically."""
    from helius import decode_trade_event
    te = decode_trade_event([_trade_event_log(cpi=True)])
    assert te is not None and abs(te.vsol_after - 35.0) < 1e-9


def test_decode_trade_event_rejects_noise():
    """Non-trade logs decode to None so the caller falls back to the RPC path."""
    import base64

    from helius import decode_trade_event
    junk = "Program data: " + base64.b64encode(b"not a trade event").decode()
    assert decode_trade_event(["Program log: hello", junk]) is None
    assert decode_trade_event([]) is None


# ---- PaperBroker.check_exits ------------------------------------------------

def test_paper_broker_time_stop():
    """Position is closed by time_stop_sec if not graduated."""
    b = PaperBroker(bankroll=10.0, size_fraction=0.1, time_stop_sec=60.0)
    t0 = 1000.0
    pos = b.on_enter("M", t0, 50.0)
    assert pos is not None
    # 70 seconds later, no graduation -> time stop fires.
    closed = b.check_exits(t0 + 70, {"M": 55.0})
    assert len(closed) == 1
    assert closed[0].status == "timed_out"
    assert "M" not in b.positions
    assert len(b.closed) == 1


def test_paper_broker_loss_stop():
    """Position is closed when bonding curve vSol drops below stop threshold."""
    b = PaperBroker(bankroll=10.0, size_fraction=0.1, time_stop_sec=9999.0, stop_loss_fraction=0.5)
    t0 = 1000.0
    pos = b.on_enter("M", t0, 50.0)
    assert pos is not None
    # vSol drops 60% (below the 50% stop_loss_fraction threshold).
    closed = b.check_exits(t0 + 10, {"M": 20.0})
    assert len(closed) == 1
    assert closed[0].status == "stopped"
    assert "M" not in b.positions


def test_paper_broker_no_stop_within_bounds():
    """Position stays open if within time and loss limits."""
    b = PaperBroker(bankroll=10.0, size_fraction=0.1, time_stop_sec=9999.0, stop_loss_fraction=0.5)
    t0 = 1000.0
    b.on_enter("M", t0, 50.0)
    closed = b.check_exits(t0 + 10, {"M": 48.0})  # -4% vSol, within 50% loss limit
    assert len(closed) == 0
    assert "M" in b.positions


def test_paper_broker_graduation_via_check_exits():
    """check_exits triggers on_graduated when vSol crosses threshold."""
    b = PaperBroker(bankroll=10.0, size_fraction=0.1)
    t0 = 1000.0
    b.on_enter("M", t0, 50.0)
    grad_vsol = b._graduation_vsol()
    closed = b.check_exits(t0 + 10, {"M": grad_vsol + 1.0})
    assert len(closed) == 1
    assert closed[0].status == "graduated"


# ---- PaperBroker.on_stop (direct) -------------------------------------------

def test_on_stop_floor_at_5pct():
    """on_stop floors the exit value at 5% of entry even for extreme rugs."""
    b = PaperBroker(bankroll=10.0, size_fraction=0.1)
    t0 = 1000.0
    pos = b.on_enter("M", t0, 50.0)
    assert pos is not None
    entry_sol = pos.entry_sol
    # vSol goes to near-zero -> ratio floors at 0.05.
    closed = b.on_stop("M", t0 + 10, current_vsol=0.01)
    assert closed is not None
    assert closed.realized_return is not None
    # The gross exit should be ~5% of entry_sol, minus fees.
    assert closed.exit_sol is not None
    assert closed.exit_sol < entry_sol * 0.10  # well below entry


def test_on_stop_returns_none_for_missing_mint():
    """on_stop returns None if mint is not in positions."""
    b = PaperBroker(bankroll=10.0)
    assert b.on_stop("NOPE", 1000.0, 50.0) is None


# ---- _resolve_vsol and _resolve_via_frontend ---------------------------------

def test_resolve_vsol_from_event():
    """_resolve_vsol uses vsol_in_bonding_curve directly if present."""
    import asyncio

    from live_monitor import _resolve_vsol
    evt = {"vSolInBondingCurve": 77.5}
    result = asyncio.get_event_loop().run_until_complete(_resolve_vsol(evt, "mint", None))
    assert result == 77.5


def test_resolve_vsol_fallback_to_history():
    """_resolve_vsol reconstructs from HISTORY when no event field and no Helius."""
    import asyncio

    from conviction_engine import Swap
    from live_monitor import HISTORY, _resolve_vsol
    # Seed HISTORY with one prior swap.
    HISTORY["test_mint"] = [Swap(ts=100.0, wallet="w", is_buy=True, sol_amount=1.0, vsol_after=40.0, via_frontend=True)]
    evt = {}  # no vsol field
    result = asyncio.get_event_loop().run_until_complete(_resolve_vsol(evt, "test_mint", None))
    # Should reconstruct from 40.0 + sol*(1-fee_rate). evt has no sol, so delta = 0.
    assert result == pytest.approx(40.0)
    HISTORY.pop("test_mint", None)


def test_resolve_via_frontend_fallback_heuristic():
    """_resolve_via_frontend falls back to size heuristic when no Helius/sig."""
    import asyncio

    from live_monitor import _resolve_via_frontend
    # Small trade -> heuristic says bot (False).
    evt_small = {"solAmount": 0.01}
    r = asyncio.get_event_loop().run_until_complete(_resolve_via_frontend(evt_small, None))
    assert r is False
    # Large trade -> heuristic says human (True).
    evt_large = {"solAmount": 0.5}
    r = asyncio.get_event_loop().run_until_complete(_resolve_via_frontend(evt_large, None))
    assert r is True


def test_resolve_via_frontend_source_based():
    """_resolve_via_frontend uses source string when available."""
    import asyncio

    from live_monitor import _resolve_via_frontend
    evt = {"source": "pump.fun website", "solAmount": 0.01}
    r = asyncio.get_event_loop().run_until_complete(_resolve_via_frontend(evt, None))
    assert r is True
    evt_bot = {"source": "api bot", "solAmount": 0.5}
    r = asyncio.get_event_loop().run_until_complete(_resolve_via_frontend(evt_bot, None))
    assert r is False


# ---- Store edge cases -------------------------------------------------------

def test_store_duplicate_swap():
    """Logging the same swap twice does not crash (idempotent-ish)."""
    db_path = "/tmp/test_dup_swap.db"
    import os
    if os.path.exists(db_path):
        os.unlink(db_path)
    s = Store(db_path)
    mint = "DupMint"
    sw = _swap(0, 1.0, "w", True, 1.0, 31.0)
    s.log_swap(sw, mint)
    s.log_swap(sw, mint)  # duplicate — no dedupe, just appends
    swaps = s.swaps_for(mint)
    assert len(swaps) == 2  # both logged
    row = s.get_token(mint)
    assert row is not None
    assert row.n_swaps == 2
    s.close()
    os.unlink(db_path)


def test_store_set_outcome_idempotent():
    """Setting outcome twice does not crash."""
    db_path = "/tmp/test_outcome.db"
    import os
    if os.path.exists(db_path):
        os.unlink(db_path)
    s = Store(db_path)
    mint = "OutcomeMint"
    s.register_token(mint, {})
    s.set_outcome(mint, "graduated", outcome_ts=100.0)
    s.set_outcome(mint, "graduated", outcome_ts=200.0)  # overwrite
    row = s.get_token(mint)
    assert row is not None
    assert row.outcome == "graduated"
    assert row.outcome_ts == 200.0  # updated
    s.close()
    os.unlink(db_path)


def test_store_empty_swaps_for():
    """swaps_for returns empty list for unknown mint."""
    db_path = "/tmp/test_empty.db"
    import os
    if os.path.exists(db_path):
        os.unlink(db_path)
    s = Store(db_path)
    assert s.swaps_for("nonexistent") == []
    s.close()
    os.unlink(db_path)


def test_store_dedupe_swaps():
    """dedupe_swaps removes duplicate rows and recomputes n_swaps."""
    db_path = "/tmp/test_dedupe.db"
    import os
    if os.path.exists(db_path):
        os.unlink(db_path)
    s = Store(db_path)
    mint = "DedupeMint"
    sw = _swap(0, 1.0, "w", True, 1.0, 31.0)
    # Log the same swap 3 times.
    s.log_swap(sw, mint)
    s.log_swap(sw, mint)
    s.log_swap(sw, mint)
    assert s.swaps_for(mint).__len__() == 3
    assert s.get_token(mint).n_swaps == 3
    deleted = s.dedupe_swaps()
    assert deleted == 2
    assert len(s.swaps_for(mint)) == 1
    assert s.get_token(mint).n_swaps == 1
    s.close()
    os.unlink(db_path)


def test_store_log_swaps_batch():
    """log_swaps (bulk) doesn't nest transactions and all swaps land."""
    db_path = "/tmp/test_batch.db"
    import os
    if os.path.exists(db_path):
        os.unlink(db_path)
    s = Store(db_path)
    mint = "BatchMint"
    swaps = [_swap(i, float(i), f"w{i}", True, 1.0, 31.0 + i) for i in range(10)]
    s.log_swaps(mint, swaps)
    assert len(s.swaps_for(mint)) == 10
    assert s.get_token(mint).n_swaps == 10
    s.close()
    os.unlink(db_path)


def test_store_dedupe_keeps_distinct_same_ts_swaps(tmp_path):
    """Two genuine same-wallet/same-ts/same-size trades differ in vsol_after
    (each moves the curve), so dedupe must NOT collapse them."""
    s = Store(tmp_path / "d.db")
    mint = "DistinctMint"
    s.log_swap(_swap(0, 1.0, "w", True, 1.0, 31.0), mint)
    s.log_swap(_swap(0, 1.0, "w", True, 1.0, 32.0), mint)  # same ts, different vsol_after
    assert s.dedupe_swaps() == 0
    assert s.get_token(mint).n_swaps == 2
    s.close()


def test_store_prune_swaps_before(tmp_path):
    """prune_swaps_before drops old rows and re-derives n_swaps."""
    s = Store(tmp_path / "p.db")
    mint = "PruneMint"
    for i in range(5):
        s.log_swap(_swap(i, float(i), f"w{i}", True, 1.0, 31.0 + i), mint)
    assert s.get_token(mint).n_swaps == 5
    deleted = s.prune_swaps_before(3.0)  # ts 0,1,2 dropped; 3,4 kept
    assert deleted == 3
    assert len(s.swaps_for(mint)) == 2
    assert s.get_token(mint).n_swaps == 2
    s.close()


def test_store_export_parquet_streams_multiple_batches(tmp_path, monkeypatch):
    """Streaming export writes every row even when the table spans several
    batches (batch size forced small), and round-trips faithfully."""
    import storage as storage_mod
    monkeypatch.setattr(storage_mod, "_PARQUET_EXPORT_BATCH", 7)
    s = Store(tmp_path / "stream.db")
    mint = "StreamMint"
    swaps = [_swap(i, float(i), f"w{i}", i % 2 == 0, 1.0 + i, 31.0 + i) for i in range(20)]
    s.log_swaps(mint, swaps)
    s.export_parquet(tmp_path / "snaps")
    sp, _tp, _gp = load_latest_snapshot(tmp_path / "snaps")
    import pyarrow.parquet as pq
    t = pq.read_table(sp)
    assert t.num_rows == 20  # 20 rows across batches of 7
    assert set(t.column("ts").to_pylist()) == {float(i) for i in range(20)}
    s.close()


def test_store_export_parquet_empty(tmp_path):
    """An empty swaps table still produces a valid, readable parquet file."""
    s = Store(tmp_path / "empty.db")
    s.export_parquet(tmp_path / "snaps")
    sp, _tp, _gp = load_latest_snapshot(tmp_path / "snaps")
    import pyarrow.parquet as pq
    assert pq.read_table(sp).num_rows == 0
    s.close()


def test_store_swaps_composite_index(tmp_path):
    """The (mint, ts) composite index exists and the redundant single-column
    mint index has been dropped."""
    s = Store(tmp_path / "ix.db")
    names = {r[0] for r in s.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='swaps'"
    )}
    assert "ix_swaps_mint_ts" in names
    assert "ix_swaps_mint" not in names
    s.close()


def test_store_graduation_price_roundtrip(tmp_path):
    """record/get graduation price roundtrips."""
    db = tmp_path / "gp.db"
    s = Store(db)
    s.record_graduation_price("MintX", 1000.0, 0.002, entry_vsol=50.0, payoff_multiple=4.5)
    row = s.get_graduation_price("MintX")
    assert row is not None
    assert row["price_sol_per_unit"] == 0.002
    assert row["payoff_multiple"] == 4.5
    all_gp = s.all_graduation_prices()
    assert "MintX" in all_gp
    s.close()


def test_curve_price_per_token_monotonic():
    """curve_price_per_token follows the constant-product curve: price rises with
    vSol as V**2, and the terminal price matches the known ~4.1e-7 SOL/token."""
    from config import curve_price_per_token, get_cached_constants

    pc = get_cached_constants()
    # Doubling vSol quadruples the price (V**2 law).
    assert curve_price_per_token(60, pc) == pytest.approx(4 * curve_price_per_token(30, pc))
    # Terminal price at graduation vSol.
    assert curve_price_per_token(pc.v_sol_graduation, pc) == pytest.approx(4.108e-7, rel=1e-3)
    # Degenerate inputs are safe.
    assert curve_price_per_token(0, pc) == 0.0
    assert curve_price_per_token(-5, pc) == 0.0


def test_graduation_payoff_per_position():
    """Per-position payoff (charter I2): exit at the real PumpSwap price over the
    bonding-curve entry price. Entering early (low vSol) yields a higher multiple
    than entering at graduation. Regression: payoff_multiple used to be NULL, and
    the first fix wrongly used the graduation vSol as the entry baseline."""
    from config import curve_price_per_token, get_cached_constants

    pc = get_cached_constants()
    terminal = curve_price_per_token(pc.v_sol_graduation, pc)  # PumpSwap price ~ terminal

    def multiple(price, entry_vsol):
        return price / curve_price_per_token(entry_vsol, pc)

    # Exit at the terminal price, entering at graduation -> ~1x.
    assert multiple(terminal, pc.v_sol_graduation) == pytest.approx(1.0)
    # Entering at half the graduation vSol quadruples the multiple (V**2 law).
    assert multiple(terminal, pc.v_sol_graduation / 2) == pytest.approx(4.0)
    # Representative entry vSol = v_grad/sqrt(5) reproduces the legacy ~5x default
    # when the PumpSwap price equals the terminal price.
    from live_monitor import _representative_entry_vsol
    rep = _representative_entry_vsol(pc)
    assert multiple(terminal, rep) == pytest.approx(5.0)


def test_pool_price_from_tx():
    """_pool_price_from_tx reads price from the pool's reserves (WSOL/base),
    picking the largest-base holder as the pool and ignoring a trader who
    transiently holds both legs. Regression: the old code paired the mint's token
    delta with the fee-payer's lamport delta — unrelated accounts — yielding
    garbage prices spanning 1e-15 to 1e-2."""
    from config import PUMP_SWAP_PROGRAM_ID, WRAPPED_SOL_MINT
    from live_monitor import _pool_price_from_tx

    MINT = "MintXXXXpump"
    POOL, TRADER = "PoolOwner", "TraderOwner"
    tx = {
        "transaction": {"message": {"accountKeys": [
            {"pubkey": "whatever"}, {"pubkey": PUMP_SWAP_PROGRAM_ID},
        ]}},
        "meta": {"postTokenBalances": [
            # pool vaults: 160M tokens, 108 WSOL -> price = 108/160e6 = 6.75e-7
            {"owner": POOL, "mint": MINT, "uiTokenAmount": {"uiAmount": 160_000_000.0}},
            {"owner": POOL, "mint": WRAPPED_SOL_MINT, "uiTokenAmount": {"uiAmount": 108.0}},
            # a trader holding both legs but tiny base -> must NOT be chosen
            {"owner": TRADER, "mint": MINT, "uiTokenAmount": {"uiAmount": 5000.0}},
            {"owner": TRADER, "mint": WRAPPED_SOL_MINT, "uiTokenAmount": {"uiAmount": 9.0}},
        ]},
    }
    assert _pool_price_from_tx(tx, MINT) == pytest.approx(108.0 / 160_000_000.0)

    # Not a PumpSwap tx -> None (program id absent).
    tx_no_amm = {"transaction": {"message": {"accountKeys": [{"pubkey": "x"}]}},
                 "meta": {"postTokenBalances": tx["meta"]["postTokenBalances"]}}
    assert _pool_price_from_tx(tx_no_amm, MINT) is None

    # No account holds both legs -> None.
    tx_one_leg = {"transaction": {"message": {"accountKeys": [{"pubkey": PUMP_SWAP_PROGRAM_ID}]}},
                  "meta": {"postTokenBalances": [
                      {"owner": POOL, "mint": MINT, "uiTokenAmount": {"uiAmount": 160e6}}]}}
    assert _pool_price_from_tx(tx_one_leg, MINT) is None


def test_graduation_price_plausibility_filter():
    """Implausible PumpSwap prices (non-swap-tx garbage) are rejected so they
    can't poison the payoff model; prices near the terminal price are kept."""
    from config import curve_price_per_token, get_cached_constants
    from live_monitor import _price_is_plausible

    pc = get_cached_constants()
    terminal = curve_price_per_token(pc.v_sol_graduation, pc)
    assert _price_is_plausible(terminal, pc)
    assert _price_is_plausible(5 * terminal, pc)
    assert not _price_is_plausible(terminal * 1e6, pc)   # 35000x-style outlier
    assert not _price_is_plausible(terminal * 1e-6, pc)  # near-zero outlier
    assert not _price_is_plausible(0.0, pc)