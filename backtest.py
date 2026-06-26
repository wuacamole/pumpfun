"""
backtest.py
===========
Phase 2.3 — the walk-forward backtest harness. This is the moment of truth:
it either shows a real edge or it doesn't. Either answer saves you money.

What it does
------------
Replays logged history chronologically. For each token, at each decision
point (driven by ``decision_every`` seconds after launch), it:

  1. Builds features using only swaps observed up to that moment (no lookahead
     — see :func:`model.featurize`).
  2. Asks the model for P(graduation). On the very first fold, or when no
     fitted model exists yet, it falls back to the hand-built
     :func:`conviction_engine._estimate_graduation_prob`.
  3. Compares to the breakeven probability ``(vSol/115)^2``. Records an ENTER
     / WATCH / SKIP decision.
  4. Resolves the trade's realized return *as of graduation time* (or stall,
     or as of the last logged swap if outcome is unknown).

Walk-forward discipline
-----------------------
The model is *retrained* on a rolling window. Fold ``k`` trains on data from
folds ``< k``, evaluates on fold ``k``. The first fold is always scored with
the hand-built heuristic (zero data to fit on). This is the honest test: the
model never sees the future.

Outputs
-------
  * :class:`BacktestReport` — aggregate metrics:
      - hit_rate of ENTER calls (fraction that graduated)
      - mean realized return per ENTER call (with fees + slippage assumed)
      - calibration: when the model said 40%, did ~40% graduate? (Brier score)
      - edge survival over time (rolling hit rate by fold)
  * :func:`write_report` — dump to JSON for the monitoring dashboard.

This is the single most important deliverable in the roadmap. Run it before
risking capital. Run it again whenever you retrain.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import curve_price_per_token, fee_at
from conviction_engine import (
    Swap,
    _breakeven_probability,
    _estimate_graduation_prob,
    _graduation_vsol,
    score_token,
)
from model import (
    CreatorHistory,
    FittedModel,
    build_training_frame,
    featurize,
    fit,
)

# Execution frictions are now resolved per-trade via fee_at(ts) so a backtest
# across a fee-regime change replays the schedule that was live at each trade.
# The constants below are only the fallback for tests that pass an explicit ts
# of 0 (pre-regime); real backtest trades carry their own decision timestamp.


@dataclass
class TradeOutcome:
    mint: str
    as_of_ts: float
    decision: str           # ENTER | WATCH | SKIP
    est_grad_prob: float
    breakeven_prob: float
    edge: float
    conviction_score: float
    entered: bool
    realized_return: float # post fees/slippage/MEV; NaN if not entered
    graduated: bool
    seconds_to_graduation: float | None
    reasons: list[str] = field(default_factory=list)


@dataclass
class BacktestReport:
    n_tokens: int
    n_decisions: int
    n_enter: int
    n_watch: int
    n_skip: int
    enter_hit_rate: float           # P(graduate | ENTER)
    enter_mean_return: float        # mean realized return per ENTER (post-cost)
    baseline_hit_rate: float        # P(graduate) across all tokens (the random bar)
    brier_score: float              # calibration of est_grad_prob vs actual (all decisions)
    edge_over_time: list[float]     # rolling enter_hit_rate per fold
    trades: list[TradeOutcome] = field(default_factory=list)
    # --- Phase 2 exit-gate diagnostics (charter §5.2) ---
    pr_auc: float = float("nan")           # PR-AUC (primary for <1% positives)
    log_loss: float = float("nan")         # log-loss
    # Calibration on the ENTER subset only (where the gate decision matters).
    brier_score_enter: float = float("nan")
    reliability_slope: float = float("nan")  # predicted vs realized (1.0 = perfect)
    # --- §5.4 capital-gate power & robustness ---
    bootstrap_ci_low: float = float("nan")   # 95% bootstrap CI lower bound on enter_mean_return
    jackknife_positive: bool = False         # positive after deleting the best trade
    n_grad_in_enter: int = 0                  # graduations inside the entered subset
    fold_stability: bool = False              # edge sign positive in every final-window fold


def _bootstrap_ci_low(
    returns: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> float:
    """95% bootstrap CI lower bound on the mean of ``returns``.

    Resamples with replacement, computes the mean each time, returns the
    ``alpha``-quantile (lower bound). Charter §5.4 requires this lower bound
    to be > 0 for the gate to pass.
    """
    if len(returns) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    arr = np.array(returns)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()
    return float(np.quantile(means, alpha))


def _jackknife_positive(returns: list[float]) -> bool:
    """Leave-one-out: is the mean still positive after deleting the best trade?

    Charter §5.4 robustness criterion — guards against a result that depends
    on a single lucky trade (kill criterion K3).
    """
    if len(returns) < 3:
        return False
    arr = np.array(returns)
    best_idx = int(np.argmax(arr))
    loo = np.delete(arr, best_idx)
    return bool(loo.mean() > 0)


def _reliability_slope(
    predicted: list[float],
    realized: list[bool],
    n_bins: int = 5,
) -> float:
    """Reliability slope: OLS slope of (mean predicted) vs (fraction positive)
    across n_bins. 1.0 = perfect calibration. Charter §5.4 requires [0.8, 1.2].
    """
    if len(predicted) < n_bins * 2:
        return float("nan")
    p = np.array(predicted)
    y = np.array([1.0 if r else 0.0 for r in realized])
    # Sort by predicted prob, bin into n_bins, compute per-bin mean pred vs frac pos.
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    bins = np.array_split(np.arange(len(p)), n_bins)
    pred_means = np.array([p_sorted[b].mean() for b in bins])
    real_fracs = np.array([y_sorted[b].mean() for b in bins])
    if pred_means.std() == 0:
        return float("nan")
    slope = float(np.polyfit(pred_means, real_fracs, 1)[0])
    return slope


def _realized_return(
    graduated: bool,
    entry_cost: float,
    exit_cost: float,
    payoff_multiple: float | None = None,
    default_graduation_multiple: float = 5.0,
    default_loss_fraction: float = 0.90,
) -> float:
    """Post-cost realized return for one trade.

    Graduation payoff: the real PumpSwap first-trade price multiple when logged
    (charter I2 — post-cost realized PnL of the *actual* strategy), else the
    conservative flat ``default_graduation_multiple`` (5x), the
    pre-data-collection placeholder. Non-graduation: a conservative flat loss
    (``default_loss_fraction``) until real exit-price data is logged.

    ``payoff_multiple`` is the exit-value multiple on the entry SOL (e.g. 5.0
    means the position was worth 5x entry at graduation). When None, the default
    is used. This is a *rough* proxy for non-graduations — replace with realized
    exit-price data from the log once available.
    """
    if not graduated:
        return -(default_loss_fraction + entry_cost)
    mult = payoff_multiple if payoff_multiple is not None and payoff_multiple > 0 else default_graduation_multiple
    return mult * (1.0 - exit_cost) - (1.0 + entry_cost)


def _resolve_outcome(swaps: list[Swap], label: str) -> tuple[bool, float | None]:
    """Return (graduated, seconds_from_first_swap_to_graduation)."""
    if label == "graduated":
        # find the first swap where vsol crossed GRADUATION_VSOL.
        t0 = swaps[0].ts
        for s in swaps:
            if s.vsol_after >= _graduation_vsol():
                return True, s.ts - t0
        return True, swaps[-1].ts - t0
    return False, None


def run_backtest(
    examples: list[tuple[str, list[Swap], str, CreatorHistory | None]],
    decision_every_sec: float = 30.0,
    fold_size: int = 50,            # retrain every N new labelled tokens
    model_kind: str = "gbm",
    entry_edge_threshold: float = 0.03,
    entry_score_threshold: float = 55.0,
    entry_bot_share_max: float = 0.6,
    graduation_prices: dict[str, dict[str, object]] | None = None,
) -> BacktestReport:
    """Walk-forward backtest.

    ``examples``: ``[(mint, swaps, outcome_label, creator?)]``.
    ``outcome_label`` in {"graduated","stalled","rugged","unknown"}.

    ``graduation_prices``: optional map of mint -> graduation-price row (from
    ``Store.all_graduation_prices``). When present, the graduation payoff uses
    the real PumpSwap first-trade multiple instead of the flat 5x placeholder
    (charter I2). Tokens missing from the map fall back to the default.

    For each token we make a decision at ``t0 + k*decision_every_sec`` while
    the token is still live and un-graduated. Only the *first* ENTER is tracked
    as a trade (one position per token).
    """
    gp = graduation_prices or {}
    # Sort tokens by launch ts so the walk-forward is chronological.
    items = []
    for mint, swaps, label, creator in examples:
        if not swaps:
            continue
        items.append((mint, sorted(swaps, key=lambda s: s.ts), label, creator))
    items.sort(key=lambda x: x[1][0].ts)

    # Rolling training set: (mint, swaps, graduated, creator, as_of) tuples.
    train_buf: list[tuple[str, list[Swap], bool, CreatorHistory | None, float | None]] = []
    model: FittedModel | None = None
    trades: list[TradeOutcome] = []
    edge_over_time: list[float] = []

    for mint, swaps, label, creator in items:
        graduated, _ = _resolve_outcome(swaps, label)
        t0 = swaps[0].ts
        last_ts = swaps[-1].ts
        # Decision points while the token is still pre-graduation.
        k = 1
        entered = False
        while True:
            as_of = t0 + k * decision_every_sec
            if as_of > last_ts or (swaps[-1].vsol_after >= _graduation_vsol() and as_of > swaps[-1].ts):
                break
            seen = [s for s in swaps if s.ts <= as_of]
            if len(seen) < 5:
                k += 1
                continue

            # Model or hand-built fallback for the graduation probability.
            if model is not None:
                feats = featurize(swaps, as_of=as_of, creator=creator)
                est = model.predict_one(feats)
            else:
                # Hand-built estimate — same path the live monitor uses today.
                snap = seen
                vsol = snap[-1].vsol_after
                bot_share = sum(1 for s in snap if not s.via_frontend) / len(snap)
                human_w = {s.wallet for s in snap if s.via_frontend}
                n_human = len(human_w)
                human_net = sum(s.sol_amount if s.is_buy else -s.sol_amount for s in snap if s.via_frontend)
                sph = human_net / n_human if n_human else 0.0
                net_sol = sum(s.sol_amount if s.is_buy else -s.sol_amount for s in snap)
                nps = net_sol / len(snap)
                est = _estimate_graduation_prob(vsol, bot_share, sph, nps, as_of - t0)

            vsol = seen[-1].vsol_after
            brk = _breakeven_probability(vsol)
            edge = est - brk

            tok = score_token(mint, seen)
            score = tok.conviction_score
            bot_share = tok.bot_trade_share

            if edge > entry_edge_threshold and score >= entry_score_threshold and bot_share < entry_bot_share_max:
                decision = "ENTER"
                entered = True
            elif edge > 0 and score >= 40:
                decision = "WATCH"
            else:
                decision = "SKIP"

            # Resolve the graduation payoff multiple from logged PumpSwap
            # prices when available (charter I2). Per-position exact: the real
            # exit price over this trade's bonding-curve entry price (the vSol at
            # the ENTER decision). Falls back to the row's representative multiple,
            # then to the default 5x. Only meaningful on the entry row (the loop
            # breaks once entered), where ``vsol`` is the entry vSol.
            payoff_mult = None
            if entered and graduated and mint in gp:
                row = gp[mint]
                price = row.get("price_sol_per_unit")
                if isinstance(price, (int, float)) and price > 0:
                    entry_price = curve_price_per_token(vsol)
                    if entry_price > 0:
                        payoff_mult = float(price) / entry_price
                if payoff_mult is None:
                    pm = row.get("payoff_multiple")
                    if isinstance(pm, (int, float)) and pm > 0:
                        payoff_mult = float(pm)
            trades.append(TradeOutcome(
                mint=mint, as_of_ts=as_of, decision=decision,
                est_grad_prob=est, breakeven_prob=brk, edge=edge,
                conviction_score=score, entered=entered,
                realized_return=(np.nan if not entered else _realized_return(
                    graduated,
                    fee_at(as_of).entry_cost(),
                    fee_at(as_of).exit_cost_pumpswap() if graduated else fee_at(as_of).exit_cost_curve(),
                    payoff_multiple=payoff_mult,
                )),
                graduated=graduated, seconds_to_graduation=(as_of - t0 if graduated else None),
            ))
            if entered:
                break
            k += 1

        # After resolving this token, add it to the training buffer and maybe
        # retrain the model walk-forward. We use features *at decision time*,
        # which is what the model will see live — not the final state.
        decision_as_of = trades[-1].as_of_ts if trades and trades[-1].mint == mint else last_ts
        train_buf.append((mint, swaps, graduated, creator, decision_as_of))
        if len(train_buf) >= fold_size and len(train_buf) % fold_size == 0:
            df = build_training_frame(train_buf)
            if not df.empty and df["graduated"].nunique() == 2:
                try:
                    model = fit(df, kind=model_kind)
                except Exception as e:
                    import sys
                    print(f"  [warn] model fit failed at fold {len(train_buf)}: {e}",
                          file=sys.stderr, flush=True)
            # Rolling hit rate over the last fold for the edge-over-time chart.
            recent = [t for t in trades if t.entered][-fold_size:]
            if recent:
                edge_over_time.append(sum(1 for t in recent if t.graduated) / len(recent))

    # Aggregate.
    enter_trades = [t for t in trades if t.entered]
    n_graduated = sum(1 for _, _, label, _ in items if label == "graduated")
    brier = float("nan")
    pr_auc = float("nan")
    logloss = float("nan")
    if trades:
        y = np.array([1.0 if t.graduated else 0.0 for t in trades])
        p = np.clip(np.array([t.est_grad_prob for t in trades]), 0.0, 1.0)
        if len(np.unique(y)) == 2:
            from sklearn.metrics import (
                average_precision_score,
                brier_score_loss,
                log_loss,
            )
            brier = float(brier_score_loss(y, p))
            pr_auc = float(average_precision_score(y, p))
            logloss = float(log_loss(y, p, labels=[0.0, 1.0]))

    # Calibration on the ENTER subset only (where the gate decision matters).
    brier_enter = float("nan")
    rel_slope = float("nan")
    if enter_trades:
        ye = np.array([1.0 if t.graduated else 0.0 for t in enter_trades])
        pe = np.clip(np.array([t.est_grad_prob for t in enter_trades]), 0.0, 1.0)
        if len(np.unique(ye)) == 2:
            from sklearn.metrics import brier_score_loss
            brier_enter = float(brier_score_loss(ye, pe))
        rel_slope = _reliability_slope(
            [float(x) for x in pe], [bool(x) for x in ye]
        )

    # §5.4 power & robustness.
    enter_returns = [t.realized_return for t in enter_trades if not np.isnan(t.realized_return)]
    ci_low = _bootstrap_ci_low(enter_returns)
    jk_pos = _jackknife_positive(enter_returns)
    n_grad_in_enter = sum(1 for t in enter_trades if t.graduated)
    # Fold stability: edge sign positive in every fold of the final window.
    fold_stable = bool(edge_over_time) and all(r > 0 for r in edge_over_time[-3:])

    return BacktestReport(
        n_tokens=len(items),
        n_decisions=len(trades),
        n_enter=len(enter_trades),
        n_watch=sum(1 for t in trades if t.decision == "WATCH"),
        n_skip=sum(1 for t in trades if t.decision == "SKIP"),
        enter_hit_rate=(sum(1 for t in enter_trades if t.graduated) / len(enter_trades))
                       if enter_trades else float("nan"),
        enter_mean_return=float(np.nanmean([t.realized_return for t in enter_trades]))
                           if enter_trades else float("nan"),
        baseline_hit_rate=(n_graduated / len(items)) if items else float("nan"),
        brier_score=brier,
        edge_over_time=edge_over_time,
        trades=trades,
        pr_auc=pr_auc,
        log_loss=logloss,
        brier_score_enter=brier_enter,
        reliability_slope=rel_slope,
        bootstrap_ci_low=ci_low,
        jackknife_positive=jk_pos,
        n_grad_in_enter=n_grad_in_enter,
        fold_stability=fold_stable,
    )


def write_report(report: BacktestReport, path: str | Path) -> Path:
    """Dump a report to JSON for the monitoring dashboard."""
    def _safe(v: float) -> float | None:
        """Replace NaN with None for valid JSON."""
        return None if math.isnan(v) else v

    p = Path(path)
    p.write_text(json.dumps({
        "n_tokens": report.n_tokens,
        "n_decisions": report.n_decisions,
        "n_enter": report.n_enter,
        "n_watch": report.n_watch,
        "n_skip": report.n_skip,
        "enter_hit_rate": _safe(report.enter_hit_rate),
        "enter_mean_return": _safe(report.enter_mean_return),
        "baseline_hit_rate": _safe(report.baseline_hit_rate),
        "brier_score": _safe(report.brier_score),
        "pr_auc": _safe(report.pr_auc),
        "log_loss": _safe(report.log_loss),
        "brier_score_enter": _safe(report.brier_score_enter),
        "reliability_slope": _safe(report.reliability_slope),
        "bootstrap_ci_low": _safe(report.bootstrap_ci_low),
        "jackknife_positive": report.jackknife_positive,
        "n_grad_in_enter": report.n_grad_in_enter,
        "fold_stability": report.fold_stability,
        "edge_over_time": report.edge_over_time,
        "trades": [
            {
                "mint": t.mint, "as_of_ts": t.as_of_ts, "decision": t.decision,
                "est_grad_prob": t.est_grad_prob, "breakeven_prob": t.breakeven_prob,
                "edge": t.edge, "conviction_score": t.conviction_score,
                "entered": t.entered,
                "realized_return": (None if math.isnan(t.realized_return) else t.realized_return),
                "graduated": t.graduated,
                "seconds_to_graduation": t.seconds_to_graduation,
                "reasons": t.reasons,
            }
            for t in report.trades
        ],
    }, indent=2, default=float))
    return p


def load_examples_from_store(
    store_db: str | Path,
    *,
    holdout_start_ts: float | None = None,
) -> tuple[
    list[tuple[str, list[Swap], str, CreatorHistory | None]],
    dict[str, dict[str, object]],
]:
    """Pull all labelled tokens out of the SQLite store as backtest examples.

    Returns ``(examples, graduation_prices)``. The graduation_prices map is
    keyed by mint and used by :func:`run_backtest` to value graduation exits at
    the real PumpSwap first-trade price (charter I2). Empty until tokens start
    graduating on the live feed and their prices get logged.

    Phase 2.2 creator features aren't wired here (we don't have the creator
    index yet) — they're returned as None and the model handles NaN. Add the
    creator-history join once you start logging creator addresses.

    ``holdout_start_ts``: when set, tokens whose first swap ts is >= this value
    are *excluded* — they belong to the locked holdout (charter §9.1) and must
    not enter training/selection. This is the fence that prevents the holdout
    from being consumed silently. The §5.4 gate evaluates on those tokens via a
    separate, single-shot command.
    """
    from storage import Store
    s = Store(store_db)
    examples: list[tuple[str, list[Swap], str, CreatorHistory | None]] = []
    for mint, outcome in s.labelled_mints():
        swaps = s.swaps_for(mint)
        if not swaps:
            continue
        if holdout_start_ts is not None and swaps[0].ts >= holdout_start_ts:
            continue  # holdout — not for training/selection
        examples.append((mint, swaps, outcome, None))
    graduation_prices = s.all_graduation_prices()
    s.close()
    return examples, graduation_prices