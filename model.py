"""
model.py
========
Phase 2.1 — replace the hand-built :func:`conviction_engine._estimate_graduation_prob`
with a model fit on your own logged data.

What this module provides
-------------------------
  * :func:`featurize` — turn a token's swap history (a ``list[Swap]``) into the
    feature vector the model consumes. This is the **time-consistent** view:
    features are computed only from swaps that happened *before* a cutoff ts,
    so backtests can't leak the future into training.
  * :class:`FittedModel` — a thin wrapper over scikit-learn that exposes both
    logistic regression (interpretable baseline) and gradient-boosted trees
    (more accurate). It returns a *calibrated* graduation probability.
  * :func:`fit` and :func:`predict_grad_prob` — the training + inference entry
    points. The live monitor / backtest call these instead of the hand weights.
  * :class:`ModelStore` — load/save a fitted model + the feature schema to disk
    so the live system can hot-swap a retrained model without code changes.

Feature set (the paper's conditioning variables, plus Phase 2.2 additions)
---------------------------------------------------------------------------
  vsol, real_sol_bonded, pct_to_graduation,
  bot_trade_share, sol_per_human_wallet, net_sol_per_swap,
  n_swaps, n_unique_wallets, n_unique_human_wallets, seconds_since_launch,
  sol_per_min (momentum),
  creator_graduated_count,    # Phase 2.2 — needs creator-history join
  creator_rugged_count,       # Phase 2.2 — needs creator-history join
  top_holder_share,           # Phase 2.2 — needs holder snapshot
  smart_money_present,        # Phase 2.2 — needs a smart-money wallet set

The creator/holder features are only emitted when the optional join data is
passed in; otherwise they're NaN and the model handles missingness natively
(GBM: NaN-aware; logistic: imputed with median + missingness indicator).

IMPORTANT — no lookahead
------------------------
:func:`featurize` takes an ``as_of`` timestamp. Only swaps with ``ts <= as_of``
contribute. This is the discipline the roadmap warns about: time-consistent
features, no lookahead. The backtest harness relies on this.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from conviction_engine import (
    Swap,
    _real_sol_target,
    _virtual_sol_start,
)

# Feature column order — frozen so saved models stay loadable.
FEATURE_COLUMNS: list[str] = [
    "vsol",
    "real_sol_bonded",
    "pct_to_graduation",
    "bot_trade_share",
    "sol_per_human_wallet",
    "net_sol_per_swap",
    "n_swaps",
    "n_unique_wallets",
    "n_unique_human_wallets",
    "seconds_since_launch",
    "sol_per_min",
    "creator_graduated_count",
    "creator_rugged_count",
    "top_holder_share",
    "smart_money_present",
]

LABEL_COL = "graduated"  # 1 if the token graduated, 0 otherwise


@dataclass
class CreatorHistory:
    """Phase 2.2 join data — per-token creator context. NaN if absent."""

    creator_graduated_count: float = float("nan")
    creator_rugged_count: float = float("nan")
    top_holder_share: float = float("nan")
    smart_money_present: float = float("nan")


def featurize(
    swaps: list[Swap],
    as_of: float | None = None,
    creator: CreatorHistory | None = None,
) -> dict[str, float]:
    """Compute time-consistent features from ``swaps`` observed up to ``as_of``.

    ``as_of`` is a unix timestamp. Swaps with ``ts > as_of`` are ignored — this
    is what prevents lookahead in the backtest. If ``as_of`` is None, all swaps
    are used (the live-monitor case: "now" is the latest swap ts).
    """
    if not swaps:
        raise ValueError("no swaps provided")
    if as_of is None:
        as_of = max(s.ts for s in swaps)
    seen = [s for s in swaps if s.ts <= as_of]
    if not seen:
        raise ValueError("no swaps before as_of")
    seen.sort(key=lambda s: s.ts)

    t0 = seen[0].ts
    latest = seen[-1]
    vsol = latest.vsol_after
    real_sol_bonded = max(0.0, vsol - _virtual_sol_start())
    seconds = max(latest.ts - t0, 1e-6)

    bot_share = sum(1 for s in seen if not s.via_frontend) / len(seen)
    net_sol = sum(s.sol_amount if s.is_buy else -s.sol_amount for s in seen)
    net_sol_per_swap = net_sol / len(seen)

    human_wallets = {s.wallet for s in seen if s.via_frontend}
    all_wallets = {s.wallet for s in seen}
    n_human = len(human_wallets)
    human_net = sum(
        s.sol_amount if s.is_buy else -s.sol_amount for s in seen if s.via_frontend
    )
    sol_per_human = human_net / n_human if n_human else 0.0

    sol_per_min = (vsol - _virtual_sol_start()) / (seconds / 60.0)
    c = creator or CreatorHistory()

    return {
        "vsol": vsol,
        "real_sol_bonded": real_sol_bonded,
        "pct_to_graduation": 100.0 * real_sol_bonded / _real_sol_target(),
        "bot_trade_share": bot_share,
        "sol_per_human_wallet": sol_per_human,
        "net_sol_per_swap": net_sol_per_swap,
        "n_swaps": float(len(seen)),
        "n_unique_wallets": float(len(all_wallets)),
        "n_unique_human_wallets": float(n_human),
        "seconds_since_launch": seconds,
        "sol_per_min": sol_per_min,
        "creator_graduated_count": c.creator_graduated_count,
        "creator_rugged_count": c.creator_rugged_count,
        "top_holder_share": c.top_holder_share,
        "smart_money_present": c.smart_money_present,
    }


def feature_matrix(rows: list[dict[str, float]]) -> pd.DataFrame:
    """Build a feature DataFrame with the canonical column order. NaN-safe."""
    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[FEATURE_COLUMNS]


# ----------------------------------------------------------------- model
@dataclass
class FittedModel:
    """A trained graduation-probability model + its calibration Brier score.

    ``kind`` is 'logistic' or 'gbm'. ``predict`` returns P(graduation | features).
    ``brier_score`` is the in-sample (training-set) Brier score — a lower bound
    on how well the model calibrates. For the honest out-of-sample Brier, run
    ``cli.py backtest``; that number is what the breakeven comparison needs.
    """

    kind: str
    pipeline: Pipeline | CalibratedClassifierCV
    brier_score: float
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_columns]
        return np.asarray(self.pipeline.predict_proba(X)[:, 1])

    def predict_one(self, features: dict[str, float]) -> float:
        X = feature_matrix([features])
        return float(self.predict(X)[0])


def fit(
    df: pd.DataFrame,
    kind: str = "gbm",
    calibrate: bool = True,
) -> FittedModel:
    """Fit a graduation model on a labelled feature DataFrame.

    ``df`` must include all FEATURE_COLUMNS + LABEL_COL. Use
    :func:`build_training_frame` to construct it from a logged swap history.

    ``kind``: 'logistic' (interpretable baseline) or 'gbm' (more accurate).
    The pipeline handles imputation (median) + scaling for logistic, and is
    NaN-native for GBM.

    ``calibrate``: when True (default), wrap the base classifier in
    ``CalibratedClassifierCV`` with isotonic regression (charter §5.2 —
    "Probabilities are calibrated (isotonic/Platt)"). Uses internal
    cross-validation so the calibration is honest (not in-sample). Disable
    for the raw classifier when you want the uncalibrated baseline.
    """
    if kind not in ("logistic", "gbm"):
        raise ValueError(f"kind must be 'logistic' or 'gbm', got {kind!r}")
    if LABEL_COL not in df.columns:
        raise ValueError(f"DataFrame must have a {LABEL_COL!r} column")

    X = df[FEATURE_COLUMNS].copy()
    y = df[LABEL_COL].astype(int).values

    if kind == "logistic":
        base = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
    else:
        # GBM needs imputation (not NaN-native). Same imputer as logistic,
        # but no scaling — tree models don't need it.
        base = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=42,
            )),
        ])

    # Charter §5.2: probabilities must be calibrated. Isotonic calibration via
    # internal CV — cv=3 (or 2 if too few positives) so it's honest, not in-
    # sample. Falls back gracefully when there aren't enough positives for 3
    # folds (uses cv=2 or skips calibration entirely on a single-class set).
    if calibrate and len(np.unique(y)) == 2 and min(sum(y), sum(1 - y)) >= 2:
        n_splits = min(3, min(int(sum(y)), int(sum(1 - y))))
        pipeline: Pipeline | CalibratedClassifierCV = CalibratedClassifierCV(
            estimator=base, method="isotonic", cv=n_splits,
        )
    else:
        pipeline = base

    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    # Brier on the training fit; the backtest reports honest OOS Brier.
    brier = float(brier_score_loss(y, proba))
    return FittedModel(kind=kind, pipeline=pipeline, brier_score=brier)


def predict_grad_prob(
    model: FittedModel,
    swaps: list[Swap],
    as_of: float | None = None,
    creator: CreatorHistory | None = None,
) -> float:
    """Drop-in replacement for :func:`conviction_engine._estimate_graduation_prob`.

    Returns the model's P(graduation) for the snapshot of ``swaps`` at ``as_of``.
    If ``model`` is None, the caller should fall back to the hand-built estimate.
    """
    feats = featurize(swaps, as_of=as_of, creator=creator)
    return model.predict_one(feats)


# ------------------------------------------------------------- training frame
def build_training_frame(
    examples: list[tuple[str, list[Swap], bool, CreatorHistory | None, float | None]],
) -> pd.DataFrame:
    """Build the labelled feature matrix used by :func:`fit`.

    Each example is ``(mint, swaps, graduated, creator?, as_of?)``.

    ``as_of`` is when the *decision* was made. For training the canonical choice
    is "as of N seconds after launch" or "as of reaching some vSol threshold".
    Using the final state is leakage — only features visible at ``as_of``
    may feed the model. If ``as_of`` is None, the last swap's ts is used.
    """
    rows: list[dict[str, float | int]] = []
    for mint, swaps, graduated, creator, as_of in examples:
        try:
            f = featurize(swaps, as_of=as_of, creator=creator)
        except ValueError:
            continue  # skip degenerate tokens
        f[LABEL_COL] = int(bool(graduated))
        f["mint"] = mint  # type: ignore[assignment]
        rows.append(f)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.reset_index(drop=True)
    return df


# ------------------------------------------------------------- persistence
class ModelStore:
    """Load/save a FittedModel to disk so the live system can hot-swap it."""

    def __init__(self, dir: str | Path = "models") -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: FittedModel, name: str = "latest") -> Path:
        path = self.dir / f"{name}.pkl"
        meta_path = self.dir / f"{name}.json"
        with path.open("wb") as f:
            pickle.dump(model, f)
        meta_path.write_text(json.dumps({
            "kind": model.kind,
            "brier_score": model.brier_score,
            "feature_columns": model.feature_columns,
            "saved_at": time.time(),
        }))
        return path

    def load(self, name: str = "latest") -> FittedModel:
        path = self.dir / f"{name}.pkl"
        with path.open("rb") as f:
            m = pickle.load(f)  # noqa: S301 — our own file, trusted
        assert isinstance(m, FittedModel)
        return m

    def latest_meta(self, name: str = "latest") -> dict[str, object] | None:
        meta_path = self.dir / f"{name}.json"
        if not meta_path.exists():
            return None
        return dict(json.loads(meta_path.read_text()))