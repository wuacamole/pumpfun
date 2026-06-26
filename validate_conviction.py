"""Validate the conviction engine against realised outcomes (priority #4).

For every token with a definitive label (graduated / stalled), we replay it to the
moment it first enters the strategy's entry zone (vSol >= ENTRY_VSOL, the ~36 the
live broker actually entered at) and score it using ONLY the swaps up to that
point — no lookahead. We then ask:

  1. Base rate: of tokens that reach the entry zone, what fraction graduate?
  2. Does conviction_score separate graduates from stalls? (AUC + buckets)
  3. Does the engine's own ENTER/WATCH/SKIP decision separate them?

Run: uv run python validate_conviction.py
"""
import sqlite3
import statistics

from conviction_engine import Swap, score_token

DB = "pumpfun.db"
ENTRY_VSOL = 36.0   # the live broker's observed entry vSol
MIN_SWAPS = 3       # need a few trades before scoring is meaningful
# Only evaluate tokens we started capturing BEFORE the entry zone (first logged
# swap below this vSol). Otherwise we have no genuine pre-entry history to score
# and the sample is biased by *when* we happened to start watching. This is also
# the only regime the live engine can act in — it needs early trades to score.
CATCH_VSOL = 32.0


def load_labeled():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    rows = c.execute(
        "SELECT mint, outcome FROM tokens WHERE outcome IN ('graduated','stalled')"
    ).fetchall()
    out = []
    for r in rows:
        swaps = [
            Swap(ts=s["ts"], wallet=s["wallet"], is_buy=bool(s["is_buy"]),
                 sol_amount=s["sol_amount"], vsol_after=s["vsol_after"],
                 via_frontend=bool(s["via_frontend"]))
            for s in c.execute(
                "SELECT ts,wallet,is_buy,sol_amount,vsol_after,via_frontend "
                "FROM swaps WHERE mint=? ORDER BY ts", (r["mint"],))
        ]
        out.append((r["mint"], r["outcome"], swaps))
    c.close()
    return out


def entry_snapshot(swaps):
    """Swaps up to and including the first one with vSol >= ENTRY_VSOL.
    Returns None if the token never reached the entry zone."""
    for i, s in enumerate(swaps):
        if s.vsol_after >= ENTRY_VSOL:
            return swaps[: i + 1]
    return None


def auc(scores_labels):
    """AUC = P(score(grad) > score(stall)) via rank-sum (Mann–Whitney)."""
    pos = [s for s, y in scores_labels if y == 1]
    neg = [s for s, y in scores_labels if y == 0]
    if not pos or not neg:
        return float("nan")
    ranked = sorted(scores_labels, key=lambda x: x[0])
    # average ranks (1-based), handle ties
    ranks = {}
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_pos = sum(ranks[k] for k, (_, y) in enumerate(ranked) if y == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    data = load_labeled()
    n_grad = sum(1 for _, o, _ in data if o == "graduated")
    n_stall = sum(1 for _, o, _ in data if o == "stalled")
    print(f"labeled tokens: {len(data)}  (graduated={n_grad} stalled={n_stall})\n")

    reached = 0
    skipped_late = skipped_short = 0
    rows = []  # (conviction, edge, decision, graduated)
    for mint, outcome, swaps in data:
        grad = 1 if outcome == "graduated" else 0
        # Must have been caught before the entry zone (genuine pre-entry history).
        if not swaps or swaps[0].vsol_after >= CATCH_VSOL:
            skipped_late += 1
            continue
        snap = entry_snapshot(swaps)
        if snap is None or len(snap) < MIN_SWAPS:
            skipped_short += 1
            continue
        reached += 1
        ts = score_token(mint, snap)
        rows.append((ts.conviction_score, ts.edge, ts.decision, grad))
    print(f"excluded: caught-late={skipped_late}  too-few-pre-entry-swaps={skipped_short}")

    # 1. Base rate
    print("=== 1. Entry-zone base rate (vSol>=%.0f, >=%d swaps) ===" % (ENTRY_VSOL, MIN_SWAPS))
    print(f"evaluated (caught early, reached entry zone): {reached}/{len(data)}")
    n_eval_grad = sum(g for *_, g in rows)
    print(f"  of which graduated={n_eval_grad}  stalled={len(rows)-n_eval_grad}")
    if rows:
        base = sum(g for *_, g in rows) / len(rows)
        print(f"graduation rate among entry-zone tokens: {base:.1%}\n")

    # 2. Conviction separation
    sl = [(c, g) for c, _, _, g in rows]
    grad_scores = [c for c, g in sl if g == 1]
    stall_scores = [c for c, g in sl if g == 0]
    print("=== 2. conviction_score vs outcome ===")
    if grad_scores and stall_scores:
        print(f"mean conviction  graduated={statistics.mean(grad_scores):.1f}  "
              f"stalled={statistics.mean(stall_scores):.1f}")
        print(f"median conviction graduated={statistics.median(grad_scores):.1f}  "
              f"stalled={statistics.median(stall_scores):.1f}")
        print(f"AUC (conviction predicts graduation): {auc(sl):.3f}   "
              f"(0.5=no signal, 1.0=perfect)")
        print("\ngraduation rate by conviction bucket:")
        for lo, hi in [(0, 20), (20, 40), (40, 55), (55, 70), (70, 101)]:
            b = [g for c, g in sl if lo <= c < hi]
            if b:
                print(f"  [{lo:>3},{hi:>3}): n={len(b):>4}  grad_rate={sum(b)/len(b):.1%}")

    # 3. Engine decision separation
    print("\n=== 3. engine decision at entry zone ===")
    for d in ("ENTER", "WATCH", "SKIP"):
        b = [g for _, _, dec, g in rows if dec == d]
        if b:
            print(f"  {d:<6} n={len(b):>4}  grad_rate={sum(b)/len(b):.1%}")
    # AUC of edge too
    el = [(e, g) for _, e, _, g in rows]
    print(f"\nAUC (edge predicts graduation): {auc(el):.3f}")


if __name__ == "__main__":
    main()
