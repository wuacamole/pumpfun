"""Quick validation: feed the engine two contrasting tokens and confirm it
separates conviction from churn the way the paper predicts."""
from conviction_engine import Swap, score_token


def churn_trap() -> list[Swap]:
    """Reaches 45 real SOL but via hundreds of tiny bot trades. Should SKIP."""
    swaps = []
    vsol = 30.0
    t = 0.0
    for i in range(400):
        t += 0.4
        is_buy = (i % 5 != 0)            # mostly buys, some sells -> churn
        amt = 0.05 + (i % 3) * 0.02      # tiny trades
        vsol += amt * 0.5 if is_buy else -amt * 0.4
        swaps.append(Swap(
            ts=t, wallet=f"bot{i % 12}", is_buy=is_buy, sol_amount=amt,
            vsol_after=vsol, via_frontend=(i % 10 == 0),  # 90% direct = bots
        ))
    return swaps

def conviction_token() -> list[Swap]:
    """Reaches 45 real SOL via ~50 large human buys, fast. Should ENTER."""
    swaps = []
    vsol = 30.0
    t = 0.0
    for i in range(50):
        t += 1.5
        is_buy = (i % 9 != 0)
        amt = 0.8 + (i % 4) * 0.3        # large, conviction-sized buys
        vsol += amt * 0.9 if is_buy else -amt * 0.5
        swaps.append(Swap(
            ts=t, wallet=f"human{i}", is_buy=is_buy, sol_amount=amt,
            vsol_after=vsol, via_frontend=(i % 8 != 0),   # mostly frontend = humans
        ))
    return swaps

for name, mk in [("CHURN TRAP", churn_trap), ("CONVICTION TOKEN", conviction_token)]:
    s = score_token(name, mk())
    print(f"\n=== {name} ===")
    print(f"  real SOL bonded     : {s.real_sol_bonded}  ({s.pct_to_graduation}% to grad)")
    print(f"  swaps / wallets     : {s.n_swaps} swaps, {s.n_unique_wallets} wallets "
          f"({s.n_unique_human_wallets} human)")
    print(f"  bot trade share     : {s.bot_trade_share:.0%}")
    print(f"  SOL per human wallet: {s.sol_per_human_wallet}  (conviction metric)")
    print(f"  net SOL per swap    : {s.net_sol_per_swap}")
    print(f"  conviction score    : {s.conviction_score}/100")
    print(f"  breakeven prob      : {s.breakeven_prob:.1%}")
    print(f"  estimated grad prob : {s.estimated_grad_prob:.1%}")
    print(f"  EDGE                : {s.edge:+.1%}")
    print(f"  >> DECISION         : {s.decision}")
    for r in s.reasons:
        print(f"     - {r}")
