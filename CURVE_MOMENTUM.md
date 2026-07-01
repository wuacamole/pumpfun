# Curve-Momentum Strategy — How It Works, Economics, and Improvements

*Last updated 2026-06-28. Companion to `DECISIONS.md` ([model] curve-momentum entries) and
the memory note `conviction-engine-inverted`. All numbers are PAPER / backtest on a single
~1.5-day capture window unless stated. **No capital has been deployed.***

---

## 1. TL;DR

After the graduation-prediction strategy was **killed** (most graduations exit at a loss, and
~80% are unsellable — see §2), one edge candidate survived every test we could run on existing
data: **enter a token early on the bonding curve, ride momentum, and exit *before* graduation —
on the curve, which is liquid.**

Two operating points on the same strategy:

| | **"Lottery"** (TP 5× / SL 0.3×) | **"Grinder"** (TP 1.7× / SL 0.2×) |
|---|---|---|
| Win rate | ~22–25% | **~44–51%** |
| Mean return / trade (moderate frictions) | ~+24% | +6–11% |
| Typical (median) trade | −42% (a loss) | ~breakeven / small +|
| Per-trade volatility | 174% | **92%** |
| Longest losing streak | 31 | **11** |
| Total PnL (window, 1 SOL/trade) | +216 SOL | +92 SOL |
| Return / max-drawdown | **6.3×** | 2.7× |
| Survives *pessimistic* execution frictions | **yes (CI-low +1.8%)** | marginal (CI-low −2.6%) |
| Tail-dependence / regime-fragility | high | **low** |

Neither dominates. The lottery makes more money but is a brutal, tail-dependent ride. The
grinder wins more often, is far smoother, and is less likely to be a one-regime fluke — but its
thinner per-trade margin means it needs **decent execution** to stay positive.

**The one thing neither has proven: regime generalization.** Everything is one ~1.5-day window.
A live forward paper-test is running now to settle it (§9).

---

## 2. Why this exists — the graduation strategy is dead

The original thesis (from the Lillo et al. paper): early behavioural "conviction" predicts
graduation; trade the graduators. We falsified the *tradeable* version of it:

- **Graduation exits are loss-heavy.** Real PumpSwap exit prices: **median 0.65×**, 57% below
  entry. The apparent "+99% return" in early backtests was an artifact of a flat-5× *placeholder*
  payoff; valued at measured prices the strategy is **−41% to −77%** (DECISIONS [kill]
  winside-bracket — proven by a same-data A/B varying only the win-side valuation).
- **~80% of graduates are dead-on-arrival.** A random sample showed only ~20% of graduated tokens
  have *any* post-graduation PumpSwap trading; the rest graduate then go silent — **no exit
  liquidity, unsellable** (DECISIONS [kill] liquidity-wall). So the "win" can't even be realized
  most of the time.

The model itself is *not* the problem — it genuinely ranks graduations (entry-zone OOS AUC 0.75).
**The target was the problem.** Which led to the key reframe ↓

---

## 3. The mechanism — why exiting *before* graduation works

The pump.fun bonding curve is a **constant-product** AMM: `vSol · vTok = k`, so the spot price is
`price(V) = V² / k` (SOL per token). Crucially, **the curve is liquid by construction** — you can
always sell back into it. That's the opposite of the dead post-graduation PumpSwap pools.

So the edge is: a token that climbs from entry (vSol ~40) toward graduation (vSol ~115) appreciates
on a **liquid** curve. You ride that climb and **exit on the curve before the graduation cliff**,
sidestepping the liquidity wall entirely. The winners are the same tokens the model is good at
spotting (climbers/graduators) — but you capture the *pre-graduation* move, which is realizable,
instead of the *post-graduation* dump, which isn't.

---

## 4. The strategy

**Entry** (one decision, at the first qualifying swap):
- Token caught **clean** — first observed vSol < 32 (genuine early history, no missed run-up).
- vSol reaches the **entry zone** (≥ 36).
- Model ranks it **top ~50%** by P(graduation) — `predict_grad_prob ≥ 0.065` (the historical
  entry-zone median; recalibrate per model). Top-50% selection lifts mean +24% vs +14.7% basket;
  finer cuts (top-10%) go noisy — use a *moderate* filter, not the extreme tail.

**Exit** — first of:
- **Take-profit**: spot ≥ TP× entry (reserve ≥ V_entry·√TP).
- **Stop-loss**: spot ≤ SL× entry (reserve ≤ V_entry·√SL).
- **Graduation**: force-exit at the last pre-graduation price (you *cannot* sell post-grad).
- **Time-stop / eviction**: exit at the current/last price if held too long or the monitor evicts
  the (now-stale) token. This is the live analogue of the replay's "exit at last swap".

`(TP, SL)` is the only real knob; §1 gives the two studied settings.

**Realized return** uses **exact constant-product entry+exit impact** (slippage scales with
position size) plus curve fee (1%/side), MEV (1% entry), and an optional adverse-fill haircut.

---

## 5. The cost / slippage math

For a trade of size `S` SOL at entry reserve `V_e`, exit reserve `V_x`:

```
S_in   = S · (1 − fee − mev − haircut)          # entry skim
Q      = k/V_e − k/(V_e + S_in)                 # tokens bought (entry price impact)
SOL_out= V_x − k/(k/V_x + Q)                     # SOL received (exit price impact)
return = SOL_out·(1 − fee − haircut) / S − 1
```

Because impact scales with `S`, sweeping size maps **capacity**: the edge is strong to ~2 SOL/
position and **dies at ~5 SOL** (thin early-curve liquidity). Run at ≤1–2 SOL/position.

---

## 6. What was tested (the validation gauntlet)

All on existing data, all **passed** for the lottery config:

1. **Slippage / capacity** — exact curve impact; edge holds to ~2 SOL/position.
2. **Overfitting** — time-ordered OOS (select rule on early tokens, test on later): held; config
   ranking transfer Spearman **0.99**; 82% of the TP/SL grid OOS-positive.
3. **Execution frictions** — pessimistic stack (latency + MEV 5% + 3% haircut): still **+10%,
   CI-low +1.8%**.
4. **Drawdown / bankroll** — survivable: ~60 SOL bankroll, return/maxDD ~6.3×.
5. **Model-selection lift** — top-50% nearly doubles mean (+14.7% → +24%).
6. **Win-rate frontier** (the grinder) — temporally stable in *both* time halves (§7).

**Unproven:** regime generalization (single window). That's the live forward test.

---

## 7. Economics in detail

**Lottery (TP5/SL0.3), OOS, moderate frictions, 1 SOL:** win ~22%, mean +24% (CI-low +10.6%),
median −42%, total +216 SOL, max DD 34 SOL, **31-trade losing streaks**, std 174%, ~60 SOL
bankroll. The +mean rides the ~12% that run up the curve — a heavy-tailed lottery.

**Grinder (TP1.7/SL0.2), OOS, both-halves stable:**

| half | win% | mean | CI-low |
|---|---|---|---|
| early | 48% | +8.4% | +4.4% |
| late | 52% | +12.4% | +8.7% |

Drawdown (moderate frictions, 1 SOL): n=1512, win 44%, mean +6.1%, median −17%, total +92 SOL,
max DD 34 SOL, **return/maxDD 2.7×**, **longest losing streak 11**, std **92%**, ~54 SOL bankroll.

The grinder halves the variance and cuts losing streaks from 31→11, at the cost of lower absolute
return and a thinner friction margin.

---

## 8. Would the grinder be profitable?

**Yes — in-sample and out-of-sample-within-window, under realistic-moderate execution — but with
caveats that make "live profitable" still unproven.**

The case *for*:
- Mean **+6–11% per trade**, CI-low positive (+1.4% to +8.7%), **stable across both time halves**
  and across the slippage/exec-stress tests at moderate friction.
- ~50% win and a near-breakeven median → it doesn't depend on rare jackpots, so it's **far less
  regime-fragile** than the lottery (the single biggest risk).
- Lower variance (std 92%) and short losing streaks (11) → **practically tradeable** on ~54 SOL.

The case *against* / what could make it unprofitable:
- **Friction-sensitive.** Small TP (1.7×) means fixed costs eat a bigger slice. Under *pessimistic*
  execution it slips to **+1.8% mean / CI-low −2.6%** — i.e. a sloppy bot (high latency, bad fills,
  heavy sandwiching) erases it. The lottery survives that; the grinder doesn't. **Execution quality
  is the swing factor.**
- **Regime unproven.** Like everything here, validated on one ~1.5-day window. A different market
  mood could break it. The live forward test (§9) is the only way to know.
- **Modest absolute edge.** +6% × ~2 SOL capacity × concurrency is real but capacity-bounded; this
  is a small, high-touch strategy, not a money printer.

**Bottom line:** the grinder is a *genuinely positive-expectancy* strategy on all data we have,
with a better risk profile than the lottery — **profitable if (a) the edge generalizes to new
regimes and (b) you execute cleanly.** Both are open questions, and (a) is being tested right now.

---

## 9. Status — live forward test

A paper forward-test is **running** (since 2026-06-28, firehose ON, ~2.15M credits/day, ~2.4-day
runway). It trades the strategy live on *fresh* tokens — the out-of-regime sample existing data
can't provide.

- Check: `uv run python cm_forward_report.py` (reads `cm_trades.jsonl`).
- Stop (save credits): `systemctl stop pumpfun-monitor.service`.
- Currently configured: **lottery (TP5/SL0.3)**. Switching to the grinder would give a *faster,
  cleaner* regime read (it converges in hours, not days, because it isn't waiting on rare tail
  events) and the higher win rate — at the cost of needing good execution.

---

## 10. Ways to improve

Done / validated:
- **Grinder operating point** (tight SL + modest TP) — higher win rate, lower variance (§7).
- **Model top-50% selection** — lifts mean, better capital efficiency.

Tested 2026-07-02 and **REJECTED** (improve_entry_filter.py / trail_stress.py; DECISIONS.md
2026-07-02 — do NOT re-mine these as "new" ideas):
1. ~~**Entry-quality filter**~~ — five as-of-entry features (velocity, sell share, pre-entry
   retrace, buy concentration, distinct buyers) add ~nothing beyond the model gate on the
   time-ordered test slice (best +0.8pt CI-low, within noise). The model's 11 features
   already carry the entry-quality signal.
2. ~~**Trailing stop instead of fixed TP**~~ — the apparent near-doubling of mean was a
   fill-convention artifact (exiting AT the stop level; memecoin swaps gap through stops).
   Under pessimistic fills (breaching swap's price) + 1-swap latency: +1–2pt mean, win rate
   flat-to-lower, loses to baseline in 4/6 regime windows. Not worth the added path-dependence.

Adopted 2026-07-02: **MODEL_SCORE_MIN 0.070 → 0.115** — the pre-committed trigger fired
(forward n=940: the [0.070,0.115) band was dead weight, win 55.1% / +1.4% / CI-low −7.7%;
replay test slice and 6-window regime check agree). Win ~+1.5–2pt, mean +2–3pt, ~70% of entries.

Promising, **not yet tested** (free, on existing data):
3. **Volatility-scaled TP/SL** — set TP/SL from the token's recent vSol volatility rather than fixed
   multiples, so the stop isn't whipsawed on choppy names.
4. **Position sizing (fractional Kelly)** — size by model score / edge estimate rather than flat
   1 SOL, within the ~2 SOL capacity cap.
5. **Regime / time-of-day filter** — if forward data shows the edge is regime-dependent, gate entries
   on a market-activity or volatility regime signal.
6. **Better tail prediction** — the model ranks *graduation*; a model trained to rank *post-entry
   curve return* directly might select climbers better (the tail-predictability question was
   underpowered at n=55; revisit once more priced data exists).
7. **Concurrency/capacity management** — cap simultaneous positions and route size to stay under the
   per-pool slippage capacity.

---

## 10.5 Live execution plan — replicate Axiom's stack via Jito (NOT route through Axiom)

**Decision (2026-06-28, after research):** do **not** route through Axiom. Replicate its execution
edge directly with Jito bundles. Rationale, confirmed:

- **Axiom has no public trading API.** axiom.trade is a UI terminal; `docs.axiom.trade` documents UI
  features only (Pulse/Swap/Portfolio/Limit). The community SDKs (`axiomtradeapi` Python,
  `axiomtrade-rs`) work by **scraping the browser session token** — fragile, breaks on any Axiom
  update, likely ToS-violating. Not viable for a ~100-trade/hr bot.
- **Axiom's actual edge is just Jito bundle routing** (+ optimized compute-unit/priority fees + Jito
  tips, submitted as a bundle through Jito's block engine to prevent sandwiching). All of that is
  directly replicable — the bot already speaks Solana/Helius RPC.
- **Axiom charges 0.95%/side (Wood tier) → 0.75% (Champion)** — ~1.9% round-trip ON TOP of the
  pump.fun 1%/side curve fee. Replicating via Jito avoids that platform fee entirely.

**Quantified — the grinder survives, and Jito-replicate is the better route** (`exec_stress.py`,
grinder TP1.7/SL0.2, OOS late slice, 1 SOL, 1-swap latency):

| routing | win% | mean | CI-low |
|---|---|---|---|
| **Jito-replicate, calm** (no platform fee, bundle MEV≈0) | 45% | **+12.2%** | **+7.4%** |
| Jito-replicate, congested (tips up, residual MEV) | 45% | +10.5% | +5.8% |
| Axiom-route, calm (0.95%/side fee) | 45% | +9.6% | +4.9% |
| Axiom-route, congested | 44% | +8.5% | +3.8% |

The grinder clears **CI-low > 0 in every case**. Jito-replicate beats Axiom-route by ~2.6%/trade —
you skip the platform fee **and** bundle protection drives MEV→0, which is exactly the grinder's weak
spot (the earlier marginal −2.6% case assumed *unprotected* MEV 5%; bundles rescue it). So Jito is
strategically superior, not a fallback.

## 10.6 Jito integration — scope (next build, pre-go-live)

| component | implementation |
|---|---|
| MEV / sandwich protection | submit entry & exit as **Jito bundles** via the block-engine (`sendBundle`); private, atomic |
| Priority fees | `ComputeBudget` set-unit-limit + set-unit-price instructions, tuned to recent congestion |
| Jito tip | tip instruction to a Jito tip account, sized to land in-block (the dominant *fixed* cost at 1–2 SOL — model it as a per-side haircut, recheck small-size economics) |
| Bonding-curve buy/sell | direct **CPI to the pump.fun program** (no aggregator). NOTE: program updated 2026-04-28 — **buy = 18 accounts (was 17), sell = 16/17**; `helius.py` already parses the live curve, so the account layout is known |
| RPC | private/low-latency RPC (Helius already in use) for fast signal→submit |
| Slippage guard | enforce a max-slippage on the curve trade (compute expected `Q` from `vSol·vTok=k`; abort if fill deviates) |

**Scoping notes / risks:** (1) Jito tips spike with congestion → the dominant variable cost at small
size; meter real tips and fold into the friction model. (2) Bundle inclusion isn't guaranteed —
need a fallback (retry / public-mempool with tight slippage) and a max-attempts cap. (3) Exit latency
is the live analogue of the 1-swap stress already modelled; measure real submit→land time. (4) Build
& dry-run on a funded test wallet at **dust size first**, comparing realized fills to the model.

**Go-live gate (unchanged in spirit):** rebuild friction inputs from *measured* Jito tips + latency,
re-run `exec_stress.py` + `drawdown_analysis.py` and require CI-low > 0 — on top of a passing forward
(regime) test, then the §5.4 capital gate. Only then, any capital.

## 11. Files & tools

| file | purpose |
|---|---|
| `curve_momentum.py` | the strategy (`CurveMomentumPaper`) — live, paper-only |
| `test_curve_momentum.py` | 10 unit tests (entry/exit/slippage/eviction) |
| `cm_forward_report.py` | read `cm_trades.jsonl`, forward vs benchmark (regime test) |
| `peak_exit_replay.py` | flat-slippage replay (first discovery) |
| `curve_slippage_replay.py` | exact curve-aware slippage + capacity sweep |
| `oos_validation.py` | time-ordered out-of-sample validation |
| `exec_stress.py` | execution-friction stress |
| `drawdown_analysis.py` | bankroll / drawdown / losing-streak |
| `model_selection_lift.py` | model top-50% selection lift |
| `win_rate_frontier.py` | TP/SL win-rate vs return frontier (found the grinder) |

## 12. Hard rules (unchanged)

- **No capital** until the §5.4 capital gate passes on *measured* economics AND the regime risk is
  settled by forward testing. Paper only.
- The locked holdout is touched **exactly once**, never for a modeling decision.
- The graduation-target strategy is **dead** — do not resurrect it; this is a *different* strategy.
