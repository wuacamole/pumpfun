import json, os, numpy as np
from collections import Counter

# --- gap filter ------------------------------------------------------------
# Under interval (duty-cycled) capture the firehose is only connected ON_SEC of
# every PERIOD; a trade whose lifetime spans an OFF blackout has a path-blind,
# unreliable exit (we never saw the price action that closed it). Such trades
# are excluded from the headline numbers below. A trade is CLEAN iff:
#   - it closed before interval mode began (continuous capture), OR
#   - it is fully contained in a logged ON window (interval_windows.jsonl), OR
#   - (pre-window-log interval trades) its hold fit inside the 90s ON window
#     that was in effect before the log existed.
INTERVAL_START_TS = 1782768720.0   # interval mode cutover (2026-06-29 21:32 UTC)
PRELOG_ON_SEC = 90.0               # ON_SEC in effect before window logging existed

_wins = []
if os.path.exists("interval_windows.jsonl"):
    _wins = [json.loads(l) for l in open("interval_windows.jsonl")]

def _clean(t):
    ein, eout = t["entry_ts"], t["exit_ts"]
    if eout < INTERVAL_START_TS:
        return True                                  # continuous-capture era
    for w in _wins:                                  # fully inside one ON window
        if w["start"] <= ein and eout <= w["end"]:
            return True
    if not any(w["start"] <= eout <= w["end"] for w in _wins):
        return (eout - ein) <= PRELOG_ON_SEC         # pre-log interval fallback
    return False                                     # spanned a blackout -> drop

rows_all = [json.loads(l) for l in open("cm_trades.jsonl")]
rows = [t for t in rows_all if _clean(t)]
dropped = len(rows_all) - len(rows)

r = np.array([x["ret"] for x in rows]); n = len(r)
idx = np.random.randint(0, n, (5000, n)); bs = r[idx].mean(1)
print(f"n={n} (dropped {dropped} gap-spanning) win={(r>0).mean():.1%} "
      f"mean={r.mean():+.1%} median={np.median(r):+.1%} std={r.std():.0%}")
print(f"CI [{np.percentile(bs,2.5):+.1%}, {np.percentile(bs,97.5):+.1%}]")
print("exits", dict(Counter(x["reason"] for x in rows)))
for k in (40, 80):
    l = r[-k:]; print(f"last{k} mean={l.mean():+.1%} win={(l>0).mean():.0%} median={np.median(l):+.1%}")
