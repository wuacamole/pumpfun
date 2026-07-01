"""Quick forward-test progress check. Gap filter (blackout-spanning trades dropped)
lives in cm_gap_filter.py, shared with cm_forward_report.py / cm_auto_report.py.
Run: uv run python _chk.py"""
from collections import Counter

import numpy as np

from cm_gap_filter import load_clean_trades

rows, dropped = load_clean_trades()
r = np.array([x["ret"] for x in rows])
n = len(r)
idx = np.random.randint(0, n, (5000, n))
bs = r[idx].mean(1)
print(f"n={n} (dropped {dropped} gap-spanning) win={(r > 0).mean():.1%} "
      f"mean={r.mean():+.1%} median={np.median(r):+.1%} std={r.std():.0%}")
print(f"CI [{np.percentile(bs, 2.5):+.1%}, {np.percentile(bs, 97.5):+.1%}]")
print("exits", dict(Counter(x["reason"] for x in rows)))
for k in (40, 80):
    tail = r[-k:]
    print(f"last{k} mean={tail.mean():+.1%} win={(tail > 0).mean():.0%} median={np.median(tail):+.1%}")
