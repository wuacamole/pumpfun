# Decision Log

Append-only. Every gate decision, threshold, material modeling choice, and
backtest run gets an entry here. If it isn't in this log, it didn't happen —
and it can't be cited later as evidence the gate was met. See CHARTER.md §9.2.

Format:
```
[YYYY-MM-DD] [gate|threshold|model|kill|run] author=…
data_hash=<git+db snapshot hash>   backtest_runs_to_date=<n>
metrics={post_cost_mean, ci_low, brier, n_grad, n_enter, …}
decision=<pass|fail|defer|kill>   rationale=<one paragraph>
```

---

[2026-06-26] [gate] author=system
data_hash=initial
backtest_runs_to_date=0
metrics={}
decision=defer
rationale=Phase 1 (data integrity) marked DONE per CHANGES.md. Phase 2 exit
gate not yet evaluated — PR-AUC, log-loss, isotonic calibration, and
reliability diagrams per fold are not yet implemented. This entry seeds the
decision log at the start of the charter-compliance work.