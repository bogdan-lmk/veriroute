# Submission journal

Rule: **exactly one change per submission.** Record the prediction before submitting,
the leaderboard result after. Predicted-vs-actual accuracy feeds the judge calibration
(offset > 1 task → tighten the local gate to 18/19).

| # | Date | Image tag | Change vs previous | Predicted acc | Leaderboard acc | Tokens (own meter) | Tokens (leaderboard) | Notes |
|---|------|-----------|--------------------|---------------|-----------------|--------------------|----------------------|-------|
| 1 |      | :sub1     | baseline: escalate-all | — (probe)  |                 |                    |                      | probe: confirms 19-task hypothesis via accuracy granularity, feedback latency |

## Open questions being tracked

- [x] ~~Track 1 inference log required?~~ RESOLVED 2026-07-09: full-text pass over the guide —
      the Track 1 section has no log requirement; the only mention is "No inference log is
      required for Track 2". Tokens are counted by the judging proxy.
- [ ] Can one team submit to multiple tracks?
- [ ] Exact submission deadline (Event Schedule tab)
- [ ] ALLOWED_MODELS composition (published on launch day) → re-run ranking + token scenarios
- [ ] Confirm eval-set size ≈ 19 via submission #1 accuracy granularity
