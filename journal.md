# Submission journal

Rule: **exactly one change per submission.** Record the prediction before submitting,
the leaderboard result after. Predicted-vs-actual accuracy feeds the judge calibration
(offset > 1 task → tighten the local gate to 18/19).

| # | Date | Image tag | Change vs previous | Predicted acc | Leaderboard acc | Tokens (own meter) | Tokens (leaderboard) | Notes |
|---|------|-----------|--------------------|---------------|-----------------|--------------------|----------------------|-------|
| 1 |      | :sub1     | baseline: escalate-all | — (probe)  |                 |                    |                      | probe: confirms 19-task hypothesis via accuracy granularity, feedback latency |

## Live-run observations

- 2026-07-09, gpt-oss-20b on 8 practice tasks: 2130 tokens, ~31 s, all answers semantically
  correct EXCEPT practice-01 — the 300-token cap cut the answer BEFORE the "body of water"
  part the question asks for (finish_reason=length with non-empty content is not detected
  as truncation by design). 3/8 answers hit the cap; the model writes verbose markdown.
  → For sub2+: prepend a terse-answer instruction instead of raising the cap; add
  "content present but question sub-parts unanswered" to the eval checks.

## Open questions being tracked

- [x] ~~Track 1 inference log required?~~ RESOLVED 2026-07-09: full-text pass over the guide —
      the Track 1 section has no log requirement; the only mention is "No inference log is
      required for Track 2". Tokens are counted by the judging proxy.
- [ ] Can one team submit to multiple tracks?
- [ ] Exact submission deadline (Event Schedule tab)
- [ ] ALLOWED_MODELS composition (published on launch day) → re-run ranking + token scenarios
- [ ] Confirm eval-set size ≈ 19 via submission #1 accuracy granularity

## ALLOWED_MODELS intel (2026-07-09)

Probable final Track 1 list, cross-confirmed in two participant repos
(Ruththra/AI-Agent README "Final Track 1 Allowed Models" + omerdduran/token-router .env.example):
`minimax-m3, kimi-k2p7-code, gemma-4-31b-it, gemma-4-26b-a4b-it, gemma-4-31b-it-nvfp4`.

- Gemma-heavy list → Track 1 Gemma prize ($1000 Best Use of Gemma via Fireworks) is in play
  automatically: our top-ranked escalation target is gemma-4-31b-it.
- Our ranking orders it correctly (locked by test_actual_track1_list_ordering).
- Gemma-4 models 404 on a personal Fireworks account (deployed only behind the judging proxy);
  live run validated the demotion path: 3x404 → minimax-m3 answered 8/8 cleanly, no
  reasoning leakage in content, ~3065 tokens (thinking model ≈ +44% vs gpt-oss baseline).
- Deadline signal from web: event "runs to 11 July 2026" — CONFIRM in Event Schedule;
  if true, freeze criterion (T-24h) lands 2026-07-10.

## sub2 policy change (2026-07-09): escalation output policy

Official tutorial intel (lablab fine-tune-query-router guide): Participant FAQ names
**MiniMax and Kimi K series** as the allowed Fireworks models → all-thinking ALLOWED_MODELS
is now the BASE scenario, not worst case. Local-model phase = critical path.

One logical change vs sub1 — escalation output policy, measured live on minimax-m3, 8 practice tasks:
| config | tokens | answers intact |
|---|---|---|
| cap300, no terse (sub1) | 3065 | 5/8 (3 truncated mid-content) |
| terse + thinking-cap 600 | 4411 | 8/8 but codegen paid retry |
| terse + thinking-cap 1000 (sub2) | 3573 | 8/8 |

TERSE_SUFFIX cut simple-task completions ~40-60%; thinking cap 1000 avoids the
empty-content->retry burn. Non-thinking cap stays 300.

## Submission live (2026-07-09)

Team **Momentum** (Bogdan + Nick), submission page:
https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/momentum/verification-driven-token-efficient-routing-agent
Image submitted: ghcr.io/bogdan-lmk/veriroute:sub2. Awaiting automated scoring.

Competitive intel from neighboring submissions: verification-cascade is now the common meta
("prove or escalate", "local first + verify", claims of 2/3 tasks at zero tokens) — BUT the
loudest competitor (TokenRouter prove-or-escalate) sits at ACCURACY_GATE_FAILED 0.0% on the
leaderboard. Strategy is table stakes; execution through the gate is the differentiator.
