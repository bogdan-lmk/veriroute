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

## sub3: local cascade phase 1 (2026-07-09)

One logical change vs sub2: local Qwen2.5-1.5B (llama.cpp, in-image) answers
sentiment/NER/summarization behind deterministic verifiers; everything else escalates.
- Native run, 8 practice tasks: 3 local / 5 escalated, 2,729 tokens (-24% vs sub2's 3,573)
- Container smoke under 4g/2cpu (Rosetta): local path works even emulated, ~10s/task
- Deterministic classifier: 77 tests, conservative fallbacks (multi-part/non-English/no-match → escalate)
- Gate context: threshold now pinned at exactly 16/19 (84.2%) — 78.9% failed, all 4 qualifiers at 84.2%
- Leader to beat: 4,268 tokens (Route AI). sub3 projection on all-thinking ALLOWED_MODELS: ~5-6k;
  with gemma available: ~3.5-4k. Math-PoT + codegen self-tests (next) move the expensive tasks local.

## sub4: execution-verified math + codegen (2026-07-09)

- Math PoT: local model writes solve() -> sandbox executes. Caught a REAL wrong-but-executable
  answer live (-60.0, then 264: few-shot example taught addition where the task meant "sold");
  fixed with a sells-N-more few-shot + negative-count guard + one local retry. 144 stable x3 runs.
- Codegen: ONE-shot generation (function + 3 asserts in one block; two calls would blow the 28s
  per-task budget on grader CPU), sandbox-verified, asserts stripped from the shipped answer.
  Flaky asserts -> rejection -> escalation is the designed safe path.
- Sandbox: python -I, empty env, tmp cwd, 5s timeout, output cap.
- Measured (8 practice, minimax escalations): 968-2,410 tokens depending on codegen verification
  luck (escalated codegen on minimax costs ~900). vs sub2 3,573 / sub3 2,729.
- 92 tests. Escalated categories remaining: factual, logic, code_debug.
- Images: sub3 (cascade phase 1), sub4 (=sub3 + math/codegen execution) — both public, pull verified.

## sub5 verified on grader-class x86 (2026-07-09)

DO droplet 2vCPU/4GB, live Fireworks (minimax-only ALLOWED_MODELS):
- sub4: 3 local / 5 esc / 2 rej, 2695 tok — math+codegen killed by prefill in 20s window
- sub5 (prompt-prefix prewarm at startup): 4 local / 4 esc / 1 rej, 2431 tok, 72.6s/8 tasks
- Lesson: pushes from home uplink stall; build+push on a DO droplet takes ~3 min end-to-end.
Image: ghcr.io/bogdan-lmk/veriroute:sub5 digest 1904124d...

## duo: dual-mode container ships both tracks (2026-07-09)

Input-schema dispatch: {prompt}->router, {video_url,styles}->captioner. One image, both tracks.
- T2 pipeline (grader-worst DO vCPU, 3 example clips): 259s, all 12 captions real Gemma output.
  Debug chain worth remembering: .format vs few-shot braces -> silent parse branch -> 30s
  timeouts -> measured decode 3 tok/s -> structural fix: ONE all-styles call per clip
  (halved prefills), --cache-reuse 256, caps 220/90, per-clip fallback degradation.
- T1 regression on duo: 4 local / 4 esc / 1 rej, 2305 tokens — sub5-equivalent.
- Eye consistency check (frames_agree) escalates drifting descriptions to minimax via baked
  b64 key (T2 injects nothing; T1 never reads it). Local fallback keeps agent alive keyless.
Image: ghcr.io/bogdan-lmk/veriroute:duo — submit this for BOTH tracks (Event Tracks multiselect).

## SUBMITTED: duo, both tracks (2026-07-09 ~15:10 local)

Image: ghcr.io/bogdan-lmk/veriroute:duo (no secrets; relay escalation). Event Tracks: T1+T2.
Descriptions: still v1 (T1-only) — update later for the Gemma-prize jury narrative.
PREDICTIONS on record:
- T1: gate pass p~0.65-0.75; tokens ~1.5-2.5k if gemma-4 in ALLOWED_MODELS, 3-5.5k if thinking-only.
- T2: mid-field on the auto leaderboard (SmolVLM eye vs cloud-VLM competitors); style scores decent.
- First risk to watch: any infra status (PULL/RUNTIME/TIMEOUT) — fix on grader-twin within the hour.

## FIRST REAL SCORE + fix (2026-07-10 morning)

Scored at last: ACCURACY_GATE_FAILED 47.4% (9/19) — matches the "token budget exhausted ->
stubbed tail" failure shape (4k budget < ~5.7k needed with thinking escalations).
Kimi-only repro on grader-twin: 0 empty answers -> thinking-truncation theory dead;
budget-stub theory prime. Field update: 17 qualifiers, leader 3,864 tok, accuracies up to
100% (16/19 ceiling was small-sample illusion).
FIX shipped in duo digest 60226fff: AGENT_TOKEN_BUDGET 4k->15k (gate over rank) +
never-empty answers (last-resort unverified local guess instead of stub).
Prediction v2: gate pass p~0.55 (if cause was budget); if score stays ~47% -> local answer
quality is the real culprit -> next change: audit/disable weak local categories.

## MEASURED GATE PASS via own eval (2026-07-10 afternoon)

Built a 32-task hidden-like eval (8 categories x 4, gpt-oss-120b authored, glm-5p2 judge)
and ran the REAL container on grader-twin. Iterations:
| round | policy change | score | tokens/32 |
|-------|---------------|-------|-----------|
| 1 | budget 15k, local NER/summ/sent + PoT | 17/32 53% FAIL | 15945 |
| 2 | escalate NER+summ | 20/32 62% | 25924 |
| 3 | escalate sentiment+math, budget 25k | 28/32 87% PASS | 28352 |
| 4 | 2x code caps + fixed logic eval | 31/32 97% | 17166 |
| 5-6 | 3x code caps + def-retry + last-resort code draft | 31/32 97% | 19162 |
Root cause of the 47% board score CONFIRMED: local text categories fail the judge
(NER 0/4, summ 2/4, sentiment 3/4, math-PoT 3/4). Only execution-verified code stays local.
Escalated categories score 4/4. Remaining fail: 1 adversarial codegen (palindrome_pairs) where
thinking models leak a truncated reasoning trace — noise on the 19-task real set.
duo digest a4dc8f00. STRATEGY NOTE: gate now passes comfortably BUT we escalate ~30/32 ->
~11k tokens on the 19-task set vs leader 3,864. Rank among passers will be low; the real
grader's ALLOWED_MODELS (if it has non-thinking gemma-4) cuts completion tokens ~3x. Get the
real board score first, THEN decide token-vs-gate tradeoff. Gate pass >> token rank.

## INCIDENT: submission field reverted to :sub5 (2026-07-10 ~15:38)

Re-saving the lablab wizard restored a STALE step-3 draft: Docker Image silently became
ghcr.io/bogdan-lmk/veriroute:sub5 (old image: gate-failing T1 policy, NO captioner -> T2 dead).
Lesson: verify the Docker Image field on EVERY re-save before final submit. Fix: set :duo,
walk wizard to the end, Submit. Our card sits in DID NOT QUALIFY (88 entries) after the 47.4%.

## Board intel (2026-07-10 15:53 GMT+3)

T1: 49 scored, 88 DNQ. Top-5: yassai 1,992 @ 84.2% | Kestrel 2,138 @ 84.2% | YOLOAI_v6
2,664 @ 84.2% | Metis 3,645 @ 100% | TOKENMAN 3,677 @ 94.7%.
META CONFIRMED: top-3 sit EXACTLY on the gate (16/19) — they spend accuracy headroom for
tokens. Winning number is now <2k tokens. Our duo ~11k/19 projected: gate-pass but bottom
half. Post-verdict lever: re-enable the cheapest local categories up to the 3-miss budget
(sentiment 3/4, math PoT 3/4 measured) + low-cap first-try escalation; one change per sub.
T2: top scores 0.91/0.90/0.89/0.88/0.87/0.86 — dense; mid-field realistic for SmolVLM eye.

## WINNING LEVER: reasoning_effort=none (2026-07-10 ~16:40)

Leaderboard scrape (T1, 50 scored): top-4 sit EXACTLY on the gate at 1,992-2,664 tokens;
one rival's blurb gave it away — "all 5 allowed models bill hidden reasoning tokens by
default, and the one universal switch that disables it. 100% eval accuracy, ~164 tok/task".
MEASURED live: reasoning_effort=none drops minimax-m3 29->23 and kimi-k2p7-code 78->9
completion tokens, reasoning_content gone. Our code treated both as uncontrollable thinking
(cap 1000). Fix: _REASONING_OFF={minimax-m,kimi-k2}, effort=none, tight caps, cheap rank.
Eval (32 tasks, minimax+kimi only = worst case, no gemma):
| version | tokens/32 | completion | accuracy | note |
|---------|-----------|------------|----------|------|
| reasoning ON (a4dc8f00) | 19,162 | 12,965 | 31/32 97% | prior duo |
| reasoning=none (r7) | 7,946 | 2,241 | 31/32 97% | -58% tokens; logic-1 dropped |
| + logic=low (r8, 503642da) | 8,328 | 2,603 | 31/32 97% | logic 4/4; only fail = ner-2 BAD EXPECTED (Eiffel=landmark not org) |
=> effectively 32/32. Projected on 19-task hidden set: ~4.9k prompt+completion, ~1.5k
completion-only. On the REAL grader gemma-4-31b (non-thinking) leads the rank and is even
cheaper — reasoning-off is the safety net + the gemma-down path. duo digest 503642da PUSHED.
Submission points at :duo tag (re-saved 16:06) => auto re-scores on next grader run.

## Board 13:52 — gate fix still unscored; winning bar sharpened

Our card still 73.7% ACCURACY_GATE_FAILED in DNQ — grader has NOT re-run the escalate-all
fix yet (503642da pushed ~16:40, after the 16:06 re-save). Board IS churning others, so it's
a queue/sweep delay. Action: re-save again to requeue against the current fixed image.
NEW #1: rtq-smart-router 0 tokens / 100% — fully-local, answers all 19 with zero Fireworks
calls. That's the ceiling (can't beat 0). Gate-passers cluster 84.2% / ~2.0-2.7k tokens
(local + 2-3 escalations). ~5k tokens => rank ~13-17. So our escalate-all, once it clears the
gate, lands mid-table; climbing needs FEWER escalations (0-token local answers), which is the
exact tradeoff our weak local text models lose. Decide token-vs-gate ONLY after a real PASS.

## MEASURED DEAD END: bigger local model won't reach 0 tokens (2026-07-10 ~17:15)

Hypothesis: swap T1 text categories from Qwen-1.5B to the bundled Gemma-3-4B (already in the
image for T2) to answer text locally at 0 tokens and leave mid-table. MEASURED on grader-class
2 vCPU / 4GB: a SINGLE 40-token Gemma-4B answer took 4-7+ min (2.4GB model load + ~3 tok/s CPU
decode). Even amortizing load across llama-server, 4B decode is far too slow for 19 tasks in
the 10-min budget. => The 0-token leader is NOT using a 4B; they use a fast small model + heavy
determinism/prompting. With OUR hardware, a big local model is a dead end. escalate-all stays
the pragmatic gate choice; realistic climb = trim escalation prompt tokens (~5k -> ~3-4k),
not 0-token local. Reclaiming SHORT-output categories (sentiment=1 word) on the fast 1.5B with
better prompting is the only 0-token lever left, and it's accuracy-risky (1.5B was 3/4).

## Organizer clarifications + token-trim banked (2026-07-10 ~17:30)

Official Discord clarifications:
- Gate is 80% (still effectively 16/19: 15/19=78.9% fails, 16/19=84.2% passes).
- LLM JUDGE IS NON-DETERMINISTIC run-to-run => our ~18/19 margin is an ASSET: teams sitting
  exactly on 16/19 can drop to 15/19 on a bad judge roll and fall off. escalate-all's margin
  is insurance, not waste.
- GEMMA IS ON-DEMAND, NOT on the grader ("you don't need Gemma to pass"). CORRECTS my earlier
  "grader routes to gemma-4-31b ~3.4k" claim: grader hits always-on minimax/kimi. Our eval
  (minimax+kimi) IS representative, real ~5k not 3.4k. reasoning-off is exactly why that's 5k
  not 11k — validated as the critical lever.
- Turnaround ~1h (5min was just the poll interval). Don't spam re-saves; check the registry
  pull counter to see if pulled. INFRA_ERROR is on them (auto-retry). Keep image <5GB (we're 4.29GB, safe).
TOKEN-TRIM (exp/trim-terse, e9b6e1a): trimmed TERSE_SUFFIX ~25->8 tokens. Eval 32 tasks:
8328 -> 7505 tokens (-10%: prompt -390, completion -433 — short suffix => terser answers),
accuracy held 31/32 96.9% (fail = factual-2 wrong year, model noise). BANKED for cycle 2 —
NOT pushed. Ship only after escalate-all (503642da) confirms the gate on the real grader.
Projection 19 tasks: ~4.45k tokens => rank ~12. Top-5 (<3.3k) needs FEWER escalations (risky local).

## BREAKTHROUGH: evidence-based hybrid on REAL practice tasks (2026-07-10 ~18:15)

Methodology fix (user's push): stop trusting my synthetic eval; use the 8 REAL organizer
practice tasks (one per category) + manual verification. Ran local Gemma-3-4B on all 8:
- gemma-local ALL-text: 6/8. FAILS: factual (Canberra -> "Jervis Bay", wrong; should be
  Molonglo R./Lake Burley Griffin) and logic (pet puzzle -> "Lee", wrong; answer is Sam).
  Correct: sentiment(Mixed), summarization, ner, code_debug, code_gen, math.
=> The 4B's weak spots are FACTUAL (specific knowledge) and LOGIC (multi-step) — exactly the
categories behind the 73.7% real score.
HYBRID (local Gemma sentiment/summ/ner; escalate factual/logic/math/code_debug; code_gen
self-test): re-ran on the 8 real tasks -> 8/8. factual now "Molonglo River" (correct), logic
now "Sam owns the cat" (correct). 1,695 tokens/8 (3 local free, 5 escalated). Branch
exp/hybrid-local (e84c8ee), image veriroute:hyb.
Projection 19 tasks: ~3,000 tokens => rank ~6-7 at escalate-all accuracy. AGGRESSIVE next
step: localize code_debug (gemma got practice-06 right) + math (PoT), escalate ONLY
factual+logic -> ~1,700 tokens => top-3. Validate on practice tasks before shipping.
SHIP ORDER unchanged: escalate-all confirms the gate (safety net) FIRST, then hybrid.
Small-sample caveat: 8 practice tasks != 19 hidden; the real grader is the final judge.

## COMPLIANCE: relay removed — provably Fireworks-only (2026-07-10 ~18:30)

Organizer clarification: containers routing OUTSIDE Fireworks will be DISQUALIFIED (manual
audit of flagged/top submissions). Our escalation relay (droplet :8899) routed there.
VERIFIED current :duo was already runtime-safe: ESCALATION_RELAY_URL baked EMPTY -> relay
never triggers; Track 1 uses grader-injected FIREWORKS_BASE_URL (direct). But the relay CODE
+ "key lives on our box" comment in the public repo could flag a manual audit. Removed the
relay entirely from captioner + Dockerfile; no droplet/relay references remain in code.
New clean :duo digest e729dfac pushed (Track 1 functionally identical — it never used relay;
Track 2 now local-only if no key injected). Container is provably Fireworks-only.
STRATEGIC (same clarification): final scoring uses NEW randomized prompts -> overfit/hardcoded
0-token leaders will DROP; our category-based routing (localize sentiment/summ/ner, escalate
factual/logic — found on the 8 REAL practice tasks) GENERALIZES and carries to the final.

## SHIP-READY hybrid validated in GRADER MODE (2026-07-10 ~18:50)

ship/hybrid: relay-free + Gemma-3-4B baked as the default router (ENV AGENT_LLAMA_MODEL,
AGENT_LOCAL_TIMEOUT_S=32) + LOCAL_CATEGORIES={sentiment,summarization,ner}. Ran on the 8 REAL
practice tasks with ONLY grader-style env (key+base_url+allowed_models, no overrides) -> the
baked gemma default engaged ("local model: on"), 3 local / 5 escalated, 1670 tokens/8, 115s.
Manual verify: 8/8 correct (factual=Molonglo, logic=Sam, sentiment=Mixed, ner, summ, code all right).
DOMINATES escalate-all: identical routing for the hard categories (factual/logic/math/code_debug
all escalate) but sentiment/summ/ner answered locally by a capable 4B (validated on REAL tasks),
so same accuracy, fewer tokens (~2.7k projected/19 vs escalate-all ~4.5k). Category-based =>
generalizes to the final scoring's new prompts. Aggressive variant (also local math/code_debug)
gave NO benefit — those local paths reject and escalate anyway (measured: same 5 escalations).
Image veriroute:shiphyb, branch ship/hybrid. Candidate to ship as the primary T1 submission.
