# veriroute

**Verification-driven, token-efficient routing agent** — AMD Developer Hackathon: ACT II, Track 1
(Hybrid Token-Efficient Routing Agent).

Local models answer for free; their answers are **verified by code** (executing generated
programs, checking formats); the Fireworks API is called only when verification fails or the
category is provably beyond a small local model. The routing decisions live in a deterministic
policy table — the LLM never decides its own escalation.

## How it works

```
/input/tasks.json
  → validate input (defensive schema checks)
  → write stub results.json immediately (crash-safe from the first millisecond)
  → classify each task by category (deterministic rules)
  → policy table: LOCAL | LOCAL+VERIFY | ESCALATE
      LOCAL     — llama.cpp server inside the container (0 tokens)
      VERIFY    — execute generated code / check formats; fail → escalate
      ESCALATE  — cheapest sufficient model from ALLOWED_MODELS via FIREWORKS_BASE_URL
  → atomically rewrite results.json after every task
/output/results.json
```

Guardrails: global 8.5-minute internal deadline with a parallel drain of unfinished tasks,
per-task time budget, hard token budget stop, `ALLOWED_MODELS` invariant asserted before any
network call, SIGTERM-safe atomic writes, exit 0 in every failure mode.

> Phase 0 status: the escalate-all baseline (no local model yet). The llama.cpp runtime is
> already packaged; the local cascade lands next.

## Run

The grading harness injects `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL` and `ALLOWED_MODELS`,
mounts `/input` and `/output`, and runs the container:

```bash
docker run --rm --platform linux/amd64 --memory=4g --cpus=2 \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/... \
  -v $PWD/input:/input:ro -v $PWD/output:/output \
  ghcr.io/bogdan-lmk/veriroute:latest
```

Without the env vars the agent still exits 0 with a valid (stub) `results.json`.

## Development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/); Docker for packaging.

```bash
make test           # unit + integration tests (no network, no credentials needed)
make build          # build linux/amd64 image
make smoke          # run the image under the grader's resource limits on practice tasks
make run-practice   # run locally against eval/practice_tasks.json (uses your env creds)
```

Layout:

| Path | Purpose |
|---|---|
| `agent/main.py` | Orchestration: guardrails, sequential pass, parallel drain |
| `agent/io_utils.py` | Defensive input parsing, atomic crash-safe output |
| `agent/fireworks_client.py` | The only network choke point; ALLOWED_MODELS invariant, retries, token meter |
| `agent/model_ranking.py` | ALLOWED_MODELS parsing + token-oriented (not price-oriented) ranking |
| `agent/deadline.py` | Wall-clock budgets |
| `eval/practice_tasks.json` | Official practice tasks (I/O contract fixtures) |
| `journal.md` | Submission log: one change per submission, prediction vs leaderboard |

## Configuration (env, all optional)

| Variable | Default | Meaning |
|---|---|---|
| `AGENT_TOTAL_BUDGET_S` | 510 | Internal wall-clock budget (grader kills at 600) |
| `AGENT_FLUSH_MARGIN_S` | 45 | When only this much is left, drain remaining tasks in parallel |
| `AGENT_PER_TASK_S` | 30 | Per-task time budget |
| `AGENT_TOKEN_BUDGET` | 4000 | Hard stop for Fireworks tokens (leader is 5121) |
| `AGENT_HTTP_TIMEOUT_S` | 25 | Per-call HTTP timeout |
| `AGENT_MAX_COMPLETION_TOKENS` | 300 | max_tokens per escalated call |
| `AGENT_INPUT_PATH` / `AGENT_OUTPUT_PATH` | `/input/tasks.json` / `/output/results.json` | Path overrides for tests |
