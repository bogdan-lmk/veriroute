"""Single source of runtime tunables.

Every knob lives here with its env override and its reason. Code imports
names, not numbers — the review found timeouts scattered as bare literals
(26.0/50.0/20.0/45.0), half env-driven and half hardcoded.
"""
from __future__ import annotations

import os


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# --- wall clock (grader kills at 600s; responses capped at 30s each) -------
TOTAL_BUDGET_S = _f("AGENT_TOTAL_BUDGET_S", 510.0)
FLUSH_MARGIN_S = _f("AGENT_FLUSH_MARGIN_S", 45.0)
PER_TASK_S = _f("AGENT_PER_TASK_S", 28.0)          # 30s/request rule - margin

# Track 2 pays no external flush drain — it may use more of the 600s cap.
T2_TOTAL_BUDGET_S = _f("AGENT_T2_TOTAL_BUDGET_S", 560.0)
T2_FLUSH_MARGIN_S = _f("AGENT_T2_FLUSH_MARGIN_S", 30.0)
T2_PER_TASK_S = _f("AGENT_T2_PER_TASK_S", 60.0)

# --- token economy ----------------------------------------------------------
# Runaway guard, NOT an optimization target: exhausting it stubs answers and
# forfeits the accuracy gate (measured: 9/19 with a 4k budget).
TOKEN_BUDGET = _i("AGENT_TOKEN_BUDGET", 15000)
MAX_COMPLETION_TOKENS = _i("AGENT_MAX_COMPLETION_TOKENS", 300)
# Thinking models bill hidden reasoning as completion tokens; cap 600 starved
# codegen into an empty-content paid retry (measured 4411 vs 3573 at 1000).
MAX_TOKENS_THINKING = _i("AGENT_MAX_TOKENS_THINKING", 1000)

# --- network ----------------------------------------------------------------
HTTP_TIMEOUT_S = _f("AGENT_HTTP_TIMEOUT_S", 25.0)
DRAIN_CALL_TIMEOUT_S = _f("AGENT_DRAIN_CALL_TIMEOUT_S", 20.0)
ESCALATION_VLM_TIMEOUT_S = _f("AGENT_ESCALATION_VLM_TIMEOUT_S", 45.0)

# --- local inference --------------------------------------------------------
LOCAL_START_WAIT_S = _f("AGENT_LOCAL_START_WAIT_S", 55.0)   # 60s-ready rule
LOCAL_RESTART_WAIT_S = _f("AGENT_LOCAL_RESTART_WAIT_S", 30.0)
LOCAL_PREWARM_TIMEOUT_S = _f("AGENT_LOCAL_PREWARM_TIMEOUT_S", 45.0)
LAST_RESORT_TIMEOUT_S = _f("AGENT_LAST_RESORT_TIMEOUT_S", 20.0)
LAST_RESORT_MAX_TOKENS = _i("AGENT_LAST_RESORT_MAX_TOKENS", 180)
# Executed categories leave room for sandbox runs inside the per-task cap.
EXEC_GEN_WINDOW_S = _f("AGENT_EXEC_GEN_WINDOW_S", 26.0)
EXEC_RESERVE_S = _f("AGENT_EXEC_RESERVE_S", 3.0)

# --- captioner (Track 2) ----------------------------------------------------
T2_STYLE_CALL_CAP_S = _f("AGENT_T2_STYLE_CALL_CAP_S", 50.0)
T2_CLIP_FETCH_TIMEOUT_S = _f("AGENT_T2_CLIP_FETCH_TIMEOUT_S", 60.0)
T2_FRAMES_PER_CLIP = _i("AGENT_T2_FRAMES_PER_CLIP", 2)
T2_DESCRIBE_MAX_TOKENS = _i("AGENT_T2_DESCRIBE_MAX_TOKENS", 90)
T2_STYLE_MAX_TOKENS = _i("AGENT_T2_STYLE_MAX_TOKENS", 220)
