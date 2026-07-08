"""Wall-clock budget management under the grader's 10-minute hard cap.

The internal budget (default 510 s = 8.5 min) leaves margin for container
start, model paging and the final write. The flush margin marks the point
where sequential work must stop and the remaining tasks get drained in
parallel or stubbed.
"""
from __future__ import annotations

import os
import time


class Deadline:
    def __init__(
        self,
        total_s: float | None = None,
        flush_margin_s: float | None = None,
        per_task_s: float | None = None,
    ):
        self._start = time.monotonic()
        self.total_s = (
            total_s
            if total_s is not None
            else float(os.environ.get("AGENT_TOTAL_BUDGET_S", "510"))
        )
        self.flush_margin_s = (
            flush_margin_s
            if flush_margin_s is not None
            else float(os.environ.get("AGENT_FLUSH_MARGIN_S", "45"))
        )
        self.per_task_s = (
            per_task_s
            if per_task_s is not None
            else float(os.environ.get("AGENT_PER_TASK_S", "30"))
        )

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def remaining(self) -> float:
        return max(0.0, self.total_s - self.elapsed())

    def flush_due(self) -> bool:
        """True once only the flush margin is left: drain, don't start new work."""
        return self.remaining() <= self.flush_margin_s

    def task_budget(self) -> float:
        """Wall-clock allowance for the next task, never past the flush point."""
        return max(0.0, min(self.per_task_s, self.remaining() - self.flush_margin_s))
