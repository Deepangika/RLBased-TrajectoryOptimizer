"""Hard run-level Gemini call budget.

Every attempted ``generate_content`` request (including billable retries)
must be charged against the active budget *before* the request is issued.
When the ceiling is reached the charge raises :class:`CallBudgetExhausted`
before any further API call can be made, so a run can never exceed its
approved call count.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class CallBudgetExhausted(RuntimeError):
    """Raised before an API call that would exceed the hard call ceiling."""


class GeminiCallBudget:
    """Persistent counter with a hard ceiling on total Gemini API calls.

    ``charge()`` must be invoked immediately before every individual
    ``generate_content`` attempt, retries included. The counter state is
    written atomically to ``path`` after every charge so an interrupted run
    reports its true usage.
    """

    def __init__(self, limit: int, path: Path, *, category: str = "search") -> None:
        if int(limit) < 1:
            raise ValueError("Call budget limit must be at least 1.")
        self.limit = int(limit)
        self.path = Path(path)
        self.category = str(category)
        self.total = 0
        self.counts: dict[str, int] = {}
        self.exhausted = False
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            self.total = int(saved.get("total_calls", 0))
            self.counts = {str(k): int(v) for k, v in dict(saved.get("calls_by_category", {})).items()}
            self.exhausted = bool(saved.get("exhausted", False))
        self._persist()

    def set_category(self, category: str) -> None:
        self.category = str(category)

    def charge(self) -> int:
        """Consume one call. Raises before the call that would exceed the limit."""
        if self.total >= self.limit:
            self.exhausted = True
            self._persist()
            raise CallBudgetExhausted(
                f"Gemini call budget exhausted: {self.total}/{self.limit} calls "
                f"already made; refusing call in category '{self.category}'."
            )
        self.total += 1
        self.counts[self.category] = int(self.counts.get(self.category, 0)) + 1
        self._persist()
        return self.total

    def summary(self) -> dict[str, object]:
        return {
            "limit": self.limit,
            "total_calls": self.total,
            "calls_by_category": dict(self.counts),
            "remaining": max(0, self.limit - self.total),
            "exhausted": bool(self.exhausted),
        }

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.summary(), indent=2)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_name, self.path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


_ACTIVE_BUDGET: GeminiCallBudget | None = None


def set_active_budget(budget: GeminiCallBudget | None) -> None:
    global _ACTIVE_BUDGET
    _ACTIVE_BUDGET = budget


def get_active_budget() -> GeminiCallBudget | None:
    return _ACTIVE_BUDGET
