"""Tests for the hard Gemini call budget."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from laban_rl.perceptual_bandit.call_budget import (
    CallBudgetExhausted,
    GeminiCallBudget,
    get_active_budget,
    set_active_budget,
)

RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluation"
    / "run_outer_learning_experiment.py"
)
_RUNNER_SPEC = importlib.util.spec_from_file_location("outer_learning_runner_budget", RUNNER_PATH)
assert _RUNNER_SPEC is not None
assert _RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(runner)


def test_budget_allows_exactly_limit_calls(tmp_path: Path) -> None:
    budget = GeminiCallBudget(3, tmp_path / "budget.json")
    assert budget.charge() == 1
    assert budget.charge() == 2
    assert budget.charge() == 3
    with pytest.raises(CallBudgetExhausted):
        budget.charge()
    saved = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert saved["total_calls"] == 3
    assert saved["exhausted"] is True
    assert saved["remaining"] == 0


def test_budget_counts_categories_separately(tmp_path: Path) -> None:
    budget = GeminiCallBudget(10, tmp_path / "budget.json")
    budget.set_category("search")
    budget.charge()
    budget.charge()
    budget.set_category("validation")
    budget.charge()
    budget.set_category("paired")
    budget.charge()
    summary = budget.summary()
    assert summary["calls_by_category"] == {"search": 2, "validation": 1, "paired": 1}
    assert summary["total_calls"] == 4


def test_budget_persists_and_resumes(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    first = GeminiCallBudget(5, path)
    first.charge()
    first.charge()
    resumed = GeminiCallBudget(5, path)
    assert resumed.total == 2
    resumed.charge()
    resumed.charge()
    resumed.charge()
    with pytest.raises(CallBudgetExhausted):
        resumed.charge()


def test_active_budget_registry(tmp_path: Path) -> None:
    assert get_active_budget() is None or True  # do not assume prior state
    budget = GeminiCallBudget(1, tmp_path / "budget.json")
    set_active_budget(budget)
    try:
        assert get_active_budget() is budget
        budget.charge()
        with pytest.raises(CallBudgetExhausted):
            budget.charge()
    finally:
        set_active_budget(None)
    assert get_active_budget() is None


def test_budget_rejects_invalid_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        GeminiCallBudget(0, tmp_path / "budget.json")


def test_main_refuses_gemini_run_exceeding_ceiling(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    argv = [
        "--config", "configs/outer_learning_beckon_fear_robustness_v1.json",
        "--stage", "stage_a",
        "--evaluator", "gemini",
        "--rounds-min", "5",
        "--rounds-max", "5",
        "--samples-per-round", "8",
        "--elite-count", "3",
        "--candidate-vlm-repeats", "2",
        "--validation-top-k", "4",
        "--validation-vlm-repeats", "5",
        "--paired-validation-repeats", "5",
        "--workers", "1",
        "--max-gemini-calls", "50",
        "--out", str(out_dir.relative_to(tmp_path)),
    ]
    # Run from a temp-relative out path: main resolves out under repo ROOT,
    # so use a repo-relative temporary directory instead.
    argv[-1] = "outputs/tmp_test_budget_ceiling"
    try:
        with pytest.raises(RuntimeError, match="exceed the hard ceiling"):
            runner.main(argv)
    finally:
        import shutil

        created = Path(runner.ROOT) / "outputs" / "tmp_test_budget_ceiling"
        shutil.rmtree(created, ignore_errors=True)


def test_main_refuses_gemini_run_with_multiple_workers(tmp_path: Path) -> None:
    argv = [
        "--config", "configs/outer_learning_beckon_fear_robustness_v1.json",
        "--stage", "stage_a",
        "--evaluator", "gemini",
        "--workers", "2",
        "--max-gemini-calls", "110",
        "--out", "outputs/tmp_test_budget_workers",
    ]
    with pytest.raises(RuntimeError, match="--workers 1"):
        runner.main(argv)
    import shutil

    shutil.rmtree(Path(runner.ROOT) / "outputs" / "tmp_test_budget_workers", ignore_errors=True)


def test_budget_exhausted_outcome_is_not_successful() -> None:
    assert not runner._successful_outcome("call_budget_exhausted")
