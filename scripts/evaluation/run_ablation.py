"""Orchestrate the 2x2 ablation: one condition at a time, hard call budgets.

Design:
- Condition A (original target, original smoothness) reuses the two completed
  live runs and is never re-executed.
- Conditions B, C, D run via ``run_outer_learning_experiment.py`` with the
  frozen live protocol (identical CLI settings to Condition A, seed 7,
  110-call per-run ceiling).
- An aggregate ledger enforces the 660-call ablation budget across all new
  live runs before each launch.
- Live order: beckon-fear B, C, D, then wave-sadness B, C, D.

The script executes exactly one pending condition per invocation, records the
git commit, refuses to overwrite completed output, and supports ``--mock``
(mock evaluator, zero Gemini calls) and ``--dry-run`` (preflight only).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = ROOT / "configs" / "ablations" / "ablation_manifest_v1.json"
AGGREGATE_BUDGET = 660
PER_RUN_BUDGET = 110
FROZEN_CLI = [
    "--stage", "stage_a",
    "--model", "gemini-2.5-flash",
    "--temperature", "0.2",
    "--seed", "7",
    "--workers", "1",
    "--max-gemini-calls", str(PER_RUN_BUDGET),
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def run_calls_used(out_dir: Path) -> int:
    total = 0
    for budget_file in out_dir.rglob("gemini_call_budget.json"):
        payload = json.loads(budget_file.read_text(encoding="utf-8"))
        total += int(payload.get("total_calls", 0))
    return total


def condition_completed(out_dir: Path) -> bool:
    status = out_dir / "live_stage_a" / "stage_status.json"
    return status.exists()


def aggregate_calls(manifest: dict, *, suffix: str) -> int:
    total = 0
    for entry in manifest["live_conditions"]:
        out_dir = ROOT / (entry["out_dir"] + suffix)
        if out_dir.exists():
            total += run_calls_used(out_dir)
    return total


def preflight(manifest: dict, *, mock: bool, suffix: str) -> list[str]:
    problems: list[str] = []
    targets = ROOT / "configs" / "ablations" / "recalibrated_targets_v1.json"
    if not targets.exists():
        problems.append("Frozen recalibrated targets artifact is missing.")
    else:
        frozen = json.loads(targets.read_text(encoding="utf-8"))
        if frozen.get("status") != "frozen":
            problems.append("Recalibrated targets are not marked frozen.")
    for reference in manifest["condition_a_reference_runs"]:
        run_dir = ROOT / reference["out_dir"]
        if not (run_dir / "live_stage_a" / "stage_status.json").exists():
            problems.append(f"Condition A reference missing: {reference['out_dir']}")
    for entry in manifest["live_conditions"]:
        config_path = ROOT / entry["config"]
        if not config_path.exists():
            problems.append(f"Config missing: {entry['config']}")
    if not mock:
        used = aggregate_calls(manifest, suffix="")
        if used + PER_RUN_BUDGET > AGGREGATE_BUDGET:
            problems.append(
                f"Aggregate budget exhausted: {used} used of {AGGREGATE_BUDGET}; "
                f"a new run could add up to {PER_RUN_BUDGET}."
            )
        import os

        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            problems.append("GOOGLE_API_KEY / GEMINI_API_KEY is not set.")
    return problems


def next_pending(manifest: dict, *, suffix: str) -> dict | None:
    for entry in manifest["live_conditions"]:
        out_dir = ROOT / (entry["out_dir"] + suffix)
        if not condition_completed(out_dir):
            return entry
    return None


def append_ledger(record: dict) -> None:
    ledger_path = ROOT / "outputs" / "experiments" / "ablation_call_ledger_v1.json"
    ledger: list[dict] = []
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger.append(record)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(ledger_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true",
                        help="Run the next condition with the mock evaluator (zero Gemini calls).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preflight and report the next pending condition without running.")
    parser.add_argument("--resume", action="store_true",
                        help="Pass --resume to the runner for an interrupted condition.")
    args = parser.parse_args()

    manifest = load_manifest()
    suffix = "_mock" if args.mock else ""
    problems = preflight(manifest, mock=args.mock, suffix=suffix)
    if problems:
        for problem in problems:
            print(f"[preflight-FAIL] {problem}")
        return 1
    print("[preflight] all checks passed")

    if not args.mock:
        used = aggregate_calls(manifest, suffix="")
        print(f"[budget] aggregate Gemini calls used so far: {used}/{AGGREGATE_BUDGET}")

    entry = next_pending(manifest, suffix=suffix)
    if entry is None:
        print("[done] all ablation conditions are complete")
        return 0
    out_dir = entry["out_dir"] + suffix
    print(f"[next] {entry['condition_id']} -> {out_dir}")
    if args.dry_run:
        return 0

    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluation" / "run_outer_learning_experiment.py"),
        "--config", entry["config"],
        "--evaluator", "mock" if args.mock else "gemini",
        "--out", out_dir,
        *FROZEN_CLI,
    ]
    if args.resume:
        command.append("--resume")
    record = {
        "condition_id": entry["condition_id"],
        "config": entry["config"],
        "out_dir": out_dir,
        "evaluator": "mock" if args.mock else "gemini",
        "git_commit": _git_commit(),
        "started_at_utc": _utc(),
    }
    print("[run]", " ".join(command))
    completed = subprocess.run(command, cwd=ROOT)
    record["finished_at_utc"] = _utc()
    record["return_code"] = completed.returncode
    if not args.mock:
        record["calls_used"] = run_calls_used(ROOT / out_dir)
        record["aggregate_calls_after"] = aggregate_calls(manifest, suffix="")
    append_ledger(record)
    if completed.returncode != 0:
        print(f"[FAIL] condition {entry['condition_id']} exited "
              f"{completed.returncode}; stopping (use --resume after diagnosis).")
        return completed.returncode
    print(f"[ok] condition {entry['condition_id']} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
