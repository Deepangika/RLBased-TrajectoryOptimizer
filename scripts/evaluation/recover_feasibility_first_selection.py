"""Recover final CEM selections from saved validation results without VLM calls."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.perceptual_bandit.selection import select_feasible_incumbent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment-dir")
    group.add_argument("--root", help="Recursively recover every completed seed directory.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def experiment_dirs(args: argparse.Namespace) -> list[Path]:
    if args.experiment_dir:
        return [Path(args.experiment_dir).resolve()]
    root = Path(args.root).resolve()
    return sorted(path.parent for path in root.rglob("independent_validation.json"))


def backup_once(path: Path) -> None:
    backup = path.with_name(path.stem + ".pre_feasibility_selection" + path.suffix)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def recover(directory: Path, *, dry_run: bool) -> None:
    validation_path = directory / "independent_validation.json"
    summary_path = directory / "results_summary.json"
    selected_path = directory / "selected_validated_profile.json"
    if (directory / "fixed_profile_holdout.json").exists():
        raise RuntimeError(
            f"Refusing to change a selection after holdout: {directory}. "
            "Preserve the scientific holdout and use a new experiment instead."
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    previous = validation.get("selection") or summary.get("independent_selection") or {}
    shortlist = previous.get("shortlist")
    if not shortlist:
        raise RuntimeError(f"No saved shortlist in {directory}")
    tolerance = float(summary.get("max_feature_error_threshold", 0.10))
    selection = select_feasible_incumbent(
        validation,
        shortlist,
        tolerance=tolerance,
    )
    selected = selection["selected"]
    result = dict(validation[selected["validation_result_key"]])
    old_source = (previous.get("selected") or {}).get("source", "unknown")
    print(
        f"{directory}: {old_source} -> {selected['source']} | "
        f"reward={selected['validation_reward']:.6f} | "
        f"strict={selection['strict_feasibility_satisfied']}"
    )
    if dry_run:
        return
    for path in (validation_path, summary_path, selected_path):
        backup_once(path)
    validation["selection"] = selection
    validation["best_sampled_profile"] = result
    summary["independent_selection"] = selection
    summary["best_sampled_profile_validation"] = result
    validation_path.write_text(json.dumps(validation, indent=2, allow_nan=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    selected_path.write_text(json.dumps(selected, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    directories = experiment_dirs(args)
    if not directories:
        raise SystemExit("No completed experiments found.")
    for directory in directories:
        recover(directory, dry_run=args.dry_run)
    print(f"Processed {len(directories)} experiment(s).")


if __name__ == "__main__":
    main()
