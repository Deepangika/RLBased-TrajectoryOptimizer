#!/usr/bin/env python3
"""Run a resumable, presentation-ready multi-context validation suite."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_cem_contextual_bandit.py"
HOLDOUT_SCRIPT = PROJECT_ROOT / "scripts" / "debug" / "evaluate_fixed_profile_holdout.py"
SUMMARY_SCRIPT = PROJECT_ROOT / "scripts" / "summarize_supervisor_validation_suite.py"

PRESETS = {
    # Covers all three gestures with contexts already useful for a short demo.
    "quick": [
        ("wave", "friendly"),
        ("reach", "confident"),
        ("point", "hesitant"),
    ],
    # Covers every affective state once and every gesture twice.
    "supervisor": [
        ("wave", "friendly"),
        ("wave", "calm"),
        ("reach", "confident"),
        ("reach", "angry"),
        ("point", "hesitant"),
        ("point", "confused"),
    ],
    "all": [
        (gesture, state)
        for gesture in ("wave", "reach", "point")
        for state in ("confident", "calm", "hesitant", "friendly", "confused", "angry")
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, independently select, freeze, hold out, and summarize multiple contexts."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="supervisor")
    parser.add_argument(
        "--contexts",
        nargs="*",
        help="Optional explicit gesture:state entries; overrides --preset.",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples-per-round", type=int, default=6)
    parser.add_argument("--training-repeats", type=int, default=3)
    parser.add_argument("--validation-repeats", type=int, default=10)
    parser.add_argument("--validation-top-k", type=int, default=3)
    parser.add_argument("--holdout-repeats", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=90)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--local-maxiter", type=int, default=300)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Intentionally replace completed context and holdout results.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not make evaluator calls; rebuild the consolidated outputs only.",
    )
    return parser.parse_args()


def parse_contexts(args: argparse.Namespace) -> list[tuple[str, str]]:
    if not args.contexts:
        return list(PRESETS[args.preset])
    contexts: list[tuple[str, str]] = []
    valid_gestures = {"wave", "reach", "point"}
    valid_states = {"confident", "calm", "hesitant", "friendly", "confused", "angry"}
    for item in args.contexts:
        try:
            gesture, state = item.lower().split(":", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid context {item!r}; expected gesture:state") from exc
        if gesture not in valid_gestures or state not in valid_states:
            raise ValueError(f"Unsupported context {item!r}")
        contexts.append((gesture, state))
    return contexts


def run_command(command: list[str], log_path: Path) -> None:
    print("\n$ " + " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}; see {log_path}")


def training_command(args: argparse.Namespace, gesture: str, state: str, out_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--gesture", gesture,
        "--target-state", state,
        "--rounds", str(args.rounds),
        "--cem-samples-per-round", str(args.samples_per_round),
        "--cem-elite-fraction", "0.4",
        "--cem-initial-width", "0.15",
        "--exploration-decay-rate", "0.98",
        "--cem-smoothing", "0.7",
        "--cem-min-std", "0.03",
        "--cem-min-elites", "3",
        "--repeats", str(args.training_repeats),
        "--validation-repeats", str(args.validation_repeats),
        "--validation-top-k", str(args.validation_top_k),
        "--evaluator", "gemini",
        "--model", args.model,
        "--temperature", str(args.temperature),
        "--maxiter", str(args.maxiter),
        "--popsize", str(args.popsize),
        "--local-maxiter", str(args.local_maxiter),
        "--seed", str(args.seed),
        "--realisation-penalty-weight", "0.25",
        "--max-feature-error-threshold", "0.10",
        "--max-feature-error-penalty-weight", "0.50",
        "--wave-flow-target-weight", "0.35",
        "--stability-penalty-weight", "0.25",
        "--reward-margin-mode", "raw",
        "--out", str(out_dir),
    ]
    if args.force:
        command.append("--overwrite")
    return command


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.out).resolve()
    suite_dir.mkdir(parents=True, exist_ok=True)
    contexts = parse_contexts(args)

    planned_training = args.rounds * args.samples_per_round * args.training_repeats
    planned_validation = (args.validation_top_k + 2) * args.validation_repeats
    planned_per_context = planned_training + planned_validation + args.holdout_repeats
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "contexts": [{"gesture": g, "target_state": s} for g, s in contexts],
        "seed": args.seed,
        "model": args.model,
        "temperature": args.temperature,
        "protocol": {
            "rounds": args.rounds,
            "samples_per_round": args.samples_per_round,
            "training_repeats": args.training_repeats,
            "validation_repeats": args.validation_repeats,
            "validation_top_k": args.validation_top_k,
            "holdout_repeats": args.holdout_repeats,
        },
        "planned_evaluator_calls_per_context": planned_per_context,
        "planned_evaluator_calls_total": planned_per_context * len(contexts),
        "headline_results_use_frozen_profile_holdout_only": True,
    }
    (suite_dir / "suite_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Contexts: {len(contexts)}")
    print(f"Planned evaluator calls: approximately {manifest['planned_evaluator_calls_total']}")
    print("Completed contexts are skipped unless --force is supplied.")

    if not args.summarize_only:
        started = time.monotonic()
        for index, (gesture, state) in enumerate(contexts, start=1):
            name = f"{gesture}_{state}_seed{args.seed}"
            experiment_dir = suite_dir / name
            summary_path = experiment_dir / "results_summary.json"
            holdout_path = experiment_dir / "fixed_profile_holdout.json"
            print(f"\n[{index}/{len(contexts)}] {gesture} -> {state}")

            if summary_path.exists() and not args.force:
                print(f"Training already complete; reusing {summary_path}")
            else:
                run_command(
                    training_command(args, gesture, state, experiment_dir),
                    suite_dir / "logs" / f"{name}_training.log",
                )

            if holdout_path.exists() and not args.force:
                print(f"Holdout already complete; reusing {holdout_path}")
            else:
                holdout_command = [
                    sys.executable,
                    str(HOLDOUT_SCRIPT),
                    "--experiment-dir", str(experiment_dir),
                    "--repeats", str(args.holdout_repeats),
                    "--model", args.model,
                    "--temperature", str(args.temperature),
                ]
                if args.force:
                    holdout_command.append("--overwrite")
                run_command(
                    holdout_command,
                    suite_dir / "logs" / f"{name}_holdout.log",
                )
        print(f"\nSuite evaluation elapsed time: {(time.monotonic() - started) / 60:.1f} minutes")

    run_command(
        [sys.executable, str(SUMMARY_SCRIPT), "--suite-dir", str(suite_dir)],
        suite_dir / "logs" / "summary.log",
    )


if __name__ == "__main__":
    main()
