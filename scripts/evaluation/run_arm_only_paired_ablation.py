"""Run a paired-only arm-rendering ablation over frozen motion snapshots."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laban_rl.perceptual_bandit.experiment import cases_from_matrix
from laban_rl.perceptual_bandit.paired_preference import (
    GeminiPairedPreferenceEvaluator,
    MockPairedPreferenceEvaluator,
    PairedPreferenceCache,
    PreferencePair,
    run_paired_preference_experiment,
)
from scripts.evaluation.run_paired_perceptual_experiment import (
    _load_motion_snapshot,
)
from scripts.train_cem_contextual_bandit import _build_optimiser_overrides


FIXED_CAMERA_LIMITS = ((-0.63, 0.63), (-0.63, 0.63))


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--motion-source-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--evaluator", choices=("mock", "gemini"), default="mock")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    return args


def main() -> int:
    args = parse_args()
    matrix = json.loads(_resolve(args.matrix).read_text(encoding="utf-8"))
    cases = []
    for case in cases_from_matrix(matrix):
        optimiser_args = SimpleNamespace(
            maxiter=45,
            popsize=5,
            local_maxiter=100,
            de_mutation=0.5,
            de_recombination=0.65,
            seed=case.seed,
            target_state=case.context.target_state,
            wave_flow_target_weight=0.35,
        )
        overrides = _build_optimiser_overrides(
            optimiser_args,
            case.context.gesture,
        )
        overrides.update(case.optimizer_overrides)
        cases.append(replace(case, optimizer_overrides=overrides))

    source = _resolve(args.motion_source_dir)
    destination = _resolve(args.out)
    results = {}
    for case in cases:
        snapshot = _load_motion_snapshot(
            case,
            source / case.candidate_id,
        )
        if snapshot is None:
            raise RuntimeError(
                f"Frozen motion snapshot is missing for {case.candidate_id!r}."
            )
        render_dir = destination / "candidates" / case.candidate_id
        render_dir.mkdir(parents=True, exist_ok=True)
        results[case.candidate_id] = replace(
            snapshot,
            output_dir=render_dir,
        )

    grouped = {}
    for case in cases:
        grouped.setdefault(case.candidate_id.rsplit("__", 1)[0], []).append(case)
    pairs = []
    for pair_id, pair_cases in sorted(grouped.items()):
        by_name = {case.candidate_name: case for case in pair_cases}
        if set(by_name) != {"reference", "styled"}:
            raise ValueError(
                f"Pair {pair_id!r} does not contain reference and styled."
            )
        reference_case = by_name["reference"]
        styled_case = by_name["styled"]
        pairs.append(
            PreferencePair(
                pair_id=pair_id,
                context=styled_case.context,
                reference_result=results[reference_case.candidate_id],
                styled_result=results[styled_case.candidate_id],
                target_layers=styled_case.target_layers,
            )
        )

    if args.evaluator == "mock":
        evaluator = MockPairedPreferenceEvaluator()
    else:
        evaluator = GeminiPairedPreferenceEvaluator(
            model=args.model,
            temperature=args.temperature,
            video_duration_seconds=2.0,
            video_lead_in_seconds=0.5,
            video_repetitions=2,
            video_inter_repeat_transition_seconds=0.5,
            video_final_hold_seconds=0.5,
            video_presentation_style="arm_only",
            fixed_camera_limits=FIXED_CAMERA_LIMITS,
            keep_uploaded_files=False,
        )
    try:
        payload = run_paired_preference_experiment(
            pairs,
            evaluator=evaluator,
            cache=PairedPreferenceCache(_resolve(args.cache)),
            repeats=args.repeats,
            out_path=destination / "paired_preferences.json",
        )
    finally:
        close = getattr(evaluator, "close", None)
        if close is not None:
            close()

    manifest = {
        "format_version": 1,
        "motion_source_dir": str(source),
        "matrix": str(_resolve(args.matrix)),
        "presentation_style": "arm_only",
        "background": "white",
        "trajectory_trail": False,
        "axes": False,
        "grid": False,
        "title": False,
        "legend": False,
        "time_annotation": False,
        "camera_limits": FIXED_CAMERA_LIMITS,
        "gesture_duration_seconds": 2.0,
        "lead_in_seconds": 0.5,
        "repetitions": 2,
        "inter_repeat_transition_seconds": 0.5,
        "final_hold_seconds": 0.5,
        "pair_count": len(pairs),
        "repeats": args.repeats,
        "planned_evaluator_calls": len(pairs) * args.repeats,
        "result_status": payload["status"],
    }
    manifest_path = destination / "ablation_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(destination / "paired_preferences.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
