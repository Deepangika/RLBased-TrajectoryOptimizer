"""Collect once and compare VAD/categorical candidate rankings from one cache."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.optimiser_api import (
    LabanOptimisationResult,
    build_reference_motion,
    optimise_laban_target,
)
from laban_rl.perceptual_bandit.environment import (
    EnvironmentRewardConfig,
    MockNoisyPerceptualEvaluator,
)
from scripts.train_cem_contextual_bandit import _build_optimiser_overrides
from laban_rl.perceptual_bandit.evaluation_cache import PerceptualObservationCache
from laban_rl.perceptual_bandit.experiment import (
    REWARD_SCALE_NOTE,
    ExperimentCase,
    cases_from_matrix,
    run_paired_experiment,
)
from laban_rl.perceptual_bandit.paired_preference import (
    GeminiPairedPreferenceEvaluator,
    MockPairedPreferenceEvaluator,
    PairedPreferenceCache,
    PreferencePair,
    run_paired_preference_experiment,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable paired perceptual matrix. Each clip/repeat is "
            "collected once and rescored as VAD and categorical reward."
        )
    )
    parser.add_argument("--matrix", required=True, help="Matrix JSON configuration.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--evaluator", choices=("mock", "gemini"), default="mock")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--video-duration-seconds", type=float, default=2.0)
    parser.add_argument("--video-lead-in-seconds", type=float, default=0.0)
    parser.add_argument("--video-repetitions", type=int, default=1)
    parser.add_argument(
        "--video-inter-repeat-transition-seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument("--video-final-hold-seconds", type=float, default=0.0)
    parser.add_argument(
        "--video-presentation-style",
        choices=("plot", "arm_only"),
        default="plot",
    )
    parser.add_argument("--mock-noise-std", type=float, default=0.08)
    parser.add_argument("--maxiter", type=int, default=45)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--local-maxiter", type=int, default=100)
    parser.add_argument("--max-feature-error-threshold", type=float, default=0.10)
    parser.add_argument("--realisation-penalty-weight", type=float, default=0.25)
    parser.add_argument("--stability-penalty-weight", type=float, default=0.0)
    parser.add_argument(
        "--paired-ab",
        action="store_true",
        help="Require exactly two candidate profiles and print focused A/B output.",
    )
    parser.add_argument(
        "--paired-preference-repeats",
        type=int,
        default=0,
        help="Collect blinded A/B/neither judgments after independent ratings.",
    )
    parser.add_argument(
        "--paired-preference-cache",
        default=None,
        help="Cache root for blinded paired judgments.",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    if args.paired_preference_repeats < 0:
        parser.error("--paired-preference-repeats cannot be negative.")
    if args.paired_preference_repeats and not args.paired_ab:
        parser.error("--paired-preference-repeats requires --paired-ab.")
    return args


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _case_identity(case: ExperimentCase) -> str:
    canonical = json.dumps(
        case.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _save_motion_snapshot(
    case: ExperimentCase,
    result: LabanOptimisationResult,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = out_dir / "paired_motion_snapshot.npz"
    temporary_arrays = out_dir / "paired_motion_snapshot.tmp.npz"
    np.savez_compressed(
        temporary_arrays,
        q_ref=np.asarray(result.q_ref, dtype=float),
        q_var=np.asarray(result.q_var, dtype=float),
        action=np.asarray(result.action_coefficients, dtype=float),
    )
    temporary_arrays.replace(arrays_path)
    metadata = {
        "case_identity": _case_identity(case),
        "gesture": result.gesture,
        "target_state": result.target_state,
        "requested_profile": dict(result.requested_profile),
        "achieved_profile": dict(result.achieved_profile),
        "achieved_profile_clipped": dict(result.achieved_profile_clipped),
        "inner_reward": float(result.inner_reward),
        "inner_loss": float(result.inner_loss),
        "reward_info": dict(result.raw_result.get("reward_info", {})),
    }
    metadata_path = out_dir / "paired_motion_snapshot.json"
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)


def _load_motion_snapshot(
    case: ExperimentCase,
    out_dir: Path,
) -> LabanOptimisationResult | None:
    arrays_path = out_dir / "paired_motion_snapshot.npz"
    metadata_path = out_dir / "paired_motion_snapshot.json"
    if not arrays_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("case_identity") != _case_identity(case):
        raise ValueError(f"Incompatible motion snapshot: {out_dir}")
    with np.load(arrays_path) as arrays:
        q_ref = np.asarray(arrays["q_ref"], dtype=float)
        q_var = np.asarray(arrays["q_var"], dtype=float)
        action = np.asarray(arrays["action"], dtype=float)
    if (
        q_ref.ndim != 2
        or q_var.shape != q_ref.shape
        or q_ref.shape[1] != 2
        or not np.all(np.isfinite(q_ref))
        or not np.all(np.isfinite(q_var))
    ):
        raise ValueError(f"Invalid motion snapshot arrays: {out_dir}")
    return LabanOptimisationResult(
        gesture=str(metadata["gesture"]),
        target_state=str(metadata["target_state"]),
        requested_profile=dict(metadata["requested_profile"]),
        achieved_profile=dict(metadata["achieved_profile"]),
        achieved_profile_clipped=dict(
            metadata["achieved_profile_clipped"]
        ),
        inner_reward=float(metadata["inner_reward"]),
        inner_loss=float(metadata["inner_loss"]),
        action_coefficients=action,
        q_ref=q_ref,
        q_var=q_var,
        output_dir=out_dir,
        raw_result={"reward_info": dict(metadata["reward_info"])},
    )


def _print_summary(payload: dict, *, paired_ab: bool) -> None:
    comparison = payload["paired_comparison"]
    print(REWARD_SCALE_NOTE)
    for group in comparison["groups"]:
        print(f"\n{group['group']}")
        print(f"  VAD ranking:         {', '.join(group['vad_ranking'])}")
        if group["categorical_ranking"] is None:
            print("  Categorical ranking: unavailable (secondary output missing/not applicable)")
        else:
            print(
                "  Categorical ranking: "
                + ", ".join(group["categorical_ranking"])
            )
            print(
                "  Ranking agreement:   "
                f"{group['pairwise_ranking_agreement']:.3f}"
            )
        if paired_ab:
            print(f"  VAD selected:        {group['vad_selected']}")
            print(
                "  Category selected:   "
                f"{group['categorical_selected'] or 'unavailable'}"
            )
    for record in payload["records"]:
        reliability = record.get("reliability")
        if reliability is None:
            print(
                f"  {record['candidate_id']}: feasible=False, "
                "perceptual evaluation skipped"
            )
            continue
        print(
            f"  {record['candidate_id']}: feasible={record['feasibility']['feasible']}, "
            f"VAD std={reliability['vad_reward_std']:.4f}, "
            f"winner agreement={reliability['winner_agreement_rate']}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = json.loads(_resolve(args.matrix).read_text(encoding="utf-8"))
    parsed_cases = cases_from_matrix(matrix)
    cases = []
    for case in parsed_cases:
        optimiser_args = SimpleNamespace(
            maxiter=args.maxiter,
            popsize=args.popsize,
            local_maxiter=args.local_maxiter,
            de_mutation=0.5,
            de_recombination=0.65,
            seed=case.seed,
            target_state=case.context.target_state,
            wave_flow_target_weight=0.35,
        )
        effective_overrides = _build_optimiser_overrides(
            optimiser_args,
            case.context.gesture,
        )
        effective_overrides.update(case.optimizer_overrides)
        cases.append(
            replace(case, optimizer_overrides=effective_overrides)
        )
    if args.paired_ab:
        profile_names = {case.candidate_name for case in cases}
        if len(profile_names) != 2:
            raise ValueError(
                "--paired-ab requires exactly two named candidate profiles."
            )

    if args.evaluator == "mock":
        evaluator = MockNoisyPerceptualEvaluator(
            noise_std=args.mock_noise_std,
            seed=0,
        )
        preference_evaluator = MockPairedPreferenceEvaluator()
    else:
        from laban_rl.perceptual_bandit.gemini_evaluator import (
            GeminiProVideoEvaluator,
        )

        evaluator = GeminiProVideoEvaluator(
            model=args.model,
            temperature=args.temperature,
            video_duration_seconds=args.video_duration_seconds,
            video_lead_in_seconds=args.video_lead_in_seconds,
            video_repetitions=args.video_repetitions,
            video_inter_repeat_transition_seconds=(
                args.video_inter_repeat_transition_seconds
            ),
            video_final_hold_seconds=args.video_final_hold_seconds,
            video_presentation_style=args.video_presentation_style,
            keep_uploaded_files=False,
        )
        preference_evaluator = GeminiPairedPreferenceEvaluator(
            model=args.model,
            temperature=args.temperature,
            video_duration_seconds=args.video_duration_seconds,
            video_lead_in_seconds=args.video_lead_in_seconds,
            video_repetitions=args.video_repetitions,
            video_inter_repeat_transition_seconds=(
                args.video_inter_repeat_transition_seconds
            ),
            video_final_hold_seconds=args.video_final_hold_seconds,
            video_presentation_style=args.video_presentation_style,
            keep_uploaded_files=False,
        )

    reward_config = EnvironmentRewardConfig(
        repeat_evaluations=args.repeats,
        perceptual_reward_mode="vad",
        max_feature_error_threshold=args.max_feature_error_threshold,
        reject_excessive_feature_error=True,
        realisation_penalty_weight=args.realisation_penalty_weight,
        stability_penalty_weight=args.stability_penalty_weight,
    )

    materialized_results = {}

    def inner_runner(case: ExperimentCase, out_dir: Path):
        result = _load_motion_snapshot(case, out_dir)
        if result is not None:
            materialized_results[case.candidate_id] = result
            return result
        if case.motion_source == "reference":
            result = build_reference_motion(
                gesture=case.context.gesture,
                target_state=case.context.target_label,
                out_dir=out_dir,
            )
        else:
            result = optimise_laban_target(
                gesture=case.context.gesture,
                target_state=case.context.target_label,
                target_profile=case.profile,
                out_dir=out_dir,
                optimiser_overrides=case.optimizer_overrides,
            )
        _save_motion_snapshot(case, result, out_dir)
        materialized_results[case.candidate_id] = result
        return result

    try:
        payload = run_paired_experiment(
            cases,
            inner_runner=inner_runner,
            evaluator=evaluator,
            cache=PerceptualObservationCache(_resolve(args.cache)),
            repeats=args.repeats,
            reward_config=reward_config,
            out_dir=_resolve(args.out),
            runner_identity={
                "implementation": "paired-reference-styled-v2",
                "maxiter": args.maxiter,
                "popsize": args.popsize,
                "local_maxiter": args.local_maxiter,
            },
            save_plots=not args.no_plots,
        )
        if args.paired_preference_repeats:
            destination = _resolve(args.out)
            for case in cases:
                if case.candidate_id not in materialized_results:
                    inner_runner(
                        case,
                        destination / "candidates" / case.candidate_id,
                    )
            grouped_cases = {}
            for case in cases:
                pair_id = case.candidate_id.rsplit("__", 1)[0]
                grouped_cases.setdefault(pair_id, []).append(case)
            pairs = []
            for pair_id, pair_cases in sorted(grouped_cases.items()):
                by_name = {
                    case.candidate_name: case for case in pair_cases
                }
                if set(by_name) != {"reference", "styled"}:
                    raise ValueError(
                        "Blinded preference requires reference and styled "
                        f"candidates for {pair_id!r}."
                    )
                reference_case = by_name["reference"]
                styled_case = by_name["styled"]
                pairs.append(
                    PreferencePair(
                        pair_id=pair_id,
                        context=styled_case.context,
                        reference_result=materialized_results[
                            reference_case.candidate_id
                        ],
                        styled_result=materialized_results[
                            styled_case.candidate_id
                        ],
                        target_layers=styled_case.target_layers,
                    )
                )
            preference_cache_path = (
                _resolve(args.paired_preference_cache)
                if args.paired_preference_cache
                else _resolve(args.cache + "_paired_preferences")
            )
            preference_payload = run_paired_preference_experiment(
                pairs,
                evaluator=preference_evaluator,
                cache=PairedPreferenceCache(preference_cache_path),
                repeats=args.paired_preference_repeats,
                out_path=destination / "paired_preferences.json",
            )
            payload["blinded_paired_preferences"] = preference_payload
    finally:
        for active_evaluator in (evaluator, preference_evaluator):
            close = getattr(active_evaluator, "close", None)
            if close is not None:
                close()
    _print_summary(payload, paired_ab=args.paired_ab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
