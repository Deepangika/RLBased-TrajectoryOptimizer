"""Collect once and compare VAD/categorical candidate rankings from one cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.optimiser_api import optimise_laban_target
from laban_rl.perceptual_bandit.environment import (
    EnvironmentRewardConfig,
    MockNoisyPerceptualEvaluator,
)
from laban_rl.perceptual_bandit.evaluation_cache import PerceptualObservationCache
from laban_rl.perceptual_bandit.experiment import (
    REWARD_SCALE_NOTE,
    ExperimentCase,
    cases_from_matrix,
    run_paired_experiment,
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
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    return args


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


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
    cases = cases_from_matrix(matrix)
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
    else:
        from laban_rl.perceptual_bandit.gemini_evaluator import (
            GeminiProVideoEvaluator,
        )

        evaluator = GeminiProVideoEvaluator(
            model=args.model,
            temperature=args.temperature,
        )

    reward_config = EnvironmentRewardConfig(
        repeat_evaluations=args.repeats,
        perceptual_reward_mode="vad",
        max_feature_error_threshold=args.max_feature_error_threshold,
        realisation_penalty_weight=args.realisation_penalty_weight,
        stability_penalty_weight=args.stability_penalty_weight,
    )

    def inner_runner(case: ExperimentCase, out_dir: Path):
        return optimise_laban_target(
            gesture=case.context.gesture,
            target_state=case.context.target_label,
            target_profile=case.profile,
            out_dir=out_dir,
            optimiser_overrides={
                "seed": case.seed,
                "maxiter": args.maxiter,
                "popsize": args.popsize,
                "local_maxiter": args.local_maxiter,
            },
        )

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
                "implementation": "optimise_laban_target-v1",
                "maxiter": args.maxiter,
                "popsize": args.popsize,
                "local_maxiter": args.local_maxiter,
            },
            save_plots=not args.no_plots,
        )
    finally:
        close = getattr(evaluator, "close", None)
        if close is not None:
            close()
    _print_summary(payload, paired_ab=args.paired_ab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
