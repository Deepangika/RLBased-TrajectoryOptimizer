"""Run one complete REAL optimiser -> Gemini -> outer reward step.

Drop into:
    scripts/test_perceptual_environment_step_gemini.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from laban_rl.perceptual_bandit.environment import (
    Context,
    EnvironmentRewardConfig,
    PerceptualBanditEnvironment,
)
from laban_rl.perceptual_bandit.gemini_evaluator import (
    GeminiProVideoEvaluator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gesture",
        choices=["wave", "reach", "point"],
        required=True,
    )
    parser.add_argument(
        "--target-state",
        choices=["friendly", "confused", "angry"],
        required=True,
    )

    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--flow", type=float, required=True)
    parser.add_argument("--space", type=float, required=True)
    parser.add_argument("--shape", type=float, required=True)

    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--video-duration-seconds",
        type=float,
        default=None,
        help=(
            "Optional MP4 duration override."
            "Omit to preserve the original GIF timing during conversion."

        )
    )

    parser.add_argument(
        "--realisation-penalty-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--stability-penalty-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--invalid-realisation-reward",
        type=float,
        default=-1.0,
    )

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maxiter", type=int, default=90)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--local-maxiter", type=int, default=300)
    parser.add_argument("--out", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    context = Context(
        gesture=args.gesture,
        target_state=args.target_state,
    )
    action_profile = {
        "weight": args.weight,
        "time": args.time,
        "flow_boundness": args.flow,
        "space_indirectness": args.space,
        "shape_arcness": args.shape,
    }

    evaluator = GeminiProVideoEvaluator(
        model=args.model,
        temperature=args.temperature,
        video_duration_seconds=args.video_duration_seconds,
    )

    environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=args.repeats,
            realisation_penalty_weight=args.realisation_penalty_weight,
            stability_penalty_weight=args.stability_penalty_weight,
            invalid_realisation_reward=args.invalid_realisation_reward,
        ),
        optimiser_overrides={
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "local_maxiter": args.local_maxiter,
            "seed": args.seed,
        },
    )

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = environment.step(
            context=context,
            action_profile=action_profile,
            out_dir=out_dir / "optimiser_outputs",
        )

        payload = result.to_dict()
        payload["gemini_model"] = args.model
        payload["gemini_video_path"] = evaluator.last_video_path

        if evaluator.last_assessment is not None:
            payload["last_gemini_assessment"] = (
                evaluator.last_assessment.model_dump()
            )

        result_path = (
            out_dir / "environment_step_result_gemini.json"
        )
        result_path.write_text(
            json.dumps(payload, indent=2, allow_nan=True),
            encoding="utf-8",
        )

        print("\n" + "=" * 92)
        print("REAL GEMINI PERCEPTUAL ENVIRONMENT STEP")
        print("=" * 92)
        print(f"Gesture:             {result.context.gesture}")
        print(f"Target state:        {result.context.target_state}")
        print(f"Valid realisation:   {result.valid_realisation}")

        if not result.valid_realisation:
            print(f"Failure reason:      {result.failure_reason}")
            print(f"Outer reward:        {result.outer_reward:.6f}")
        else:
            print(
                f"Realisation RMSE:    "
                f"{result.realisation_rmse:.6f}"
            )
            print(
                f"Mean target prob:    "
                f"{result.mean_target_probability:.6f}"
            )
            print(
                f"Mean margin:         "
                f"{result.mean_margin:.6f}"
            )
            print(
                f"Mean perceptual R:   "
                f"{result.mean_perceptual_reward:.6f}"
            )
            print(
                f"Perceptual R std:    "
                f"{result.perceptual_reward_std:.6f}"
            )
            print(
                f"Outer reward:        "
                f"{result.outer_reward:.6f}"
            )

            print("\nRepeated Gemini evaluations:")
            for index, probabilities in enumerate(
                result.perceptual_evaluations,
                start=1,
            ):
                text = " | ".join(
                    f"{label}={probability:.4f}"
                    for label, probability in probabilities.items()
                )
                print(f"  Evaluation {index}: {text}")

        print(f"\nSaved: {result_path}")
        print("=" * 92)
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
