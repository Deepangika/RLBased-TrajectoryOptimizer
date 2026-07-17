"""Run one complete outer environment step with a manually supplied action.

Drop this file into:
    scripts/test_perceptual_environment_step.py

Examples
--------
Moderate profile expected to be easy to realise:

    python scripts/test_perceptual_environment_step.py ^
        --gesture wave ^
        --target-state confident ^
        --weight 0.6 ^
        --time 0.6 ^
        --flow 0.3 ^
        --space 0.6 ^
        --shape 0.5 ^
        --maxiter 90 ^
        --popsize 8 ^
        --local-maxiter 300 ^
        --out outputs/environment_step_moderate

More difficult confident-like profile:

    python scripts/test_perceptual_environment_step.py ^
        --gesture wave ^
        --target-state confident ^
        --weight 0.8 ^
        --time 0.75 ^
        --flow 0.65 ^
        --space 0.2 ^
        --shape 0.75 ^
        --maxiter 90 ^
        --popsize 8 ^
        --local-maxiter 300 ^
        --out outputs/environment_step_confident
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
    MockNoisyPerceptualEvaluator,
    PerceptualBanditEnvironment,
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
        choices=[
            "confident",
            "calm",
            "hesitant",
            "friendly",
        ],
        required=True,
    )

    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--flow", type=float, required=True)
    parser.add_argument("--space", type=float, required=True)
    parser.add_argument("--shape", type=float, required=True)

    parser.add_argument("--repeats", type=int, default=3)
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

    parser.add_argument(
        "--mock-vlm-noise-std",
        type=float,
        default=0.08,
    )
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--maxiter", type=int, default=90)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument(
        "--local-maxiter",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    action_profile = {
        "weight": args.weight,
        "time": args.time,
        "flow_boundness": args.flow,
        "space_indirectness": args.space,
        "shape_arcness": args.shape,
    }

    context = Context(
        gesture=args.gesture,
        target_state=args.target_state,
    )

    evaluator = MockNoisyPerceptualEvaluator(
        noise_std=args.mock_vlm_noise_std,
        seed=args.seed,
    )

    reward_config = EnvironmentRewardConfig(
        repeat_evaluations=args.repeats,
        realisation_penalty_weight=(
            args.realisation_penalty_weight
        ),
        stability_penalty_weight=(
            args.stability_penalty_weight
        ),
        invalid_realisation_reward=(
            args.invalid_realisation_reward
        ),
    )

    environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=reward_config,
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

    result = environment.step(
        context=context,
        action_profile=action_profile,
        out_dir=out_dir / "optimiser_outputs",
    )

    result_path = out_dir / "environment_step_result.json"
    result_path.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("PERCEPTUAL ENVIRONMENT STEP RESULT")
    print("=" * 88)
    print(f"Gesture:             {result.context.gesture}")
    print(
        f"Target state:        "
        f"{result.context.target_state}"
    )
    print(
        f"Valid realisation:   "
        f"{result.valid_realisation}"
    )

    if not result.valid_realisation:
        print(
            f"Failure reason:      "
            f"{result.failure_reason}"
        )
        print(
            f"Invalid features:    "
            f"{result.invalid_features}"
        )
        print(
            f"Outer reward:        "
            f"{result.outer_reward:.6f}"
        )
        print(f"Saved:               {result_path}")
        print("=" * 88)
        return

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

    print("\nRequested -> achieved:")
    for key in action_profile:
        print(
            f"  {key:22s}: "
            f"{result.requested_profile[key]:.4f} "
            f"-> {result.achieved_profile[key]:.4f}"
        )

    print("\nRepeated mock perceptual evaluations:")
    for index, probabilities in enumerate(
        result.perceptual_evaluations,
        start=1,
    ):
        probability_text = " | ".join(
            f"{label}={probability:.4f}"
            for label, probability in probabilities.items()
        )
        print(f"  Evaluation {index}: {probability_text}")

    print(f"\nSaved: {result_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
