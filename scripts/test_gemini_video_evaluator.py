
"""Test Gemini on an existing optimiser output with same-style variant-only video.

Drop into:
    scripts/test_gemini_video_evaluator.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in [PROJECT_ROOT, SRC_DIR]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.perceptual_bandit.environment import Context
from laban_rl.perceptual_bandit.gemini_evaluator import (
    GeminiProVideoEvaluator,
)


class ExistingOptimiserResult:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

        variant_path = output_dir / "best_variant.npz"
        if not variant_path.exists():
            raise FileNotFoundError(
                f"Could not find {variant_path}."
            )

        with np.load(variant_path, allow_pickle=False) as data:
            self.q_ref = np.asarray(data["q_ref"], dtype=float)
            self.q_var = np.asarray(data["q_var"], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimiser-output", required=True)
    parser.add_argument(
        "--gesture",
        choices=["wave", "reach", "point"],
        required=True,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help=(
            "Optional FPS override. Omit to preserve the trajectory's "
            "2-second duration using all comparison-style frames."
        ),
    )
    parser.add_argument(
        "--video-duration-seconds",
        type=float,
        default=None,
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    optimiser_output = Path(args.optimiser_output)
    if not optimiser_output.is_absolute():
        optimiser_output = PROJECT_ROOT / optimiser_output

    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    context = Context(
        gesture=args.gesture,
        target_state="confident",
    )
    optimisation_result = ExistingOptimiserResult(
        optimiser_output
    )

    evaluator = GeminiProVideoEvaluator(
        model=args.model,
        temperature=args.temperature,
        video_duration_seconds=args.video_duration_seconds,
        video_fps=args.video_fps,
    )

    records = []

    try:
        for index in range(1, args.repeats + 1):
            evaluation = evaluator.evaluate(
                context,
                optimisation_result,
            )
            assessment = evaluator.last_assessment

            record = {
                "evaluation": index,
                "probabilities": evaluation.probabilities,
                "perceived_state": assessment.perceived_state,
                "confidence": assessment.confidence,
                "reasoning_summary": assessment.reasoning_summary,
            }
            records.append(record)

            print("\n" + "=" * 88)
            print(f"GEMINI EVALUATION {index}")
            print("=" * 88)

            for label, probability in evaluation.probabilities.items():
                print(f"{label:12s}: {probability:.4f}")

            print(f"\nPerceived state: {assessment.perceived_state}")
            print(f"Confidence:      {assessment.confidence:.4f}")
            print(f"Reasoning:       {assessment.reasoning_summary}")

        payload = {
            "gesture": args.gesture,
            "model": args.model,
            "video_path": evaluator.last_video_path,
            "evaluations": records,
        }

        output_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        print(f"\nSaved: {output_path}")
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
