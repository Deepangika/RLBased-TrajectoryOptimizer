"""Evaluate a frozen CEM-selected profile on a fresh Gemini holdout set.

This script never re-ranks or changes the selected profile. It is intended for
post-selection performance reporting without selection optimism.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.perceptual_bandit.environment import (  # noqa: E402
    Context,
    EnvironmentRewardConfig,
    PerceptualBanditEnvironment,
)
from laban_rl.perceptual_bandit.gemini_evaluator import GeminiProVideoEvaluator  # noqa: E402
from scripts.train_cem_contextual_bandit import _build_optimiser_overrides  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing holdout result. Default: refuse.",
    )
    return parser.parse_args()


def selected_profile(summary: dict) -> tuple[dict[str, float], dict]:
    recovery = summary.get("revalidation_selection")
    if recovery and recovery.get("selected"):
        selected = recovery["selected"]
        return dict(selected["profile"]), {
            "selection_source": "revalidation_selection",
            "selected_training_rank": selected.get("rank"),
            "selected_training_source": selected.get("source"),
        }

    selection = summary.get("independent_selection")
    if selection and selection.get("selected"):
        selected = selection["selected"]
        return dict(selected["profile"]), {
            "selection_source": "independent_selection",
            "selected_training_rank": selected.get("rank"),
            "selected_training_source": selected.get("source"),
        }

    raise RuntimeError("No independently selected fixed profile was found.")


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    experiment_dir = Path(args.experiment_dir).resolve()
    summary_path = experiment_dir / "results_summary.json"
    result_path = experiment_dir / "fixed_profile_holdout.json"
    if result_path.exists() and not args.overwrite:
        raise RuntimeError(
            f"{result_path} already exists. Holdout data should not be silently "
            "reused or replaced; pass --overwrite only if intentionally rerunning it."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    profile, selection_metadata = selected_profile(summary)
    model = args.model or summary.get("model") or "gemini-2.5-flash"
    temperature = (
        float(args.temperature)
        if args.temperature is not None
        else float(summary.get("temperature", 0.2))
    )

    optimiser_args = SimpleNamespace(
        maxiter=int(summary["inner_maxiter"]),
        popsize=int(summary["inner_popsize"]),
        local_maxiter=int(summary["inner_local_maxiter"]),
        seed=int(summary["seed"]),
        wave_flow_target_weight=float(summary.get("wave_flow_target_weight", 0.35)),
    )
    evaluator = GeminiProVideoEvaluator(model=model, temperature=temperature)
    environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=args.repeats,
            reward_margin_mode=summary.get("reward_margin_mode", "raw"),
            realisation_penalty_weight=float(summary.get("realisation_penalty_weight", 0.25)),
            stability_penalty_weight=float(summary.get("stability_penalty_weight", 0.25)),
            max_feature_error_threshold=float(summary.get("max_feature_error_threshold", 0.10)),
            max_feature_error_penalty_weight=float(summary.get("max_feature_error_penalty_weight", 0.50)),
            reject_excessive_feature_error=False,
        ),
        optimiser_overrides=_build_optimiser_overrides(
            optimiser_args, summary["gesture"]
        ),
    )
    context = Context(
        gesture=summary["gesture"], target_state=summary["target_state"]
    )

    try:
        result = environment.step(
            context=context,
            action_profile=profile,
            out_dir=experiment_dir / "holdout" / "optimiser_outputs",
        )
    finally:
        evaluator.close()

    payload = {
        "evaluation_type": "fixed_profile_post_selection_holdout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gesture": summary["gesture"],
        "target_state": summary["target_state"],
        "seed": int(summary["seed"]),
        "model": model,
        "temperature": temperature,
        "planned_repeats": args.repeats,
        "completed_repeats": len(result.perceptual_evaluations),
        "profile_was_frozen": True,
        **selection_metadata,
        "fixed_requested_profile": profile,
        "holdout_result": result.to_dict(),
    }
    result_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Saved holdout result to {result_path}")
    print(f"Completed repeats: {payload['completed_repeats']}/{args.repeats}")
    print(f"Target classification rate: {result.target_classification_rate}")
    print(f"Mean target probability: {result.mean_target_probability}")
    print(f"Mean margin: {result.mean_margin}")
    print(f"Outer reward: {result.outer_reward}")


if __name__ == "__main__":
    main()
