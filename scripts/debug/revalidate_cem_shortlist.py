"""Re-evaluate an existing CEM shortlist without repeating CEM training."""
from __future__ import annotations

import argparse
import json
import sys
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
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    summary_path = experiment_dir / "results_summary.json"
    validation_path = experiment_dir / "independent_validation.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    previous_selection = validation.get("selection") or summary.get("independent_selection")
    if not previous_selection or not previous_selection.get("shortlist"):
        raise RuntimeError("No saved shortlist was found in this experiment.")

    model = args.model or summary.get("model") or "gemini-2.5-flash"
    temperature = args.temperature if args.temperature is not None else float(summary.get("temperature", 0.2))
    evaluator = GeminiProVideoEvaluator(model=model, temperature=temperature)
    optimiser_args = SimpleNamespace(
        maxiter=int(summary["inner_maxiter"]),
        popsize=int(summary["inner_popsize"]),
        local_maxiter=int(summary["inner_local_maxiter"]),
        seed=int(summary["seed"]),
        wave_flow_target_weight=float(summary.get("wave_flow_target_weight", 0.35)),
    )
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
        optimiser_overrides=_build_optimiser_overrides(optimiser_args, summary["gesture"]),
    )
    context = Context(gesture=summary["gesture"], target_state=summary["target_state"])
    recovered = []
    try:
        for item in previous_selection["shortlist"]:
            rank = int(item["rank"])
            key = f"revalidation_rank_{rank:02d}"
            print(f"Revalidating shortlist rank {rank} with {args.repeats} repeats...")
            result = environment.step(
                context=context,
                action_profile=item["profile"],
                out_dir=experiment_dir / "revalidation" / key / "optimiser_outputs",
            )
            validation[key] = result.to_dict()
            recovered.append({
                **item,
                "revalidation_reward": float(result.outer_reward),
                "revalidation_result_key": key,
                "revalidation_failure": result.failure_reason,
            })
            validation_path.write_text(json.dumps(validation, indent=2, allow_nan=False), encoding="utf-8")
    finally:
        evaluator.close()

    valid = [item for item in recovered if item["revalidation_failure"] is None]
    if not valid:
        raise RuntimeError("Every shortlist revalidation failed; no profile can be selected.")
    selected = max(valid, key=lambda item: item["revalidation_reward"])
    selection = {
        "method": "recovered_independent_top_k_reranking",
        "repeats": args.repeats,
        "selected": selected,
        "shortlist": recovered,
    }
    validation["revalidation_selection"] = selection
    validation["best_sampled_profile"] = dict(validation[selected["revalidation_result_key"]])
    validation_path.write_text(json.dumps(validation, indent=2, allow_nan=False), encoding="utf-8")
    summary["revalidation_selection"] = selection
    summary["best_sampled_profile_validation"] = validation["best_sampled_profile"]
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (experiment_dir / "selected_validated_profile_recovered.json").write_text(
        json.dumps(selected, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Selected rank {selected['rank']} with reward {selected['revalidation_reward']:.6f}")


if __name__ == "__main__":
    main()
