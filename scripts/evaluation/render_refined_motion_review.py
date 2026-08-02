"""Render the six motions selected by focused feasibility refinement."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import optimise_laban_target
from laban_rl.targets import TARGET_PROFILES
from scripts.evaluation.run_focused_inner_screen import write_json_atomic
from scripts.train_cem_contextual_bandit import (
    _build_optimiser_overrides,
    load_projected_initial_profile,
)

CONDITIONS = (
    ("wave", "anger"),
    ("wave", "disgust"),
    ("wave", "sadness"),
    ("wave", "fear"),
    ("beckon", "fear"),
    ("beckon", "surprise"),
)


def optimizer_overrides(gesture: str, state: str, seed: int) -> dict:
    args = SimpleNamespace(
        maxiter=45,
        popsize=5,
        local_maxiter=100,
        de_mutation=0.5,
        de_recombination=0.65,
        seed=seed,
        target_state=state,
        wave_flow_target_weight=0.75,
    )
    return _build_optimiser_overrides(args, gesture)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/refined_motion_review")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    summaries = []
    for gesture, state in CONDITIONS:
        projected, projection_metadata = load_projected_initial_profile(
            gesture, state
        )
        profile = projected or TARGET_PROFILES[state]
        source = (
            "projected_feasible_target"
            if projected is not None
            else "affect_derived_target"
        )
        condition_dir = out_dir / gesture / state
        print(f"Rendering {gesture} {state} ({source})", flush=True)
        result = optimise_laban_target(
            gesture=gesture,
            target_state=state,
            target_profile=profile,
            out_dir=condition_dir,
            optimiser_overrides=optimizer_overrides(
                gesture, state, args.seed
            ),
        )
        errors = {
            key: abs(
                result.achieved_profile[key] - result.requested_profile[key]
            )
            for key in FEATURE_KEYS
        }
        reward_info = result.raw_result["reward_info"]
        summaries.append(
            {
                "gesture": gesture,
                "state": state,
                "target_source": source,
                "original_affect_target": TARGET_PROFILES[state],
                "rendered_target": profile,
                "projection_metadata": projection_metadata,
                "achieved_profile": result.achieved_profile,
                "per_feature_abs_error": errors,
                "realisation_rmse": result.realisation_rmse,
                "max_abs_feature_error": max(errors.values()),
                "path_length_ratio": reward_info["path_length_ratio"],
                "nearest_path_mse": reward_info["nearest_path_mse"],
                "nearest_path_max_dist": reward_info[
                    "nearest_path_max_dist"
                ],
                "endpoint_error": reward_info["endpoint_error"],
                "direction_error": reward_info["direction_error"],
                "smoothness_error": reward_info["smoothness_error"],
                "joint_limit_error": reward_info["joint_limit_error"],
                "optimizer_overrides": optimizer_overrides(
                    gesture, state, args.seed
                ),
                "animation": str(condition_dir / "arm_comparison.gif"),
                "feature_plot": str(
                    condition_dir / "feature_comparison.png"
                ),
            }
        )
        write_json_atomic(out_dir / "render_summary.json", summaries)
    print(out_dir / "render_summary.json", flush=True)


if __name__ == "__main__":
    main()
