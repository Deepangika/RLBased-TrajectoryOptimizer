"""Build the frozen 6-gesture by 6-state Gemini evaluation matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laban_rl.config import EMOTION_STATES, GESTURE_TYPES
from laban_rl.perceptual_bandit.gemini_evaluator import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    SEQUENCE_RENDERER_VERSION,
)
from laban_rl.perceptual_bandit.paired_preference import (
    PAIR_PROMPT_VERSION,
    PAIR_SCHEMA_VERSION,
)
from laban_rl.perceptual_bandit.variant_video import FROZEN_RENDER_STYLE
from laban_rl.targets import TARGET_PROFILES


def build_matrix(
    pilot_config: dict,
    projection_config: dict,
) -> dict:
    references = {}
    for condition in pilot_config["cases"]:
        gesture = condition["gesture"]
        references.setdefault(
            gesture,
            condition["candidate_profiles"]["reference"]["profile"],
        )
    missing = sorted(set(GESTURE_TYPES) - set(references))
    if missing:
        raise ValueError(f"Pilot config lacks reference profiles: {missing}")

    cases = []
    for gesture in GESTURE_TYPES:
        for state in EMOTION_STATES:
            original = dict(TARGET_PROFILES[state])
            projection = (
                projection_config["states"].get(state)
                if gesture == "wave"
                else None
            )
            projected = (
                dict(projection["projected_feasible_target"])
                if projection is not None
                else None
            )
            condition = {
                "id": (
                    f"{gesture.replace('_', '-')}-{state}-"
                    f"{'projected' if projected is not None else 'original'}"
                ),
                "gesture": gesture,
                "target": {"state": state},
                "seed": 7,
                "target_layers": {
                    "original_affect_derived_laban_target": original,
                    "projected_feasible_laban_target": projected,
                },
                "candidate_profiles": {
                    "reference": {
                        "motion_source": "reference",
                        "profile": dict(references[gesture]),
                    },
                    "styled": {
                        "profile": projected or original,
                    },
                },
            }
            overrides = _optimizer_overrides(gesture, state)
            if overrides:
                condition["candidate_profiles"]["styled"][
                    "optimizer_overrides"
                ] = overrides
            cases.append(condition)
    return {
        "format_version": 1,
        "description": "Frozen full 6-gesture by 6-state Gemini experiment.",
        "evaluation_protocol": {
            "model": "gemini-2.5-flash",
            "temperature": 0.2,
            "independent_prompt_version": PROMPT_VERSION,
            "independent_schema_version": SCHEMA_VERSION,
            "paired_prompt_version": PAIR_PROMPT_VERSION,
            "paired_schema_version": PAIR_SCHEMA_VERSION,
            "renderer_version": SEQUENCE_RENDERER_VERSION,
            "render_style": FROZEN_RENDER_STYLE,
            "gesture_duration_seconds": 2.0,
            "lead_in_seconds": 0.5,
            "repetitions": 2,
            "inter_repeat_transition_seconds": 0.5,
            "final_hold_seconds": 0.5,
            "total_clip_duration_seconds": 5.5,
            "gesture_wide_camera_limits": True,
            "independent_target_blind": True,
            "independent_repeats": 5,
            "paired_preference_repeats": 5,
            "paired_neither_enabled": True,
        },
        "condition_count": 36,
        "logical_video_count": 72,
        "unique_video_count": 42,
        "planned_independent_calls": 210,
        "planned_paired_calls": 180,
        "planned_total_calls": 390,
        "cases": cases,
    }


def _optimizer_overrides(gesture: str, state: str) -> dict:
    overrides = {}
    if (gesture, state) in {
        ("point", "disgust"),
        ("point", "surprise"),
        ("circle", "surprise"),
        ("celebratory_pump", "disgust"),
    }:
        overrides = {"maxiter": 75, "popsize": 8}
    elif gesture == "wave" and state == "fear":
        overrides = {"maxiter": 75, "popsize": 8}
    elif gesture == "beckon" and state == "anger":
        overrides = {
            "maxiter": 75,
            "popsize": 8,
            "n_timing_basis": 5,
            "shape_arcness_target_weight": 0.2,
        }
    elif gesture == "beckon" and state == "disgust":
        overrides = {
            "maxiter": 75,
            "popsize": 8,
            "n_timing_basis": 5,
            "time_target_weight": 0.1,
            "flow_boundness_target_weight": 0.2,
            "shape_arcness_target_weight": 0.05,
        }
    elif gesture == "beckon" and state == "fear":
        overrides = {
            "maxiter": 75,
            "popsize": 8,
            "n_timing_basis": 5,
            "time_target_weight": 0.1,
            "flow_boundness_target_weight": 0.1,
            "shape_arcness_target_weight": 0.06,
        }
    elif gesture == "beckon" and state == "happiness":
        overrides = {
            "maxiter": 75,
            "popsize": 8,
            "n_timing_basis": 6,
            "time_target_weight": 0.2,
            "flow_boundness_target_weight": 0.2,
        }
    elif gesture == "beckon" and state == "surprise":
        overrides = {
            "maxiter": 75,
            "popsize": 8,
            "n_timing_basis": 6,
            "time_target_weight": 0.1,
            "flow_boundness_target_weight": 0.1,
        }
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-config",
        default="configs/limited_gemini_pilot.json",
    )
    parser.add_argument(
        "--projection-config",
        default="configs/wave_feasible_target_projections.json",
    )
    parser.add_argument(
        "--out",
        default="configs/full_gemini_matrix.json",
    )
    args = parser.parse_args()
    matrix = build_matrix(
        json.loads(Path(args.pilot_config).read_text(encoding="utf-8")),
        json.loads(
            Path(args.projection_config).read_text(encoding="utf-8")
        ),
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(matrix, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
