"""Condition-level reporting for matched independent and blinded evaluations."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from laban_rl.affect import VAD_KEYS
from laban_rl.config import FEATURE_KEYS


def _profile_rmse(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    return float(
        np.sqrt(
            np.mean(
                [
                    (float(left[key]) - float(right[key])) ** 2
                    for key in FEATURE_KEYS
                ]
            )
        )
    )


def build_full_experiment_report(
    independent: Mapping[str, Any],
    preferences: Mapping[str, Any],
) -> dict[str, Any]:
    preference_by_pair = {
        record["pair_id"]: record
        for record in preferences.get("records", [])
    }
    grouped = {}
    for record in independent.get("records", []):
        pair_id = record["candidate_id"].rsplit("__", 1)[0]
        grouped.setdefault(pair_id, {})[record["candidate_name"]] = record
    conditions = []
    for pair_id, candidates in sorted(grouped.items()):
        if set(candidates) != {"reference", "styled"}:
            raise ValueError(f"Incomplete matched condition: {pair_id}")
        if pair_id not in preference_by_pair:
            raise ValueError(f"Missing paired preference condition: {pair_id}")
        reference = candidates["reference"]
        styled = candidates["styled"]
        preference = preference_by_pair[pair_id]
        reference_reliability = reference["reliability"]
        styled_reliability = styled["reliability"]
        layers = styled.get("target_layers", {})
        original = layers.get("original_affect_derived_laban_target")
        projected = layers.get("projected_feasible_laban_target")
        requested = styled["requested_profile"]
        achieved = styled["achieved_profile"]
        if not isinstance(original, Mapping):
            raise ValueError(f"Missing original Laban target: {pair_id}")
        conditions.append(
            {
                "condition_id": pair_id,
                "gesture": styled["context"]["gesture"],
                "target_state": styled["context"].get("target_state"),
                "target_vad": dict(styled["context"]["target_vad"]),
                "profiles": {
                    "original_affect_derived": dict(original),
                    "projected_feasible": (
                        dict(projected)
                        if isinstance(projected, Mapping)
                        else None
                    ),
                    "requested_optimization": dict(requested),
                    "achieved": dict(achieved),
                },
                "vad": {
                    "reference_mean": dict(
                        reference_reliability["mean_observed_vad"]
                    ),
                    "styled_mean": dict(
                        styled_reliability["mean_observed_vad"]
                    ),
                    "per_axis_change_styled_minus_reference": {
                        key: float(
                            styled_reliability["mean_observed_vad"][key]
                            - reference_reliability["mean_observed_vad"][key]
                        )
                        for key in VAD_KEYS
                    },
                    "reference_mean_reward": float(
                        reference_reliability["mean_vad_reward"]
                    ),
                    "styled_mean_reward": float(
                        styled_reliability["mean_vad_reward"]
                    ),
                    "delta_vad_reward": float(
                        styled_reliability["mean_vad_reward"]
                        - reference_reliability["mean_vad_reward"]
                    ),
                    "delta_penalized_vad_score": float(
                        styled["vad_score"] - reference["vad_score"]
                    ),
                },
                "paired_preference": {
                    "styled_preference_rate": float(
                        preference["styled_preference_rate"]
                    ),
                    "reference_preference_rate": float(
                        preference["reference_preference_rate"]
                    ),
                    "neither_rate": float(preference["neither_rate"]),
                    "choice_counts": dict(preference["choice_counts"]),
                    "mean_confidence": float(
                        preference["mean_confidence"]
                    ),
                },
                "classification": {
                    "styled_target_classification_rate": (
                        styled_reliability.get(
                            "target_classification_rate"
                        )
                    ),
                    "styled_categorical_coverage": (
                        styled_reliability.get("categorical_coverage")
                    ),
                    "styled_ambiguous_count": (
                        styled_reliability.get(
                            "ambiguous_category_count"
                        )
                    ),
                    "styled_missing_count": (
                        styled_reliability.get("missing_category_count")
                    ),
                },
                "repeat_variability": {
                    "reference_vad_reward_std": float(
                        reference_reliability["vad_reward_std"]
                    ),
                    "styled_vad_reward_std": float(
                        styled_reliability["vad_reward_std"]
                    ),
                    "reference_per_axis_vad_std": dict(
                        reference_reliability["per_axis_vad_std"]
                    ),
                    "styled_per_axis_vad_std": dict(
                        styled_reliability["per_axis_vad_std"]
                    ),
                },
                "realisation": {
                    "projection_applied": projected is not None,
                    "projection_error_e_proj": (
                        _profile_rmse(projected, original)
                        if isinstance(projected, Mapping)
                        else None
                    ),
                    "realisation_error_e_real": _profile_rmse(
                        achieved,
                        requested,
                    ),
                    "original_to_achieved_error": _profile_rmse(
                        achieved,
                        original,
                    ),
                    **dict(styled["feasibility"]),
                },
            }
        )
    return {
        "format_version": 1,
        "condition_count": len(conditions),
        "independent_repeat_count": independent["run_identity"][
            "configuration"
        ]["repeats"],
        "paired_preference_repeat_count": preferences["repeat_count"],
        "conditions": conditions,
    }


def write_full_experiment_report(
    independent: Mapping[str, Any],
    preferences: Mapping[str, Any],
    out_dir: str | Path,
) -> dict[str, Any]:
    payload = build_full_experiment_report(independent, preferences)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "full_experiment_report.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(json_path)
    rows = []
    for condition in payload["conditions"]:
        row = {
            "condition_id": condition["condition_id"],
            "gesture": condition["gesture"],
            "target_state": condition["target_state"],
            "delta_vad_reward": condition["vad"]["delta_vad_reward"],
            "delta_penalized_vad_score": condition["vad"][
                "delta_penalized_vad_score"
            ],
            "styled_preference_rate": condition["paired_preference"][
                "styled_preference_rate"
            ],
            "neither_rate": condition["paired_preference"]["neither_rate"],
            "target_classification_rate": condition["classification"][
                "styled_target_classification_rate"
            ],
            "styled_vad_reward_std": condition["repeat_variability"][
                "styled_vad_reward_std"
            ],
            "projection_error_e_proj": condition["realisation"][
                "projection_error_e_proj"
            ],
            "realisation_error_e_real": condition["realisation"][
                "realisation_error_e_real"
            ],
            "path_length_ratio": condition["realisation"][
                "path_length_ratio"
            ],
            "nearest_path_mse": condition["realisation"].get(
                "nearest_path_mse"
            ),
            "max_abs_feature_error": condition["realisation"][
                "max_abs_feature_error"
            ],
        }
        for key, value in condition["vad"][
            "per_axis_change_styled_minus_reference"
        ].items():
            row[f"delta_{key}"] = value
        rows.append(row)
    csv_path = destination / "full_experiment_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload
