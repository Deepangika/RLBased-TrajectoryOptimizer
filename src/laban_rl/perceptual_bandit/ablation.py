"""Offline reward and feasibility ablations over paired cached observations."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from laban_rl.perceptual_bandit.environment import EnvironmentRewardConfig
from laban_rl.perceptual_bandit.scoring import (
    apply_outer_penalties,
    score_perceptual_observations,
)


ABLATION_FORMAT_VERSION = 1
INNER_RERUN_KEYS = {
    "maxiter",
    "popsize",
    "local_maxiter",
    "de_mutation",
    "de_recombination",
    "optimizer",
}


def validate_stability_evidence(
    weight: float,
    reliability: Mapping[str, Any] | None,
) -> None:
    if weight == 0.0:
        return
    if not reliability or reliability.get("status") not in ("ok", "partial"):
        raise ValueError(
            "A non-zero stability penalty requires measured multi-clip "
            "test-retest reliability (status 'ok' or 'partial')."
        )
    repeat_counts = reliability.get("repeat_counts", [])
    if not repeat_counts or min(int(value) for value in repeat_counts) < 2:
        raise ValueError(
            "A non-zero stability penalty requires at least two repeats per clip."
        )


def _config(base: EnvironmentRewardConfig, settings: Mapping[str, Any]):
    unknown = set(settings) - set(asdict(base)) - INNER_RERUN_KEYS
    if unknown:
        raise ValueError(f"Unknown ablation settings: {sorted(unknown)}")
    overrides = {
        key: value for key, value in settings.items() if key in asdict(base)
    }
    resolved = replace(base, **overrides)
    resolved.validate()
    return resolved


def _feasible(
    feasibility: Mapping[str, Any],
    config: EnvironmentRewardConfig,
) -> tuple[bool, bool]:
    raw_available = all(
        feasibility.get(key) is not None
        for key in ("path_length_ratio", "joint_limit_error")
    )
    if raw_available:
        path_ratio = float(feasibility["path_length_ratio"])
        joint_error = float(feasibility["joint_limit_error"])
        physical = (
            config.minimum_path_length_ratio
            <= path_ratio
            <= config.maximum_path_length_ratio
            and joint_error <= config.joint_limit_tolerance
        )
    else:
        physical = bool(feasibility.get("physically_acceptable"))
    max_error = feasibility.get("max_abs_feature_error")
    feature_ok = max_error is not None and float(max_error) <= (
        config.max_feature_error_threshold
    )
    feasible = bool(
        feasibility.get("valid_realisation")
        and physical
        and (feature_ok or not config.reject_excessive_feature_error)
    )
    return feasible, raw_available


def rescore_paired_ablation(
    paired_payload: Mapping[str, Any],
    configurations: Sequence[Mapping[str, Any]],
    *,
    base_config: EnvironmentRewardConfig | None = None,
) -> dict[str, Any]:
    """Rescore cached observations and rank separately within each reward view."""
    base = base_config or EnvironmentRewardConfig()
    reliability = paired_payload.get("test_retest_reliability")
    outputs: list[dict[str, Any]] = []
    for index, item in enumerate(configurations):
        name = str(item.get("name") or f"configuration-{index + 1}")
        settings = dict(item.get("settings", {}))
        config = _config(base, settings)
        validate_stability_evidence(config.stability_penalty_weight, reliability)
        requires_inner_rerun = bool(INNER_RERUN_KEYS.intersection(settings))
        records: list[dict[str, Any]] = []
        for source in paired_payload.get("records", []):
            feasibility = dict(source.get("feasibility") or {})
            feasible, raw_gates_available = _feasible(feasibility, config)
            observations = list(source.get("observations") or [])
            requires_rerun = requires_inner_rerun
            reason = None
            if not raw_gates_available and any(
                key in settings
                for key in (
                    "minimum_path_length_ratio",
                    "maximum_path_length_ratio",
                    "joint_limit_tolerance",
                )
            ):
                requires_rerun = True
                reason = "raw feasibility gate measurements are unavailable"
            if feasible and not observations:
                requires_rerun = True
                reason = (
                    reason
                    or "candidate was previously gated before perceptual collection"
                )
            vad_score = None
            categorical_score = None
            scored = None
            if observations:
                context = source["context"]
                scored = score_perceptual_observations(
                    observations,
                    target_vad=context["target_vad"],
                    target_state=context.get("target_state"),
                    reward_config=config,
                )
                rmse = feasibility.get("realisation_rmse")
                max_error = feasibility.get("max_abs_feature_error")
                if rmse is not None and max_error is not None:
                    vad_score = apply_outer_penalties(
                        scored["mean_vad_reward"],
                        perceptual_reward_std=scored["vad_reward_std"],
                        realisation_rmse=float(rmse),
                        max_abs_feature_error=float(max_error),
                        reward_config=config,
                    )
                    if scored["categorical_reward"] is not None:
                        categorical_score = apply_outer_penalties(
                            scored["categorical_reward"],
                            perceptual_reward_std=scored[
                                "categorical_reward_std"
                            ],
                            realisation_rmse=float(rmse),
                            max_abs_feature_error=float(max_error),
                            reward_config=config,
                        )
            records.append(
                {
                    "candidate_id": source["candidate_id"],
                    "context": source["context"],
                    "seed": source["seed"],
                    "feasible": feasible,
                    "vad_score": vad_score,
                    "categorical_score": categorical_score,
                    "requires_inner_optimizer_rerun": requires_rerun,
                    "rerun_reason": reason,
                    "categorical_complete": bool(
                        scored and scored["categorical_complete"]
                    ),
                }
            )
        for metric in ("vad_score", "categorical_score"):
            rankable = [
                row
                for row in records
                if row[metric] is not None
                and not row["requires_inner_optimizer_rerun"]
            ]
            rankable.sort(
                key=lambda row: (row["feasible"], row[metric]), reverse=True
            )
            for rank, row in enumerate(rankable, start=1):
                row[f"{metric}_rank"] = rank
        outputs.append(
            {
                "name": name,
                "settings": asdict(config),
                "records": records,
                "requires_inner_optimizer_rerun": any(
                    row["requires_inner_optimizer_rerun"] for row in records
                ),
            }
        )
    return {
        "format_version": ABLATION_FORMAT_VERSION,
        "reward_scale_note": (
            "VAD and categorical magnitudes are not equivalent; rankings are "
            "computed independently and must not be compared across views."
        ),
        "configurations": outputs,
    }


def write_ablation(
    paired_path: str | Path,
    configuration_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    paired = json.loads(Path(paired_path).read_text(encoding="utf-8"))
    configurations = json.loads(
        Path(configuration_path).read_text(encoding="utf-8")
    )
    if not isinstance(configurations, list):
        raise ValueError("Ablation configuration must be a JSON array.")
    payload = rescore_paired_ablation(paired, configurations)
    Path(output_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload
