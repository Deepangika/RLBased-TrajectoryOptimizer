"""Resumable paired perceptual experiment orchestration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import LabanOptimisationResult
from laban_rl.perceptual_bandit.environment import (
    Context,
    EnvironmentRewardConfig,
)
from laban_rl.perceptual_bandit.evaluation_cache import (
    EvaluatorCacheIdentity,
    PerceptualObservationCache,
)
from laban_rl.perceptual_bandit.scoring import (
    apply_outer_penalties,
    score_perceptual_observations,
    test_retest_reliability,
)


EXPERIMENT_FORMAT_VERSION = 4
REWARD_SCALE_NOTE = (
    "VAD and categorical reward scales are not directly comparable; compare "
    "candidate rankings, repeat stability, feasibility, and selected identity."
)


@dataclass(frozen=True)
class ExperimentCase:
    candidate_id: str
    candidate_name: str
    context: Context
    profile: Mapping[str, float]
    seed: int
    motion_source: str = "optimized"
    require_feature_match: bool = True
    optimizer_overrides: Mapping[str, Any] = field(default_factory=dict)
    target_layers: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "context": self.context.to_dict(),
            "profile": {key: float(self.profile[key]) for key in FEATURE_KEYS},
            "seed": int(self.seed),
            "motion_source": self.motion_source,
            "require_feature_match": bool(self.require_feature_match),
            "optimizer_overrides": dict(self.optimizer_overrides),
            "target_layers": dict(self.target_layers),
        }


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()
    return cleaned or "candidate"


def cases_from_matrix(config: Mapping[str, Any]) -> list[ExperimentCase]:
    """Expand gestures x targets x seeds x named profiles deterministically."""
    if "cases" in config:
        return _cases_from_explicit_config(config)

    gestures = [str(value) for value in config.get("gestures", [])]
    targets = list(config.get("targets", []))
    seeds = [int(value) for value in config.get("seeds", [])]
    profiles = dict(config.get("candidate_profiles", {}))
    if not gestures or not targets or not seeds or not profiles:
        raise ValueError(
            "Matrix requires gestures, targets, seeds, and candidate_profiles."
        )

    cases: list[ExperimentCase] = []
    for gesture in gestures:
        for target in targets:
            if not isinstance(target, Mapping):
                raise ValueError("Each matrix target must be an object.")
            if "state" in target:
                context = Context(gesture, target_state=str(target["state"]))
            elif "vad" in target:
                context = Context(gesture, target_vad=dict(target["vad"]))
            else:
                raise ValueError("Each target requires either 'state' or 'vad'.")
            target_name = str(target.get("label") or context.target_label)
            for seed in seeds:
                for candidate_name, profile_payload in profiles.items():
                    profile = {
                        key: float(dict(profile_payload)[key]) for key in FEATURE_KEYS
                    }
                    candidate_id = "__".join(
                        (
                            _slug(gesture),
                            _slug(target_name),
                            f"seed-{seed}",
                            _slug(str(candidate_name)),
                        )
                    )
                    cases.append(
                        ExperimentCase(
                            candidate_id=candidate_id,
                            candidate_name=str(candidate_name),
                            context=context,
                            profile=profile,
                            seed=seed,
                        )
                    )
    if len({case.candidate_id for case in cases}) != len(cases):
        raise ValueError("Matrix expands to duplicate candidate identifiers.")
    return cases


def _profile(payload: Mapping[str, Any], *, label: str) -> dict[str, float]:
    if set(payload) != set(FEATURE_KEYS):
        raise ValueError(f"{label} must contain exactly the five Laban features.")
    profile = {key: float(payload[key]) for key in FEATURE_KEYS}
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in profile.values()):
        raise ValueError(f"{label} values must be finite and lie in [0, 1].")
    return profile


def _cases_from_explicit_config(
    config: Mapping[str, Any],
) -> list[ExperimentCase]:
    conditions = list(config.get("cases", []))
    if not conditions:
        raise ValueError("Explicit matrix requires a non-empty cases array.")
    cases: list[ExperimentCase] = []
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise ValueError("Each explicit case must be an object.")
        condition_id = str(condition.get("id", "")).strip()
        gesture = str(condition.get("gesture", "")).strip()
        target = condition.get("target")
        candidates = condition.get("candidate_profiles")
        if not condition_id or not gesture or not isinstance(target, Mapping):
            raise ValueError("Each explicit case requires id, gesture, and target.")
        if not isinstance(candidates, Mapping) or not candidates:
            raise ValueError(
                f"Explicit case {condition_id!r} requires candidate_profiles."
            )
        if "state" in target:
            context = Context(gesture, target_state=str(target["state"]))
        elif "vad" in target:
            context = Context(gesture, target_vad=dict(target["vad"]))
        else:
            raise ValueError("Each target requires either 'state' or 'vad'.")
        seed = int(condition.get("seed", 7))
        target_layers = dict(condition.get("target_layers", {}))
        common_overrides = dict(condition.get("optimizer_overrides", {}))
        for candidate_name, raw_candidate in candidates.items():
            if not isinstance(raw_candidate, Mapping):
                raise ValueError(
                    f"Candidate {candidate_name!r} in {condition_id!r} "
                    "must be an object."
                )
            candidate = dict(raw_candidate)
            profile_payload = candidate.get("profile")
            if not isinstance(profile_payload, Mapping):
                raise ValueError(
                    f"Candidate {candidate_name!r} requires a profile."
                )
            motion_source = str(candidate.get("motion_source", "optimized"))
            if motion_source not in {"optimized", "reference"}:
                raise ValueError(
                    f"Unsupported motion_source {motion_source!r}."
                )
            overrides = {
                **common_overrides,
                **dict(candidate.get("optimizer_overrides", {})),
            }
            cases.append(
                ExperimentCase(
                    candidate_id="__".join(
                        (
                            _slug(condition_id),
                            f"seed-{seed}",
                            _slug(str(candidate_name)),
                        )
                    ),
                    candidate_name=str(candidate_name),
                    context=context,
                    profile=_profile(
                        profile_payload,
                        label=f"{condition_id}.{candidate_name}.profile",
                    ),
                    seed=seed,
                    motion_source=motion_source,
                    require_feature_match=bool(
                        candidate.get(
                            "require_feature_match",
                            motion_source != "reference",
                        )
                    ),
                    optimizer_overrides=overrides,
                    target_layers=target_layers,
                )
            )
    if len({case.candidate_id for case in cases}) != len(cases):
        raise ValueError("Explicit matrix contains duplicate candidate identifiers.")
    return cases


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_identity(
    *,
    cases: Sequence[ExperimentCase],
    repeats: int,
    evaluator_identity: EvaluatorCacheIdentity,
    reward_config: EnvironmentRewardConfig,
    runner_identity: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "format_version": EXPERIMENT_FORMAT_VERSION,
        "cases": [case.to_dict() for case in cases],
        "repeats": int(repeats),
        "evaluator": evaluator_identity.to_dict(),
        "reward_config": asdict(reward_config),
        "runner": dict(runner_identity),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "configuration": payload,
    }


def _realisation_metrics(
    result: LabanOptimisationResult,
    reward_config: EnvironmentRewardConfig,
    *,
    require_feature_match: bool = True,
) -> dict[str, Any]:
    requested = np.asarray(
        [result.requested_profile[key] for key in FEATURE_KEYS], dtype=float
    )
    achieved = np.asarray(
        [result.achieved_profile[key] for key in FEATURE_KEYS], dtype=float
    )
    finite = bool(np.all(np.isfinite(achieved)))
    errors = np.abs(requested - achieved) if finite else np.full_like(requested, np.inf)
    rmse = float(np.sqrt(np.mean((requested - achieved) ** 2))) if finite else None
    max_error = float(np.max(errors)) if finite else None
    reward_info = result.raw_result.get("reward_info", {})
    path_ratio = float(reward_info.get("path_length_ratio", np.nan))
    joint_error = float(reward_info.get("joint_limit_error", np.inf))
    path_preserved = bool(
        np.isfinite(path_ratio)
        and reward_config.minimum_path_length_ratio
        <= path_ratio
        <= reward_config.maximum_path_length_ratio
    )
    joints_ok = bool(
        np.isfinite(joint_error)
        and joint_error <= reward_config.joint_limit_tolerance
    )
    feature_ok = bool(
        max_error is not None
        and max_error <= reward_config.max_feature_error_threshold
    )
    return {
        "valid_realisation": finite,
        "path_length_ratio": (
            path_ratio if np.isfinite(path_ratio) else None
        ),
        "joint_limit_error": (
            joint_error if np.isfinite(joint_error) else None
        ),
        "path_preserved": path_preserved,
        "joint_limits_satisfied": joints_ok,
        "physically_acceptable": path_preserved and joints_ok,
        "feature_realisation_acceptable": feature_ok,
        "feature_match_required": require_feature_match,
        "feasible": (
            finite
            and path_preserved
            and joints_ok
            and (feature_ok or not require_feature_match)
        ),
        "realisation_rmse": rmse,
        "max_abs_feature_error": max_error,
        "per_feature_abs_error": {
            key: (float(errors[index]) if finite else None)
            for index, key in enumerate(FEATURE_KEYS)
        },
    }


def _rank_records(
    records: Sequence[Mapping[str, Any]],
    score_key: str,
) -> list[Mapping[str, Any]]:
    return sorted(
        records,
        key=lambda row: (
            bool(row["feasibility"]["feasible"]),
            row[score_key],
        ),
        reverse=True,
    )


def paired_comparison_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare within-context rankings without comparing raw reward magnitudes."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        context = record["context"]
        group_key = (
            f"{context['gesture']}::{context['target_mode']}::"
            f"{context.get('target_state')}::{json.dumps(context['target_vad'], sort_keys=True)}::"
            f"seed-{record['seed']}"
        )
        groups.setdefault(group_key, []).append(record)

    comparisons: list[dict[str, Any]] = []
    agreements: list[float] = []
    for group_key, group_records in sorted(groups.items()):
        vad_order = [
            row["candidate_id"]
            for row in _rank_records(group_records, "vad_score")
        ]
        categorical_available = all(
            row.get("categorical_score") is not None
            and bool((row.get("reliability") or {}).get("categorical_complete"))
            for row in group_records
        )
        categorical_order: list[str] | None = None
        agreement: float | None = None
        categorical_selected: str | None = None
        if categorical_available:
            categorical_order = [
                row["candidate_id"]
                for row in _rank_records(group_records, "categorical_score")
            ]
            positions = {candidate: index for index, candidate in enumerate(vad_order)}
            comparable = 0
            concordant = 0
            for left_index, left in enumerate(categorical_order):
                for right in categorical_order[left_index + 1 :]:
                    comparable += 1
                    concordant += int(positions[left] < positions[right])
            agreement = (
                float(concordant / comparable) if comparable else 1.0
            )
            agreements.append(agreement)
            categorical_selected = categorical_order[0]
        comparisons.append(
            {
                "group": group_key,
                "vad_ranking": vad_order,
                "categorical_ranking": categorical_order,
                "pairwise_ranking_agreement": agreement,
                "vad_selected": vad_order[0],
                "categorical_selected": categorical_selected,
                "same_selected_candidate": (
                    vad_order[0] == categorical_selected
                    if categorical_selected is not None
                    else None
                ),
            }
        )
    return {
        "reward_scale_note": REWARD_SCALE_NOTE,
        "mean_pairwise_ranking_agreement": (
            float(np.mean(agreements)) if agreements else None
        ),
        "groups": comparisons,
    }


def _save_csv(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        reliability = record.get("reliability") or {}
        feasibility = record["feasibility"]
        row = {
            "candidate_id": record["candidate_id"],
            "candidate_name": record["candidate_name"],
            "gesture": record["context"]["gesture"],
            "target_mode": record["context"]["target_mode"],
            "target_state": record["context"].get("target_state"),
            "target_vad": json.dumps(record["context"]["target_vad"], sort_keys=True),
            "seed": record["seed"],
            "repeat_count": reliability.get("repeat_reliability", {}).get(
                "repeat_count", 0
            ),
            "vad_score": record["vad_score"],
            "categorical_score": record.get("categorical_score"),
            "vad_reward_std": reliability.get("vad_reward_std"),
            "categorical_reward_std": reliability.get("categorical_reward_std"),
            "winner_agreement_rate": reliability.get("winner_agreement_rate"),
            "mean_probability_entropy": reliability.get("mean_probability_entropy"),
            "mean_confidence": reliability.get("mean_confidence"),
            "confidence_std": reliability.get("confidence_std"),
            "feasible": feasibility["feasible"],
            "realisation_rmse": feasibility["realisation_rmse"],
            "max_abs_feature_error": feasibility["max_abs_feature_error"],
        }
        for axis, value in (reliability.get("mean_observed_vad") or {}).items():
            row[f"{axis}_mean"] = value
            row[f"{axis}_std"] = reliability["per_axis_vad_std"][axis]
        rows.append(row)
    if not rows:
        return
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_plots(
    records: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
) -> None:
    """Write concise paired ranking and repeat-stability plots."""
    import matplotlib.pyplot as plt

    destination = Path(out_dir)
    plottable = [row for row in records if row.get("reliability") is not None]
    if not plottable:
        return
    labels = [str(row["candidate_id"]) for row in plottable]
    x = np.arange(len(plottable))
    vad_order = {
        row["candidate_id"]: rank
        for rank, row in enumerate(
            _rank_records(plottable, "vad_score"),
            start=1,
        )
    }
    categorical_rows = [
        row for row in plottable if row.get("categorical_score") is not None
    ]
    categorical_order = {
        row["candidate_id"]: rank
        for rank, row in enumerate(
            _rank_records(categorical_rows, "categorical_score"),
            start=1,
        )
    }

    plt.figure(figsize=(max(7, len(plottable) * 0.6), 4.5))
    plt.plot(x, [vad_order[label] for label in labels], "o-", label="VAD rank")
    if categorical_rows:
        plt.plot(
            x,
            [categorical_order.get(label, np.nan) for label in labels],
            "s--",
            label="Categorical rank",
        )
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Rank (1 = best)")
    plt.gca().invert_yaxis()
    plt.title("Paired candidate rankings (reward scales differ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination / "paired_rankings.png", dpi=160)
    plt.close()

    plt.figure(figsize=(max(7, len(plottable) * 0.6), 4.5))
    plt.bar(
        x,
        [row["reliability"]["vad_reward_std"] for row in plottable],
        label="VAD reward std",
    )
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Within-clip repeat standard deviation")
    plt.title("Paired repeat stability")
    plt.tight_layout()
    plt.savefig(destination / "paired_stability.png", dpi=160)
    plt.close()


def run_paired_experiment(
    cases: Sequence[ExperimentCase],
    *,
    inner_runner: Callable[[ExperimentCase, Path], LabanOptimisationResult],
    evaluator: Any,
    cache: PerceptualObservationCache,
    repeats: int,
    reward_config: EnvironmentRewardConfig,
    out_dir: str | Path,
    runner_identity: Mapping[str, Any],
    save_plots: bool = True,
    evaluator_identity: EvaluatorCacheIdentity | None = None,
) -> dict[str, Any]:
    """Run or resume a paired experiment, checkpointing after every candidate."""
    reward_config.validate()
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    identity = evaluator_identity or EvaluatorCacheIdentity.from_evaluator(evaluator)
    run_identity = _run_identity(
        cases=cases,
        repeats=repeats,
        evaluator_identity=identity,
        reward_config=reward_config,
        runner_identity=runner_identity,
    )
    results_path = destination / "paired_results.json"
    records: list[dict[str, Any]] = []
    if results_path.exists():
        saved = json.loads(results_path.read_text(encoding="utf-8"))
        if saved.get("run_identity") != run_identity:
            raise ValueError(
                "Existing paired_results.json is incompatible with this matrix run."
            )
        records = list(saved.get("records", []))
    completed = {record["candidate_id"] for record in records}

    for case in cases:
        if case.candidate_id in completed:
            continue
        result = inner_runner(case, destination / "candidates" / case.candidate_id)
        feasibility = _realisation_metrics(
            result,
            reward_config,
            require_feature_match=case.require_feature_match,
        )
        rmse = feasibility["realisation_rmse"]
        max_error = feasibility["max_abs_feature_error"]
        evaluator_eligible = bool(
            feasibility["valid_realisation"]
            and feasibility["physically_acceptable"]
            and (
                feasibility["feature_realisation_acceptable"]
                or not case.require_feature_match
                or not reward_config.reject_excessive_feature_error
            )
        )
        observations: list[dict[str, Any]] = []
        paired: dict[str, Any] | None = None
        observation_cache_key: str | None = None
        if evaluator_eligible:
            observation_cache_key = cache.cache_key(
                context=case.context,
                result=result,
                evaluator_identity=identity,
            )
            observations = cache.collect(
                context=case.context,
                result=result,
                evaluator=evaluator,
                repeats=repeats,
                evaluator_identity=identity,
            )
            paired = score_perceptual_observations(
                observations,
                target_vad=case.context.target_vad or {},
                target_state=case.context.target_state,
                reward_config=reward_config,
            )
        if paired is None or rmse is None or max_error is None:
            vad_score = reward_config.invalid_realisation_reward
            categorical_score = None
        else:
            penalty_rmse = rmse if case.require_feature_match else 0.0
            penalty_max_error = (
                max_error if case.require_feature_match else 0.0
            )
            vad_score = apply_outer_penalties(
                paired["mean_vad_reward"],
                perceptual_reward_std=paired["vad_reward_std"],
                realisation_rmse=penalty_rmse,
                max_abs_feature_error=penalty_max_error,
                reward_config=reward_config,
            )
            categorical_score = (
                apply_outer_penalties(
                    paired["categorical_reward"],
                    perceptual_reward_std=paired["categorical_reward_std"],
                    realisation_rmse=penalty_rmse,
                    max_abs_feature_error=penalty_max_error,
                    reward_config=reward_config,
                )
                if paired["categorical_reward"] is not None
                else None
            )
        records.append(
            {
                **case.to_dict(),
                "requested_profile": {
                    key: float(result.requested_profile[key])
                    for key in FEATURE_KEYS
                },
                "achieved_profile": {
                    key: float(result.achieved_profile[key])
                    for key in FEATURE_KEYS
                },
                "observations": observations,
                "observation_cache_key": observation_cache_key,
                "feasibility": feasibility,
                "vad_score": float(vad_score),
                "categorical_score": (
                    float(categorical_score)
                    if categorical_score is not None
                    else None
                ),
                "reliability": paired,
            }
        )
        partial_payload = {
            "run_identity": run_identity,
            "status": "in_progress",
            "records": records,
            "reward_scale_note": REWARD_SCALE_NOTE,
        }
        _atomic_json(results_path, partial_payload)

    unique_observation_sets = {
        record["observation_cache_key"]: record["observations"]
        for record in records
        if record["observations"] and record.get("observation_cache_key")
    }
    reliability = test_retest_reliability(
        list(unique_observation_sets.values())
    )
    comparison = paired_comparison_summary(records)
    payload = {
        "run_identity": run_identity,
        "status": "complete",
        "records": records,
        "test_retest_reliability": reliability,
        "paired_comparison": comparison,
        "reward_scale_note": REWARD_SCALE_NOTE,
    }
    _atomic_json(results_path, payload)
    _save_csv(records, destination / "paired_results.csv")
    if save_plots and records:
        save_comparison_plots(records, destination)
    return payload
