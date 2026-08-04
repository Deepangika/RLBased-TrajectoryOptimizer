"""Run gesture-conditioned outer learning from the fixed-profile baseline."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.affect import VAD_KEYS
from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import LabanOptimisationResult
from laban_rl.perceptual_bandit.baseline import (
    BASELINE_EXPERIMENT_NAME,
    BASELINE_ROOT,
    baseline_condition,
    baseline_optimizer_overrides,
    estimate_initial_covariance_from_baseline,
    file_sha256,
    guard_baseline_output_path,
    is_protected_baseline_path,
    load_baseline_motion,
)
from laban_rl.perceptual_bandit.environment import (
    Context,
    EnvironmentRewardConfig,
    MockNoisyPerceptualEvaluator,
    PerceptualBanditEnvironment,
)
from laban_rl.perceptual_bandit.evaluation_cache import PerceptualObservationCache
from laban_rl.perceptual_bandit.call_budget import (
    CallBudgetExhausted,
    GeminiCallBudget,
    get_active_budget,
    set_active_budget,
)
from laban_rl.perceptual_bandit.gemini_evaluator import GeminiProVideoEvaluator
from laban_rl.perceptual_bandit.outer_learning import (
    ContextualOuterLearner,
    LatentGaussianCEMDistribution,
    LatentSample,
    OuterLearningContext,
    classify_outcome,
    profile_to_vector,
    require_reward_consumption,
    robust_rank_key,
    sigmoid,
    stopping_reason,
    vector_to_profile,
    weighted_vad_reward,
)
from laban_rl.perceptual_bandit.paired_preference import (
    GeminiPairedPreferenceEvaluator,
    MockPairedPreferenceEvaluator,
    PairedPreferenceCache,
    PreferencePair,
    run_paired_preference_experiment,
)
from laban_rl.perceptual_bandit.selection import strict_realisability
from laban_rl.perceptual_bandit.variant_video import (
    render_variant_only_mp4,
    shared_camera_limits,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/outer_learning_diagnostic_v1.json",
    )
    parser.add_argument(
        "--stage",
        choices=("mock", "stage_a", "stage_b", "full"),
        default="mock",
    )
    parser.add_argument(
        "--evaluator",
        choices=("mock", "gemini", "synthetic", "synthetic_invalid"),
        default="synthetic",
    )
    parser.add_argument(
        "--out",
        default="outputs/experiments/outer_learning_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite-baseline", action="store_true")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--rounds-min", type=int, default=None)
    parser.add_argument("--rounds-max", type=int, default=None)
    parser.add_argument("--samples-per-round", type=int, default=None)
    parser.add_argument("--elite-count", type=int, default=None)
    parser.add_argument("--candidate-vlm-repeats", type=int, default=None)
    parser.add_argument("--validation-top-k", type=int, default=None)
    parser.add_argument("--validation-vlm-repeats", type=int, default=None)
    parser.add_argument("--paired-validation-repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.50)
    parser.add_argument("--covariance-smoothing", type=float, default=0.30)
    parser.add_argument("--mean-smoothing", type=float, default=0.40)
    parser.add_argument("--minimum-eigenvalue", type=float, default=1e-3)
    parser.add_argument("--maximum-eigenvalue", type=float, default=1.0)
    parser.add_argument("--covariance-history-rounds", type=int, default=3)
    parser.add_argument("--round-weight-decay", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--minimum-reward-improvement", type=float, default=0.01)
    parser.add_argument(
        "--maximum-action-std-for-convergence",
        type=float,
        default=0.04,
    )
    parser.add_argument("--realisation-penalty-weight", type=float, default=0.25)
    parser.add_argument("--max-feature-error-threshold", type=float, default=0.10)
    parser.add_argument(
        "--robust-elite-max-feature-error",
        type=float,
        default=None,
        help=(
            "Robust-elite ranking margin. Candidates at or below this maximum "
            "feature error are ranked as robustly feasible. Does not redefine "
            "formal feasibility, which stays at --max-feature-error-threshold."
        ),
    )
    parser.add_argument(
        "--max-gemini-calls",
        type=int,
        default=110,
        help=(
            "Hard ceiling on the total number of Gemini API calls for this run, "
            "including billable retries. The run stops safely before the call "
            "that would exceed the ceiling. Only applies with --evaluator gemini."
        ),
    )
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_payload(config: dict[str, Any], stage: str) -> dict[str, Any]:
    return dict(config["stages"][stage])


def _stage_dir(base: Path, stage: str) -> Path:
    return {
        "mock": base / "mock",
        "stage_a": base / "live_stage_a",
        "stage_b": base / "live_stage_b",
        "full": base / "full",
    }[stage]


def _context_dir(base: Path, context: OuterLearningContext) -> Path:
    return base / context.gesture / context.target_state


def _short_cache_root(out_dir: Path, label: str) -> Path:
    digest = hashlib.sha256(str(out_dir).encode("utf-8")).hexdigest()[:16]
    return ROOT / "outputs" / "_outer_learning_cache" / digest / label


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_seed(
    *,
    base_seed: int,
    context: OuterLearningContext,
    round_index: int,
    sample_index: int,
) -> int:
    payload = f"{base_seed}|{context.key}|{round_index}|{sample_index}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big") % (2**32)


def _checkpoint_file(
    out_dir: Path,
    *,
    round_index: int,
    sample_index: int,
) -> Path:
    return (
        out_dir
        / "checkpoint"
        / "candidates"
        / f"round_{round_index:03d}"
        / f"sample_{sample_index:03d}.json"
    )


def _round_state_file(out_dir: Path, *, round_index: int) -> Path:
    return out_dir / "checkpoint" / "rounds" / f"round_{round_index:03d}_state.json"


def _checkpoint_state_file(out_dir: Path) -> Path:
    return out_dir / "checkpoint" / "checkpoint_state.json"


def _checkpoint_record_is_complete(
    payload: dict[str, Any],
    *,
    sample: LatentSample,
    round_index: int,
    sample_index: int,
    candidate_seed: int,
) -> bool:
    required = {
        "schema_version",
        "status",
        "round_index",
        "sample_index",
        "sample_id",
        "candidate_seed",
        "latent",
        "action_profile",
        "feasible",
        "ineligibility_reasons",
        "reward",
        "result",
    }
    if not required.issubset(payload):
        return False
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        return False
    if int(payload.get("round_index", -1)) != int(round_index):
        return False
    if int(payload.get("sample_index", -1)) != int(sample_index):
        return False
    if str(payload.get("sample_id")) != sample.sample_id:
        return False
    if int(payload.get("candidate_seed", -1)) != int(candidate_seed):
        return False
    latent = payload.get("latent")
    action_profile = payload.get("action_profile")
    result = payload.get("result")
    if not isinstance(latent, list) or len(latent) != len(FEATURE_KEYS):
        return False
    if not isinstance(action_profile, dict) or any(key not in action_profile for key in FEATURE_KEYS):
        return False
    if not isinstance(result, dict):
        return False
    try:
        reward = float(payload.get("reward"))
        result_reward = float(result.get("outer_reward"))
    except (TypeError, ValueError):
        return False
    if not np.isfinite(reward) or not np.isfinite(result_reward):
        return False
    if abs(reward - result_reward) > 1e-9:
        return False
    return True


def _robustness_fields(
    *,
    feasible: bool,
    result_dict: dict[str, Any],
    tolerance: float,
    robust_margin: float,
    sample_index: int,
) -> dict[str, Any]:
    """Diagnostic ranking fields. Never redefines formal feasibility."""
    raw_error = result_dict.get("max_abs_feature_error")
    max_error = float(raw_error) if raw_error is not None else None
    robust = bool(feasible and max_error is not None and max_error <= robust_margin)
    if not feasible:
        category = "infeasible"
    elif robust:
        category = "robust_feasible"
    else:
        category = "marginal_feasible"
    return {
        "strictly_feasible": bool(feasible),
        "robustly_feasible": robust,
        "max_abs_feature_error": max_error,
        "distance_to_official_tolerance": (
            float(tolerance) - max_error if max_error is not None else None
        ),
        "distance_to_robust_margin": (
            float(robust_margin) - max_error if max_error is not None else None
        ),
        "ranking_category": category,
        "sample_index": int(sample_index),
    }


def _checkpoint_to_completed_sample(
    payload: dict[str, Any],
    *,
    tolerance: float,
    robust_margin: float,
) -> LatentSample:
    result = payload["result"] if isinstance(payload.get("result"), dict) else {}
    metadata = {
        "mean_observed_vad": result.get("mean_observed_vad"),
        "realisation_rmse": result.get("realisation_rmse"),
        "candidate_seed": int(payload["candidate_seed"]),
    }
    metadata.update(
        _robustness_fields(
            feasible=bool(payload["feasible"]),
            result_dict=result,
            tolerance=tolerance,
            robust_margin=robust_margin,
            sample_index=int(payload["sample_index"]),
        )
    )
    return LatentSample(
        sample_id=str(payload["sample_id"]),
        round_index=int(payload["round_index"]),
        action={key: float(payload["action_profile"][key]) for key in FEATURE_KEYS},
        latent=tuple(float(value) for value in payload["latent"]),
        reward=float(payload["reward"]),
        feasible=bool(payload["feasible"]),
        evaluation_consumed=True,
        metadata=metadata,
    )


def _summarise_optimisation_result(
    optimisation_result: LabanOptimisationResult | None,
) -> dict[str, Any] | None:
    if optimisation_result is None:
        return None
    return {
        "output_dir": str(optimisation_result.output_dir),
        "inner_reward": float(optimisation_result.inner_reward),
        "inner_loss": float(optimisation_result.inner_loss),
        "requested_profile": dict(optimisation_result.requested_profile),
        "achieved_profile": dict(optimisation_result.achieved_profile),
    }


def _candidate_record(
    *,
    sample: LatentSample,
    round_index: int,
    sample_index: int,
    candidate_seed: int,
    result_dict: dict[str, Any],
    feasible: bool,
    ineligibility_reasons: list[str],
    sample_output_dir: Path,
    optimisation_result: LabanOptimisationResult | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": _utc_timestamp(),
        "round_index": round_index,
        "sample_index": sample_index,
        "sample_id": sample.sample_id,
        "candidate_seed": int(candidate_seed),
        "latent": [float(value) for value in sample.latent],
        "action_profile": {key: float(sample.action[key]) for key in FEATURE_KEYS},
        "feasible": bool(feasible),
        "ineligibility_reasons": list(ineligibility_reasons),
        "reward": float(result_dict["outer_reward"]),
        "result": result_dict,
        "optimisation_summary": _summarise_optimisation_result(optimisation_result),
        "sample_output_dir": str(sample_output_dir),
    }


def _checkpoint_settings_payload(
    *,
    context: OuterLearningContext,
    evaluator_name: str,
    model: str,
    temperature: float,
    rounds_min: int,
    rounds_max: int,
    samples_per_round: int,
    elite_count: int,
    candidate_vlm_repeats: int,
    validation_top_k: int,
    validation_vlm_repeats: int,
    paired_validation_repeats: int,
    covariance_shrinkage: float,
    covariance_smoothing: float,
    mean_smoothing: float,
    minimum_eigenvalue: float,
    maximum_eigenvalue: float,
    covariance_history_rounds: int,
    round_weight_decay: float,
    plateau_patience: int,
    minimum_reward_improvement: float,
    maximum_action_std_for_convergence: float,
    realisation_penalty_weight: float,
    max_feature_error_threshold: float,
    robust_elite_max_feature_error: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "context": asdict(context),
        "evaluator": evaluator_name,
        "model": model if evaluator_name == "gemini" else None,
        "temperature": float(temperature),
        "rounds_min": int(rounds_min),
        "rounds_max": int(rounds_max),
        "samples_per_round": int(samples_per_round),
        "elite_count": int(elite_count),
        "candidate_vlm_repeats": int(candidate_vlm_repeats),
        "validation_top_k": int(validation_top_k),
        "validation_vlm_repeats": int(validation_vlm_repeats),
        "paired_validation_repeats": int(paired_validation_repeats),
        "covariance_shrinkage": float(covariance_shrinkage),
        "covariance_smoothing": float(covariance_smoothing),
        "mean_smoothing": float(mean_smoothing),
        "minimum_eigenvalue": float(minimum_eigenvalue),
        "maximum_eigenvalue": float(maximum_eigenvalue),
        "covariance_history_rounds": int(covariance_history_rounds),
        "round_weight_decay": float(round_weight_decay),
        "plateau_patience": int(plateau_patience),
        "minimum_reward_improvement": float(minimum_reward_improvement),
        "maximum_action_std_for_convergence": float(maximum_action_std_for_convergence),
        "realisation_penalty_weight": float(realisation_penalty_weight),
        "max_feature_error_threshold": float(max_feature_error_threshold),
        "robust_elite_max_feature_error": float(robust_elite_max_feature_error),
        "seed": int(seed),
    }


def _evaluate_candidate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    context = OuterLearningContext(
        gesture=str(payload["context"]["gesture"]),
        target_state=str(payload["context"]["target_state"]),
    )
    target_context = Context(gesture=context.gesture, target_state=context.target_state)
    synthetic = bool(payload["synthetic"])
    synthetic_invalid = bool(payload["synthetic_invalid"])
    evaluator_name = str(payload["evaluator_name"])
    optimiser_overrides = dict(payload["optimiser_overrides"])
    candidate_seed = int(payload["candidate_seed"])
    optimiser_overrides["seed"] = candidate_seed
    sample_output_dir = Path(str(payload["sample_output_dir"]))
    action_profile = {key: float(payload["action_profile"][key]) for key in FEATURE_KEYS}
    repeats = int(payload["candidate_vlm_repeats"])
    model = str(payload["model"])
    temperature = float(payload["temperature"])
    realisation_penalty_weight = float(payload["realisation_penalty_weight"])
    max_feature_error_threshold = float(payload["max_feature_error_threshold"])
    result_dict: dict[str, Any]
    optimisation_result: LabanOptimisationResult | None
    if synthetic:
        result_dict, optimisation_result = _evaluate_profile(
            None,
            target_context,
            action_profile,
            sample_output_dir,
            synthetic=True,
            synthetic_invalid=synthetic_invalid,
        )
    else:
        cache_label = str(payload["cache_label"])
        environment = _make_environment(
            evaluator_name=evaluator_name,
            model=model,
            temperature=temperature,
            repeats=repeats,
            optimiser_overrides=optimiser_overrides,
            observation_cache=_short_cache_root(Path(str(payload["out_dir"])), cache_label),
            realisation_penalty_weight=realisation_penalty_weight,
            max_feature_error_threshold=max_feature_error_threshold,
            evaluator_seed=candidate_seed,
        )
        try:
            result_dict, optimisation_result = _evaluate_profile(
                environment,
                target_context,
                action_profile,
                sample_output_dir,
                synthetic=False,
            )
        finally:
            close = getattr(environment.evaluator, "close", None)
            if callable(close):
                close()
    return {
        "result": result_dict,
        "optimisation_summary": _summarise_optimisation_result(optimisation_result),
    }


def _successful_outcome(outcome: str) -> bool:
    return outcome in {
        "full_perceptual_success",
        "relative_preference_improvement",
        "vad_improvement_without_quadrant_match",
        "quadrant_match_without_stable_reward",
        "synthetic_objective_improvement",
        "synthetic_optimum_recovered",
        "mock_validation_complete",
    }


def _require_stage_a_success(base_out: Path) -> None:
    status_path = base_out / "live_stage_a" / "stage_status.json"
    if not status_path.exists():
        raise RuntimeError("Stage B requires a completed successful Stage A.")
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    if not bool(payload.get("all_successful", False)):
        raise RuntimeError("Stage B requires Stage A to complete successfully.")


def _contexts_from_stage(stage_payload: dict[str, Any]) -> list[OuterLearningContext]:
    return [
        OuterLearningContext(
            gesture=str(item["gesture"]),
            target_state=str(item["target_state"]),
        )
        for item in stage_payload["contexts"]
    ]


def _canonical_feasibility(
    result: dict[str, Any],
    *,
    tolerance: float,
) -> tuple[bool, list[str]]:
    return strict_realisability(result, tolerance=tolerance)


def _assert_feasibility_consistency(
    *,
    feasible: bool,
    result: dict[str, Any],
    tolerance: float,
    source: str,
) -> None:
    canonical, reasons = _canonical_feasibility(result, tolerance=tolerance)
    if bool(feasible) != bool(canonical):
        raise RuntimeError(
            f"Feasibility invariant failed for {source}: "
            f"sample/selection feasible={feasible} but canonical={canonical} "
            f"with reasons={reasons}"
        )


def _estimate_live_calls(
    stage_payload: dict[str, Any],
    *,
    rounds_max: int,
    samples_per_round: int,
    candidate_vlm_repeats: int,
    validation_top_k: int,
    validation_vlm_repeats: int,
    paired_validation_repeats: int,
) -> dict[str, int]:
    contexts = len(stage_payload["contexts"])
    per_context = (
        rounds_max * samples_per_round * candidate_vlm_repeats
        + validation_top_k * validation_vlm_repeats
        + 2 * paired_validation_repeats
    )
    return {
        "contexts": contexts,
        "estimated_calls_per_context": per_context,
        "estimated_total_calls": contexts * per_context,
    }


def _make_environment(
    *,
    evaluator_name: str,
    model: str,
    temperature: float,
    repeats: int,
    optimiser_overrides: dict[str, Any],
    observation_cache: Path,
    realisation_penalty_weight: float,
    max_feature_error_threshold: float,
    evaluator_seed: int = 7,
) -> PerceptualBanditEnvironment:
    if evaluator_name == "gemini":
        evaluator = GeminiProVideoEvaluator(
            model=model,
            temperature=temperature,
            video_duration_seconds=2.0,
            video_lead_in_seconds=0.5,
            video_repetitions=2,
            video_inter_repeat_transition_seconds=0.5,
            video_final_hold_seconds=0.5,
            keep_uploaded_files=False,
        )
    else:
        evaluator = MockNoisyPerceptualEvaluator(seed=evaluator_seed)
    return PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=repeats,
            perceptual_reward_mode="vad",
            valence_weight=0.20,
            arousal_weight=0.40,
            dominance_weight=0.40,
            realisation_penalty_weight=realisation_penalty_weight,
            max_feature_error_threshold=max_feature_error_threshold,
            reject_excessive_feature_error=True,
            evaluator_failure_mode="raise",
        ),
        optimiser_overrides=optimiser_overrides,
        observation_cache=PerceptualObservationCache(observation_cache),
    )


def _synthetic_hidden_profile(context: OuterLearningContext) -> dict[str, float]:
    seed = int.from_bytes(
        hashlib.sha256(context.key.encode("utf-8")).digest()[:8],
        "big",
    ) % (2**32)
    rng = np.random.default_rng(seed)
    latent = rng.multivariate_normal(
        mean=np.array([0.4, -0.1, 0.3, -0.2, 0.2], dtype=float),
        cov=np.array(
            [
                [0.35, 0.18, 0.12, 0.00, 0.08],
                [0.18, 0.30, 0.10, -0.05, 0.02],
                [0.12, 0.10, 0.28, 0.06, 0.08],
                [0.00, -0.05, 0.06, 0.24, 0.12],
                [0.08, 0.02, 0.08, 0.12, 0.22],
            ],
            dtype=float,
        ),
    )
    return vector_to_profile(sigmoid(latent))


def _synthetic_objective_distance(
    action_profile: dict[str, float],
    optimum: dict[str, float],
) -> float:
    action = profile_to_vector(action_profile)
    optimum_vector = profile_to_vector(optimum)
    return float(np.sqrt(np.mean((action - optimum_vector) ** 2)))


def _synthetic_step(
    context: OuterLearningContext,
    action_profile: dict[str, float],
    *,
    inject_invalid: bool = False,
) -> dict[str, Any]:
    optimum = _synthetic_hidden_profile(context)
    action = profile_to_vector(action_profile)
    objective_distance = _synthetic_objective_distance(action_profile, optimum)
    reward = 1.0 - objective_distance
    observed_vad = {
        "valence": float(0.55 * action[0] + 0.45 * action[4]),
        "arousal": float(0.65 * action[1] + 0.35 * action[2]),
        "dominance": float(0.55 * action[0] + 0.45 * action[3]),
    }
    valid_realisation = not inject_invalid
    physically_acceptable = not inject_invalid
    feature_realisation_acceptable = not inject_invalid
    return {
        "outer_reward": reward,
        "mean_vad_reward": reward,
        "mean_observed_vad": observed_vad,
        "realisation_rmse": 0.0 if not inject_invalid else 1.0,
        "valid_realisation": valid_realisation,
        "physically_acceptable": physically_acceptable,
        "feature_realisation_acceptable": feature_realisation_acceptable,
        "max_abs_feature_error": 0.0 if not inject_invalid else 1.0,
        "per_feature_abs_error": {key: 0.0 if not inject_invalid else 1.0 for key in FEATURE_KEYS},
        "synthetic_objective_distance": objective_distance,
        "optimisation_result": None,
        "raw_result": {"synthetic_optimum": optimum, "inject_invalid": inject_invalid},
    }


def _camera_limits_for_gesture(
    gesture: str,
    learned_result: LabanOptimisationResult,
) -> tuple[tuple[float, float], tuple[float, float]]:
    trajectories = [learned_result.q_var]
    for state in ("anger", "disgust", "fear", "happiness", "sadness", "surprise"):
        trajectories.append(load_baseline_motion(gesture, state, candidate="styled").q_var)
        trajectories.append(load_baseline_motion(gesture, state, candidate="reference").q_var)
    return shared_camera_limits(
        trajectories,
        duration_seconds=2.0,
        lead_in_seconds=0.5,
        repetitions=2,
        inter_repeat_transition_seconds=0.5,
        final_hold_seconds=0.5,
    )


def _render_final_assets(
    out_dir: Path,
    optimisation_result: LabanOptimisationResult,
    *,
    gesture: str,
) -> None:
    camera_limits = _camera_limits_for_gesture(gesture, optimisation_result)
    render_variant_only_mp4(
        optimisation_result.q_ref,
        optimisation_result.q_var,
        out_dir / "final_motion.mp4",
        duration_seconds=2.0,
        lead_in_seconds=0.5,
        repetitions=2,
        inter_repeat_transition_seconds=0.5,
        final_hold_seconds=0.5,
        camera_limits=camera_limits,
        overwrite=True,
    )
    source_gif = optimisation_result.output_dir / "arm_comparison.gif"
    if source_gif.exists():
        shutil.copy2(source_gif, out_dir / "final_motion.gif")


def _plot_reward_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(
        [row["round"] for row in rows],
        [row["best_reward"] for row in rows],
        marker="o",
    )
    plt.xlabel("Round")
    plt.ylabel("Best reward")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_covariance_history(path: Path, diagonals: list[np.ndarray]) -> None:
    if not diagonals:
        return
    plt.figure(figsize=(8, 4))
    for index, key in enumerate(FEATURE_KEYS):
        plt.plot(
            range(1, len(diagonals) + 1),
            [row[index] for row in diagonals],
            marker="o",
            label=key,
        )
    plt.xlabel("Round")
    plt.ylabel("Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_vad_trajectory(
    path: Path,
    *,
    target_vad: dict[str, float],
    baseline_vad: dict[str, float],
    round_best: list[dict[str, float]],
    final_vad: dict[str, float],
) -> None:
    plt.figure(figsize=(6, 6))
    plt.axhline(0.5, color="0.8")
    plt.axvline(0.5, color="0.8")
    plt.scatter(
        [target_vad["valence"]],
        [target_vad["arousal"]],
        label="Canonical target",
        marker="*",
        s=180,
    )
    plt.scatter(
        [baseline_vad["valence"]],
        [baseline_vad["arousal"]],
        label="Fixed baseline",
        marker="s",
    )
    if round_best:
        plt.plot(
            [item["valence"] for item in round_best],
            [item["arousal"] for item in round_best],
            marker="o",
            label="Best per round",
        )
    plt.scatter(
        [final_vad["valence"]],
        [final_vad["arousal"]],
        label="Final validated",
        marker="D",
    )
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_covariance_history(path: Path, matrices: list[np.ndarray]) -> None:
    payload = {
        f"round_{index + 1:03d}": matrix
        for index, matrix in enumerate(matrices)
    }
    np.savez_compressed(path, **payload)


def _evaluate_profile(
    environment: PerceptualBanditEnvironment | None,
    context: Context,
    profile: dict[str, float],
    out_dir: Path,
    *,
    synthetic: bool,
    synthetic_invalid: bool = False,
) -> tuple[dict[str, Any], LabanOptimisationResult | None]:
    if synthetic:
        return _synthetic_step(
            OuterLearningContext(context.gesture, str(context.target_state)),
            profile,
            inject_invalid=synthetic_invalid,
        ), None
    optimisation_result = environment.run_inner_step(
        context=context,
        action_profile=profile,
        out_dir=out_dir,
    )
    step_result = environment.step_from_result(
        context=context,
        optimisation_result=optimisation_result,
    )
    return step_result.to_dict(), optimisation_result


def _paired_validation(
    *,
    pair_id: str,
    context: Context,
    learned: LabanOptimisationResult,
    other: LabanOptimisationResult,
    target_layers: dict[str, Any],
    out_path: Path,
    cache_root: Path,
    repeats: int,
    evaluator_name: str,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    if evaluator_name == "gemini":
        evaluator = GeminiPairedPreferenceEvaluator(
            model=model,
            temperature=temperature,
            video_duration_seconds=2.0,
            video_lead_in_seconds=0.5,
            video_repetitions=2,
            video_inter_repeat_transition_seconds=0.5,
            video_final_hold_seconds=0.5,
            keep_uploaded_files=False,
        )
    else:
        evaluator = MockPairedPreferenceEvaluator()
    try:
        return run_paired_preference_experiment(
            [
                PreferencePair(
                    pair_id=pair_id,
                    context=context,
                    reference_result=other,
                    styled_result=learned,
                    target_layers=target_layers,
                )
            ],
            evaluator=evaluator,
            cache=PairedPreferenceCache(cache_root),
            repeats=repeats,
            out_path=out_path,
        )
    finally:
        close = getattr(evaluator, "close", None)
        if callable(close):
            close()


def run_context(
    *,
    context: OuterLearningContext,
    out_dir: Path,
    evaluator_name: str,
    model: str,
    temperature: float,
    rounds_min: int,
    rounds_max: int,
    samples_per_round: int,
    elite_count: int,
    candidate_vlm_repeats: int,
    validation_top_k: int,
    validation_vlm_repeats: int,
    paired_validation_repeats: int,
    covariance_shrinkage: float,
    covariance_smoothing: float,
    mean_smoothing: float,
    minimum_eigenvalue: float,
    maximum_eigenvalue: float,
    covariance_history_rounds: int,
    round_weight_decay: float,
    plateau_patience: int,
    minimum_reward_improvement: float,
    maximum_action_std_for_convergence: float,
    realisation_penalty_weight: float,
    max_feature_error_threshold: float,
    seed: int,
    robust_elite_max_feature_error: float = 0.08,
    resume: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    baseline = baseline_condition(context.gesture, context.target_state)
    target_context = Context(
        gesture=context.gesture,
        target_state=context.target_state,
    )
    initial_profile = (
        baseline.projected_feasible_initialisation
        or baseline.requested_optimization
    )
    initial_covariance = estimate_initial_covariance_from_baseline(
        context.gesture,
        min_eigenvalue=minimum_eigenvalue,
    )
    initial_covariance_snapshot = np.asarray(initial_covariance, dtype=float).copy()
    distribution = LatentGaussianCEMDistribution(
        initial_action_mean=initial_profile,
        initial_covariance=initial_covariance_snapshot.copy(),
        covariance_shrinkage=covariance_shrinkage,
        covariance_smoothing=covariance_smoothing,
        mean_smoothing=mean_smoothing,
        minimum_eigenvalue=minimum_eigenvalue,
        maximum_eigenvalue=maximum_eigenvalue,
        covariance_history_rounds=covariance_history_rounds,
        round_weight_decay=round_weight_decay,
        seed=seed,
    )
    learner = ContextualOuterLearner({context.key: distribution})
    _ = learner.distribution(context)
    optimiser_overrides = baseline_optimizer_overrides(
        context.gesture,
        context.target_state,
    )
    optimiser_overrides.setdefault("seed", seed)
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if not 0.0 < robust_elite_max_feature_error <= max_feature_error_threshold:
        raise ValueError(
            "robust_elite_max_feature_error must be positive and must not "
            "exceed the official max_feature_error_threshold."
        )
    synthetic_invalid = evaluator_name == "synthetic_invalid"
    synthetic = evaluator_name == "synthetic" or synthetic_invalid
    effective_workers = 1 if synthetic else workers
    checkpoint_settings = _checkpoint_settings_payload(
        context=context,
        evaluator_name=evaluator_name,
        model=model,
        temperature=temperature,
        rounds_min=rounds_min,
        rounds_max=rounds_max,
        samples_per_round=samples_per_round,
        elite_count=elite_count,
        candidate_vlm_repeats=candidate_vlm_repeats,
        validation_top_k=validation_top_k,
        validation_vlm_repeats=validation_vlm_repeats,
        paired_validation_repeats=paired_validation_repeats,
        covariance_shrinkage=covariance_shrinkage,
        covariance_smoothing=covariance_smoothing,
        mean_smoothing=mean_smoothing,
        minimum_eigenvalue=minimum_eigenvalue,
        maximum_eigenvalue=maximum_eigenvalue,
        covariance_history_rounds=covariance_history_rounds,
        round_weight_decay=round_weight_decay,
        plateau_patience=plateau_patience,
        minimum_reward_improvement=minimum_reward_improvement,
        maximum_action_std_for_convergence=maximum_action_std_for_convergence,
        realisation_penalty_weight=realisation_penalty_weight,
        max_feature_error_threshold=max_feature_error_threshold,
        robust_elite_max_feature_error=robust_elite_max_feature_error,
        seed=seed,
    )
    checkpoint_state_path = _checkpoint_state_file(out_dir)
    checkpoint_state = {
        "schema_version": 1,
        "created_at_utc": _utc_timestamp(),
        "context": asdict(context),
        "run_settings": checkpoint_settings,
        "resume_events": 0,
        "recovered_candidates": 0,
        "recomputed_candidates": 0,
        "worker_failures": 0,
        "latest_round_completed": 0,
    }
    if resume:
        if not checkpoint_state_path.exists():
            # No prior progress for this context: start it fresh within a
            # stage-level resume instead of failing the whole stage.
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            loaded_state = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))
            saved_settings = loaded_state.get("run_settings")
            if saved_settings != checkpoint_settings:
                raise RuntimeError(
                    f"Cannot resume {context.key}: saved run settings differ. "
                    "Start a new run with --overwrite."
                )
            checkpoint_state = dict(loaded_state)
            checkpoint_state["resume_events"] = int(checkpoint_state.get("resume_events", 0)) + 1
            checkpoint_state["resumed_at_utc"] = _utc_timestamp()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(checkpoint_state_path, checkpoint_state)

    validation_environment = None
    if not synthetic:
        validation_environment = _make_environment(
            evaluator_name=evaluator_name,
            model=model,
            temperature=temperature,
            repeats=validation_vlm_repeats,
            optimiser_overrides=optimiser_overrides,
            observation_cache=_short_cache_root(out_dir, "v"),
            realisation_penalty_weight=realisation_penalty_weight,
            max_feature_error_threshold=max_feature_error_threshold,
            evaluator_seed=seed,
        )

    round_history: list[dict[str, Any]] = []
    reward_history: list[dict[str, Any]] = []
    sample_history: list[dict[str, Any]] = []
    elite_history: list[dict[str, Any]] = []
    mean_history: list[dict[str, Any]] = []
    correlation_history: list[dict[str, Any]] = []
    covariance_matrices: list[np.ndarray] = []
    best_by_round: list[dict[str, float]] = []
    archived_records: list[dict[str, Any]] = []
    best_reward_so_far = float("-inf")
    plateau_rounds = 0
    stop = None
    final_selected_result: dict[str, Any] | None = None
    final_selected_opt_result: LabanOptimisationResult | None = None
    selection_status = "pending"
    synthetic_initial_distance = (
        _synthetic_objective_distance(initial_profile, _synthetic_hidden_profile(context))
        if synthetic
        else None
    )
    for round_index in range(1, rounds_max + 1):
        _budget = get_active_budget()
        if _budget is not None:
            _budget.set_category("search")
        samples = distribution.sample_batch(
            samples_per_round,
            round_index=round_index,
            prefix=context.gesture,
        )
        completed_samples: list[LatentSample] = []
        feasible_count = 0
        round_rewards: list[float] = []
        round_best_vad: dict[str, float] | None = None
        sample_records: list[tuple[int, LatentSample, dict[str, Any], dict[str, Any] | None]] = []
        payload_by_index: dict[int, dict[str, Any]] = {}
        for sample_index, sample in enumerate(samples):
            sample_output_dir = out_dir / "rounds" / f"round_{round_index:03d}" / f"sample_{sample_index:03d}"
            candidate_seed = _candidate_seed(
                base_seed=seed,
                context=context,
                round_index=round_index,
                sample_index=sample_index,
            )
            payload_by_index[sample_index] = {
                "context": asdict(context),
                "evaluator_name": evaluator_name,
                "model": model,
                "temperature": temperature,
                "candidate_vlm_repeats": candidate_vlm_repeats,
                "optimiser_overrides": dict(optimiser_overrides),
                "realisation_penalty_weight": realisation_penalty_weight,
                "max_feature_error_threshold": max_feature_error_threshold,
                "candidate_seed": candidate_seed,
                "synthetic": synthetic,
                "synthetic_invalid": synthetic_invalid,
                "action_profile": dict(sample.action),
                "sample_output_dir": str(sample_output_dir),
                "out_dir": str(out_dir),
                "cache_label": f"t_r{round_index:03d}_s{sample_index:03d}",
            }
            checkpoint_path = _checkpoint_file(
                out_dir,
                round_index=round_index,
                sample_index=sample_index,
            )
            if checkpoint_path.exists():
                try:
                    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    checkpoint_payload = None
                if checkpoint_payload is not None and _checkpoint_record_is_complete(
                    checkpoint_payload,
                    sample=sample,
                    round_index=round_index,
                    sample_index=sample_index,
                    candidate_seed=candidate_seed,
                ):
                    completed = _checkpoint_to_completed_sample(
                        checkpoint_payload,
                        tolerance=max_feature_error_threshold,
                        robust_margin=robust_elite_max_feature_error,
                    )
                    sample_records.append(
                        (
                            sample_index,
                            completed,
                            dict(checkpoint_payload["result"]),
                            (
                                dict(checkpoint_payload["optimisation_summary"])
                                if checkpoint_payload.get("optimisation_summary") is not None
                                else None
                            ),
                        )
                    )
                    checkpoint_state["recovered_candidates"] = int(checkpoint_state.get("recovered_candidates", 0)) + 1
                    _write_json(checkpoint_state_path, checkpoint_state)
                    continue
                checkpoint_state["recomputed_candidates"] = int(checkpoint_state.get("recomputed_candidates", 0)) + 1
                _write_json(checkpoint_state_path, checkpoint_state)
            payload_by_index[sample_index]["sample_id"] = sample.sample_id
            payload_by_index[sample_index]["latent"] = list(sample.latent)
            payload_by_index[sample_index]["round_index"] = round_index
            payload_by_index[sample_index]["sample_index"] = sample_index

        pending_indices = [
            index for index in range(len(samples))
            if not any(record[0] == index for record in sample_records)
        ]
        if pending_indices:
            if effective_workers == 1:
                for sample_index in pending_indices:
                    sample = samples[sample_index]
                    payload = payload_by_index[sample_index]
                    try:
                        output = _evaluate_candidate_payload(payload)
                    except CallBudgetExhausted:
                        stop = "call_budget_exhausted"
                        selection_status = "call_budget_exhausted"
                        break
                    except Exception:
                        stop = stopping_reason(
                            round_index=round_index,
                            minimum_rounds=rounds_min,
                            maximum_rounds=rounds_max,
                            plateau_rounds=plateau_rounds,
                            plateau_patience=plateau_patience,
                            current_action_std=distribution.action_standard_deviation(),
                            maximum_action_std_for_convergence=maximum_action_std_for_convergence,
                            feasible_candidates=feasible_count,
                            evaluator_failed=True,
                        )
                        selection_status = "evaluator_failure"
                        break
                    result_dict = dict(output["result"])
                    feasible, reasons = _canonical_feasibility(
                        result_dict,
                        tolerance=max_feature_error_threshold,
                    )
                    _assert_feasibility_consistency(
                        feasible=feasible,
                        result=result_dict,
                        tolerance=max_feature_error_threshold,
                        source=sample.sample_id,
                    )
                    completed = LatentSample(
                        sample_id=sample.sample_id,
                        round_index=round_index,
                        action=sample.action,
                        latent=sample.latent,
                        reward=float(result_dict["outer_reward"]),
                        feasible=feasible,
                        evaluation_consumed=True,
                        metadata={
                            "mean_observed_vad": result_dict.get("mean_observed_vad"),
                            "realisation_rmse": result_dict.get("realisation_rmse"),
                            "candidate_seed": int(payload["candidate_seed"]),
                            **_robustness_fields(
                                feasible=feasible,
                                result_dict=result_dict,
                                tolerance=max_feature_error_threshold,
                                robust_margin=robust_elite_max_feature_error,
                                sample_index=sample_index,
                            ),
                        },
                    )
                    sample_records.append(
                        (
                            sample_index,
                            completed,
                            result_dict,
                            (
                                dict(output["optimisation_summary"])
                                if output.get("optimisation_summary") is not None
                                else None
                            ),
                        )
                    )
                    checkpoint_payload = _candidate_record(
                        sample=completed,
                        round_index=round_index,
                        sample_index=sample_index,
                        candidate_seed=int(payload["candidate_seed"]),
                        result_dict=result_dict,
                        feasible=feasible,
                        ineligibility_reasons=reasons,
                        sample_output_dir=Path(str(payload["sample_output_dir"])),
                        optimisation_result=None,
                    )
                    checkpoint_payload["optimisation_summary"] = output.get("optimisation_summary")
                    _write_json(
                        _checkpoint_file(out_dir, round_index=round_index, sample_index=sample_index),
                        checkpoint_payload,
                    )
                    _write_json(checkpoint_state_path, checkpoint_state)
            else:
                with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                    futures = {
                        executor.submit(_evaluate_candidate_payload, payload_by_index[sample_index]): sample_index
                        for sample_index in pending_indices
                    }
                    for future in as_completed(futures):
                        sample_index = futures[future]
                        sample = samples[sample_index]
                        payload = payload_by_index[sample_index]
                        try:
                            output = future.result()
                        except Exception:
                            checkpoint_state["worker_failures"] = int(checkpoint_state.get("worker_failures", 0)) + 1
                            _write_json(checkpoint_state_path, checkpoint_state)
                            stop = stopping_reason(
                                round_index=round_index,
                                minimum_rounds=rounds_min,
                                maximum_rounds=rounds_max,
                                plateau_rounds=plateau_rounds,
                                plateau_patience=plateau_patience,
                                current_action_std=distribution.action_standard_deviation(),
                                maximum_action_std_for_convergence=maximum_action_std_for_convergence,
                                feasible_candidates=feasible_count,
                                evaluator_failed=True,
                            )
                            selection_status = "evaluator_failure"
                            for pending in futures:
                                pending.cancel()
                            break
                        result_dict = dict(output["result"])
                        feasible, reasons = _canonical_feasibility(
                            result_dict,
                            tolerance=max_feature_error_threshold,
                        )
                        _assert_feasibility_consistency(
                            feasible=feasible,
                            result=result_dict,
                            tolerance=max_feature_error_threshold,
                            source=sample.sample_id,
                        )
                        completed = LatentSample(
                            sample_id=sample.sample_id,
                            round_index=round_index,
                            action=sample.action,
                            latent=sample.latent,
                            reward=float(result_dict["outer_reward"]),
                            feasible=feasible,
                            evaluation_consumed=True,
                            metadata={
                                "mean_observed_vad": result_dict.get("mean_observed_vad"),
                                "realisation_rmse": result_dict.get("realisation_rmse"),
                                "candidate_seed": int(payload["candidate_seed"]),
                                **_robustness_fields(
                                    feasible=feasible,
                                    result_dict=result_dict,
                                    tolerance=max_feature_error_threshold,
                                    robust_margin=robust_elite_max_feature_error,
                                    sample_index=sample_index,
                                ),
                            },
                        )
                        sample_records.append(
                            (
                                sample_index,
                                completed,
                                result_dict,
                                (
                                    dict(output["optimisation_summary"])
                                    if output.get("optimisation_summary") is not None
                                    else None
                                ),
                            )
                        )
                        checkpoint_payload = _candidate_record(
                            sample=completed,
                            round_index=round_index,
                            sample_index=sample_index,
                            candidate_seed=int(payload["candidate_seed"]),
                            result_dict=result_dict,
                            feasible=feasible,
                            ineligibility_reasons=reasons,
                            sample_output_dir=Path(str(payload["sample_output_dir"])),
                            optimisation_result=None,
                        )
                        checkpoint_payload["optimisation_summary"] = output.get("optimisation_summary")
                        _write_json(
                            _checkpoint_file(out_dir, round_index=round_index, sample_index=sample_index),
                            checkpoint_payload,
                        )
                        _write_json(checkpoint_state_path, checkpoint_state)
        if stop in ("evaluator_failure", "call_budget_exhausted"):
            break
        sample_records.sort(key=lambda item: item[0])
        robust_count = 0
        marginal_count = 0
        for sample_index, completed, result_dict, optimisation_summary in sample_records:
            reward = float(completed.reward)
            reasons = _canonical_feasibility(
                result_dict,
                tolerance=max_feature_error_threshold,
            )[1]
            metadata = completed.metadata or {}
            if completed.feasible:
                feasible_count += 1
                if bool(metadata.get("robustly_feasible", False)):
                    robust_count += 1
                else:
                    marginal_count += 1
            completed_samples.append(completed)
            round_rewards.append(reward)
            sample_row = {
                "round": round_index,
                "sample_index": sample_index,
                "sample_id": completed.sample_id,
                "feasible": completed.feasible,
                "ineligibility_reasons": "|".join(reasons),
                "reward": reward,
                "evaluation_consumed": True,
                "valid_realisation": bool(result_dict.get("valid_realisation", False)),
                "physically_acceptable": bool(result_dict.get("physically_acceptable", False)),
                "feature_realisation_acceptable": bool(result_dict.get("feature_realisation_acceptable", False)),
                "realisation_rmse": result_dict.get("realisation_rmse"),
                "mean_vad_reward": result_dict.get("mean_vad_reward"),
                "max_abs_feature_error": result_dict.get("max_abs_feature_error"),
                "synthetic_objective_distance": result_dict.get("synthetic_objective_distance"),
                "strictly_feasible": bool(metadata.get("strictly_feasible", completed.feasible)),
                "robustly_feasible": bool(metadata.get("robustly_feasible", False)),
                "distance_to_official_tolerance": metadata.get("distance_to_official_tolerance"),
                "distance_to_robust_margin": metadata.get("distance_to_robust_margin"),
                "ranking_category": metadata.get("ranking_category"),
                "candidate_seed": metadata.get("candidate_seed"),
            }
            for key in FEATURE_KEYS:
                sample_row[key] = completed.action[key]
            sample_history.append(sample_row)
            archived_records.append(
                {
                    "sample": completed,
                    "result": result_dict,
                    "optimisation_result": optimisation_summary,
                }
            )
            observed = result_dict.get("mean_observed_vad")
            if observed and (round_best_vad is None or reward >= max(round_rewards)):
                round_best_vad = dict(observed)
        if stop in ("evaluator_failure", "call_budget_exhausted"):
            break
        require_reward_consumption(completed_samples)
        elites = distribution.update(
            completed_samples,
            round_index=round_index,
            elite_count=elite_count,
        )
        diagnostics = distribution.last_update
        covariance_matrices.append(np.asarray(distribution.covariance, dtype=float).copy())
        for elite in elites:
            elite_metadata = elite.metadata or {}
            elite_history.append(
                {
                    "round": round_index,
                    "sample_id": elite.sample_id,
                    "reward": elite.reward,
                    "ranking_category": elite_metadata.get("ranking_category"),
                    "robustly_feasible": bool(elite_metadata.get("robustly_feasible", False)),
                    "max_abs_feature_error": elite_metadata.get("max_abs_feature_error"),
                }
            )
        mean_profile = distribution.mean_action()
        mean_row = {"round": round_index}
        for key, value in mean_profile.items():
            mean_row[key] = value
        mean_history.append(mean_row)
        corr_row = {
            "round": round_index,
            "log_determinant": diagnostics.log_determinant,
            "min_eigenvalue": diagnostics.min_eigenvalue,
            "max_eigenvalue": diagnostics.max_eigenvalue,
            "condition_number": diagnostics.condition_number,
            "covariance_source": diagnostics.covariance_source,
            "correction": diagnostics.correction,
        }
        for row_index, left in enumerate(FEATURE_KEYS):
            for column_index, right in enumerate(FEATURE_KEYS):
                corr_row[f"corr_{left}_{right}"] = diagnostics.correlation_matrix[row_index][column_index]
        correlation_history.append(corr_row)
        _write_json(
            _round_state_file(out_dir, round_index=round_index),
            {
                "schema_version": 1,
                "created_at_utc": _utc_timestamp(),
                "context": asdict(context),
                "round_index": round_index,
                "distribution_state": distribution.state_dict(),
                "mean_action": distribution.mean_action(),
                "covariance": distribution.state_dict()["covariance"],
            },
        )
        checkpoint_state["latest_round_completed"] = round_index
        _write_json(checkpoint_state_path, checkpoint_state)
        round_best = max(round_rewards) if round_rewards else float("-inf")
        improvement = round_best - best_reward_so_far
        if improvement >= minimum_reward_improvement:
            plateau_rounds = 0
            best_reward_so_far = round_best
        else:
            plateau_rounds += 1
        current_std = distribution.action_standard_deviation()
        if round_best_vad is not None:
            best_by_round.append(round_best_vad)
        round_history.append(
            {
                "round": round_index,
                "best_reward": round_best,
                "mean_reward": float(np.mean(round_rewards)) if round_rewards else float("nan"),
                "feasible_candidates": feasible_count,
                "robust_candidates": robust_count,
                "marginal_candidates": marginal_count,
                "elite_count": len(elites),
                "marginal_elites_used": int(
                    sum(
                        1
                        for elite in elites
                        if not bool((elite.metadata or {}).get("robustly_feasible", True))
                    )
                ),
                "insufficient_robust_candidates": bool(robust_count < len(elites)),
                "action_std": current_std,
            }
        )
        reward_history.append(
            {
                "round": round_index,
                "best_reward": round_best,
                "mean_reward": float(np.mean(round_rewards)) if round_rewards else float("nan"),
            }
        )
        if (
            not synthetic
            and
            round_index >= rounds_min
            and round_best >= baseline.baseline_vad_reward + 0.05
        ):
            stop = "perceptual_success"
        else:
            stop = stopping_reason(
                round_index=round_index,
                minimum_rounds=rounds_min,
                maximum_rounds=rounds_max,
                plateau_rounds=plateau_rounds,
                plateau_patience=plateau_patience,
                current_action_std=current_std,
                maximum_action_std_for_convergence=maximum_action_std_for_convergence,
                feasible_candidates=feasible_count,
            )
        if stop is not None:
            break

    validation_rows: list[dict[str, Any]] = []
    had_feasible_training = False
    if stop not in ("evaluator_failure", "call_budget_exhausted"):
        _budget = get_active_budget()
        if _budget is not None:
            _budget.set_category("validation")
        feasible_ranked = sorted(
            [row for row in archived_records if bool(row["sample"].feasible)],
            key=lambda row: robust_rank_key(row["sample"]),
        )
        had_feasible_training = bool(feasible_ranked)
        if not feasible_ranked:
            selection_status = "no_valid_candidate"
            stop = stop or "no_feasible_candidates"
        shortlist = feasible_ranked[: max(1, validation_top_k)]
        for rank, item in enumerate(shortlist, start=1):
            try:
                result_dict, optimisation_result = _evaluate_profile(
                    validation_environment,
                    target_context,
                    dict(item["sample"].action),
                    out_dir / "validation" / f"candidate_{rank:02d}",
                    synthetic=synthetic,
                    synthetic_invalid=synthetic_invalid,
                )
            except CallBudgetExhausted:
                stop = "call_budget_exhausted"
                selection_status = "call_budget_exhausted"
                break
            except Exception:
                stop = "evaluator_failure"
                selection_status = "evaluator_failure"
                break
            validation_feasible, reasons = _canonical_feasibility(
                result_dict,
                tolerance=max_feature_error_threshold,
            )
            _assert_feasibility_consistency(
                feasible=validation_feasible,
                result=result_dict,
                tolerance=max_feature_error_threshold,
                source=f"validation_{rank:02d}",
            )
            validation_rows.append(
                {
                    "rank": rank,
                    "sample_id": item["sample"].sample_id,
                    "training_reward": float(item["sample"].reward),
                    "validation_reward": float(result_dict["outer_reward"]),
                    "mean_observed_vad": dict(result_dict.get("mean_observed_vad") or {}),
                    "valid_realisation": bool(result_dict.get("valid_realisation", False)),
                    "physically_acceptable": bool(result_dict.get("physically_acceptable", False)),
                    "feature_realisation_acceptable": bool(result_dict.get("feature_realisation_acceptable", False)),
                    "realisation_rmse": result_dict.get("realisation_rmse"),
                    "max_abs_feature_error": result_dict.get("max_abs_feature_error"),
                    "synthetic_objective_distance": result_dict.get("synthetic_objective_distance"),
                    "feasible": validation_feasible,
                    "ineligibility_reasons": reasons,
                    "profile": dict(item["sample"].action),
                    "result": result_dict,
                    "training_seed": (item["sample"].metadata or {}).get("candidate_seed"),
                    "validation_seed": int(seed),
                    "training_max_abs_feature_error": (item["sample"].metadata or {}).get("max_abs_feature_error"),
                    "validation_max_abs_feature_error": result_dict.get("max_abs_feature_error"),
                    "training_realisation_rmse": (item["sample"].metadata or {}).get("realisation_rmse"),
                    "validation_realisation_rmse": result_dict.get("realisation_rmse"),
                    "training_ranking_category": (item["sample"].metadata or {}).get("ranking_category"),
                    "training_robustly_feasible": bool(
                        (item["sample"].metadata or {}).get("robustly_feasible", False)
                    ),
                    "feasibility_survived_seed_change": bool(validation_feasible),
                    "max_feature_error_change": (
                        float(result_dict["max_abs_feature_error"])
                        - float((item["sample"].metadata or {})["max_abs_feature_error"])
                        if result_dict.get("max_abs_feature_error") is not None
                        and (item["sample"].metadata or {}).get("max_abs_feature_error") is not None
                        else None
                    ),
                }
            )
            if not validation_feasible:
                continue
            if final_selected_result is None or float(result_dict["outer_reward"]) > float(final_selected_result["validation_reward"]):
                final_selected_result = validation_rows[-1]
                final_selected_opt_result = optimisation_result
        if selection_status == "pending":
            selection_status = "selected" if final_selected_result is not None else "no_valid_candidate"
    elif stop == "call_budget_exhausted":
        selection_status = "call_budget_exhausted"
    else:
        selection_status = "evaluator_failure"

    learned_vs_reference = {"status": "synthetic_not_run", "records": []}
    learned_vs_baseline = {"status": "synthetic_not_run", "records": []}
    if selection_status != "selected":
        learned_vs_reference = {"status": "skipped_no_valid_candidate", "records": []}
        learned_vs_baseline = {"status": "skipped_no_valid_candidate", "records": []}
    if selection_status == "selected" and not synthetic and final_selected_opt_result is not None:
        _budget = get_active_budget()
        if _budget is not None:
            _budget.set_category("paired")
        target_layers = {
            "original_affect_hypothesis": dict(baseline.original_affect_hypothesis),
            "projected_feasible_initialisation": (
                dict(baseline.projected_feasible_initialisation)
                if baseline.projected_feasible_initialisation is not None
                else None
            ),
        }
        reference_result = load_baseline_motion(
            context.gesture,
            context.target_state,
            candidate="reference",
        )
        baseline_result = load_baseline_motion(
            context.gesture,
            context.target_state,
            candidate="styled",
        )
        try:
            learned_vs_reference = _paired_validation(
                pair_id=f"{context.gesture}-{context.target_state}-learned-vs-reference",
                context=target_context,
                learned=final_selected_opt_result,
                other=reference_result,
                target_layers=target_layers,
                out_path=out_dir / "paired_learned_vs_reference.json",
                cache_root=_short_cache_root(out_dir, "pr"),
                repeats=paired_validation_repeats,
                evaluator_name=evaluator_name,
                model=model,
                temperature=temperature,
            )
        except CallBudgetExhausted:
            stop = "call_budget_exhausted"
            learned_vs_reference = {"status": "skipped_call_budget_exhausted", "records": []}
        try:
            if stop == "call_budget_exhausted":
                raise CallBudgetExhausted("budget already exhausted before baseline pairing")
            learned_vs_baseline = _paired_validation(
                pair_id=f"{context.gesture}-{context.target_state}-learned-vs-baseline",
                context=target_context,
                learned=final_selected_opt_result,
                other=baseline_result,
                target_layers=target_layers,
                out_path=out_dir / "paired_learned_vs_baseline.json",
                cache_root=_short_cache_root(out_dir, "pb"),
                repeats=paired_validation_repeats,
                evaluator_name=evaluator_name,
                model=model,
                temperature=temperature,
            )
        except CallBudgetExhausted:
            stop = "call_budget_exhausted"
            learned_vs_baseline = {"status": "skipped_call_budget_exhausted", "records": []}
        _render_final_assets(
            out_dir,
            final_selected_opt_result,
            gesture=context.gesture,
        )

    final_vad = dict(
        final_selected_result["mean_observed_vad"] if final_selected_result is not None
        else baseline.baseline_observed_vad
    )
    target_vad = dict(baseline.target_vad)
    selected_validation_reward = (
        float(final_selected_result["validation_reward"])
        if final_selected_result is not None
        else None
    )
    initial_synthetic_reward = (
        (1.0 - float(synthetic_initial_distance))
        if synthetic_initial_distance is not None
        else None
    )
    final_synthetic_distance = (
        float(final_selected_result["result"].get("synthetic_objective_distance"))
        if synthetic and final_selected_result is not None
        else None
    )
    final_synthetic_reward = (
        float(final_selected_result["validation_reward"])
        if synthetic and final_selected_result is not None
        else None
    )
    synthetic_distance_reduction = (
        float(synthetic_initial_distance) - float(final_synthetic_distance)
        if synthetic_initial_distance is not None and final_synthetic_distance is not None
        else None
    )
    synthetic_reward_improvement = (
        float(final_synthetic_reward) - float(initial_synthetic_reward)
        if initial_synthetic_reward is not None and final_synthetic_reward is not None
        else None
    )

    if evaluator_name == "gemini":
        comparison_status = "comparable"
        baseline_validation_reward = float(baseline.baseline_vad_reward)
        delta_vs_baseline = (
            selected_validation_reward - baseline_validation_reward
            if selected_validation_reward is not None
            else None
        )
    elif synthetic:
        comparison_status = "not_comparable_in_synthetic_mode"
        baseline_validation_reward = None
        delta_vs_baseline = None
    else:
        comparison_status = "not_comparable_without_gemini_evaluator"
        baseline_validation_reward = None
        delta_vs_baseline = None
    learned_wins = 0
    if learned_vs_baseline.get("records"):
        learned_wins = int(learned_vs_baseline["records"][0]["choice_counts"]["styled"])
    in_target_quadrant = (
        (final_vad["valence"] - 0.5) * (target_vad["valence"] - 0.5) >= 0.0
        and (final_vad["arousal"] - 0.5) * (target_vad["arousal"] - 0.5) >= 0.0
    )
    if stop == "call_budget_exhausted" or selection_status == "call_budget_exhausted":
        outcome = "call_budget_exhausted"
    elif stop == "evaluator_failure":
        outcome = "evaluator_failure"
    elif selection_status != "selected":
        # Distinguish "training never produced a feasible candidate" from
        # "feasible training candidates existed but none survived independent
        # validation". Both remain unsuccessful stage outcomes.
        outcome = (
            "no_validation_feasible_candidate"
            if had_feasible_training
            else "no_feasible_candidates"
        )
    elif synthetic:
        final_distance = float(final_selected_result["result"]["synthetic_objective_distance"])
        if final_distance <= 0.03:
            outcome = "synthetic_optimum_recovered"
        elif synthetic_initial_distance is not None and final_distance < synthetic_initial_distance:
            outcome = "synthetic_objective_improvement"
        else:
            outcome = "synthetic_converged_unsuccessfully"
    elif evaluator_name == "gemini":
        outcome = classify_outcome(
            valid_realisation=bool(final_selected_result["result"]["valid_realisation"]),
            validated_vad_improvement_over_baseline=float(delta_vs_baseline),
            learned_preference_wins_over_baseline=learned_wins,
            paired_repeats=paired_validation_repeats,
            in_target_quadrant=in_target_quadrant,
            stable_reward=bool(
                float(final_selected_result["validation_reward"]) >= baseline.baseline_vad_reward + 0.01
            ),
            converged=stop == "distribution_converged",
        )
    else:
        outcome = "mock_validation_complete"

    profile_comparison = {
        "reference_profile": load_baseline_motion(
            context.gesture,
            context.target_state,
            candidate="reference",
        ).requested_profile if not synthetic else None,
        "original_affect_hypothesis": dict(baseline.original_affect_hypothesis),
        "projected_profile": (
            dict(baseline.projected_feasible_initialisation)
            if baseline.projected_feasible_initialisation is not None
            else None
        ),
        "initial_cem_mean": dict(initial_profile),
        "final_cem_mean": distribution.mean_action(),
        "selected_learned_profile": (
            dict(final_selected_result["profile"])
            if final_selected_result is not None
            else None
        ),
        "achieved_learned_profile": (
            dict(final_selected_result["result"]["achieved_profile"])
            if final_selected_result is not None and "achieved_profile" in final_selected_result["result"]
            else None
        ),
        "initial_observed_vad": dict(baseline.baseline_observed_vad),
        "learned_observed_vad": dict(final_vad),
        "canonical_target_vad": dict(target_vad),
        "projection_error": baseline.projection_error,
        "realisation_error": (
            final_selected_result["result"].get("realisation_rmse")
            if final_selected_result is not None
            else None
        ),
        "vad_improvement_over_reference": (
            float(final_selected_result["validation_reward"]) - float(
                weighted_vad_reward(baseline.baseline_observed_vad, target_vad)
            )
        ) if final_selected_result is not None else None,
        "vad_improvement_over_fixed_baseline": delta_vs_baseline,
        "comparison_status": comparison_status,
        "initial_synthetic_objective_distance": synthetic_initial_distance,
        "final_synthetic_objective_distance": final_synthetic_distance,
        "synthetic_distance_reduction": synthetic_distance_reduction,
        "initial_synthetic_reward": initial_synthetic_reward,
        "final_synthetic_reward": final_synthetic_reward,
        "synthetic_reward_improvement": synthetic_reward_improvement,
        "selection_status": selection_status,
        "paired_preference_results": {
            "learned_vs_reference": learned_vs_reference,
            "learned_vs_baseline": learned_vs_baseline,
        },
        "outcome": outcome,
    }
    if selection_status == "selected" and learned_vs_baseline.get("status") not in {"synthetic_not_run", "complete"}:
        raise RuntimeError("Baseline comparison is mandatory for final outcomes.")

    _write_json(
        out_dir / "run_manifest.json",
        {
            "experiment_name": "gesture_conditioned_outer_learning_v1",
            "baseline_experiment_name": BASELINE_EXPERIMENT_NAME,
            "context": asdict(context),
            "evaluator": evaluator_name,
            "model": model if evaluator_name == "gemini" else None,
            "temperature": temperature,
            "rounds_min": rounds_min,
            "rounds_max": rounds_max,
            "samples_per_round": samples_per_round,
            "elite_count": elite_count,
            "candidate_vlm_repeats": candidate_vlm_repeats,
            "validation_top_k": validation_top_k,
            "validation_vlm_repeats": validation_vlm_repeats,
            "paired_validation_repeats": paired_validation_repeats,
            "seed": seed,
            "covariance_shrinkage": covariance_shrinkage,
            "covariance_smoothing": covariance_smoothing,
            "mean_smoothing": mean_smoothing,
            "minimum_eigenvalue": minimum_eigenvalue,
            "maximum_eigenvalue": maximum_eigenvalue,
            "covariance_history_rounds": covariance_history_rounds,
            "round_weight_decay": round_weight_decay,
            "realisation_penalty_weight": realisation_penalty_weight,
            "max_feature_error_threshold": max_feature_error_threshold,
            "robust_elite_max_feature_error": robust_elite_max_feature_error,
            "baseline_directory": str(BASELINE_ROOT),
            "selection_status": selection_status,
            "comparison_status": comparison_status,
            "initial_synthetic_objective_distance": synthetic_initial_distance,
            "final_synthetic_objective_distance": final_synthetic_distance,
            "synthetic_distance_reduction": synthetic_distance_reduction,
            "initial_synthetic_reward": initial_synthetic_reward,
            "final_synthetic_reward": final_synthetic_reward,
            "synthetic_reward_improvement": synthetic_reward_improvement,
        },
    )
    _write_json(
        out_dir / "stopping_reason.json",
        {
            "stopping_reason": stop,
            "outcome": outcome,
            "selection_status": selection_status,
            "rounds_completed": len(round_history),
            "comparison_status": comparison_status,
        },
    )
    _write_json(
        out_dir / "initial_profile.json",
        {
            "original_affect_hypothesis": dict(baseline.original_affect_hypothesis),
            "projected_feasible_initialisation": (
                dict(baseline.projected_feasible_initialisation)
                if baseline.projected_feasible_initialisation is not None
                else None
            ),
            "initial_cem_mean": dict(initial_profile),
            "initial_cem_covariance": initial_covariance_snapshot.tolist(),
        },
    )
    _write_json(
        out_dir / "learned_profile.json",
        {
            "learned_cem_mean": distribution.mean_action(),
            "learned_cem_covariance": distribution.state_dict()["covariance"],
        },
    )
    _write_json(out_dir / "independent_validation.json", {"candidates": validation_rows})
    if selection_status == "selected" and final_selected_result is not None:
        _write_json(out_dir / "selected_validated_profile.json", final_selected_result)
    _write_json(out_dir / "profile_comparison.json", profile_comparison)
    _write_csv(out_dir / "round_history.csv", round_history)
    _write_csv(out_dir / "sample_history.csv", sample_history)
    _write_csv(out_dir / "elite_history.csv", elite_history)
    _write_csv(out_dir / "mean_history.csv", mean_history)
    _write_csv(out_dir / "correlation_history.csv", correlation_history)
    _write_csv(out_dir / "reward_history.csv", reward_history)
    _save_covariance_history(out_dir / "covariance_history.npz", covariance_matrices)
    _plot_reward_history(out_dir / "reward_convergence.png", reward_history)
    _plot_covariance_history(
        out_dir / "covariance_evolution.png",
        [np.diag(matrix) for matrix in covariance_matrices],
    )
    _plot_vad_trajectory(
        out_dir / "vad_trajectory.png",
        target_vad=target_vad,
        baseline_vad=baseline.baseline_observed_vad,
        round_best=best_by_round,
        final_vad=final_vad,
    )

    for environment in (validation_environment,):
        if environment is not None:
            close = getattr(environment.evaluator, "close", None)
            if callable(close):
                close()
    checkpoint_state["completed_at_utc"] = _utc_timestamp()
    checkpoint_state["final_outcome"] = outcome
    checkpoint_state["final_selection_status"] = selection_status
    _write_json(checkpoint_state_path, checkpoint_state)
    return {
        "context": asdict(context),
        "stopping_reason": stop,
        "outcome": outcome,
        "selection_status": selection_status,
        "selected_validation_reward": selected_validation_reward,
        "baseline_validation_reward": baseline_validation_reward,
        "improvement_over_baseline": delta_vs_baseline,
        "comparison_status": comparison_status,
        "initial_synthetic_objective_distance": synthetic_initial_distance,
        "final_synthetic_objective_distance": final_synthetic_distance,
        "synthetic_distance_reduction": synthetic_distance_reduction,
        "initial_synthetic_reward": initial_synthetic_reward,
        "final_synthetic_reward": final_synthetic_reward,
        "synthetic_reward_improvement": synthetic_reward_improvement,
        "resume_used": bool(resume),
        "resume_events": int(checkpoint_state.get("resume_events", 0)),
        "recovered_candidates": int(checkpoint_state.get("recovered_candidates", 0)),
        "recomputed_candidates": int(checkpoint_state.get("recomputed_candidates", 0)),
        "worker_failures": int(checkpoint_state.get("worker_failures", 0)),
        "all_outputs": sorted(
            str(path.relative_to(out_dir))
            for path in out_dir.rglob("*")
            if path.is_file()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume and args.overwrite:
        raise RuntimeError("Cannot combine --resume with --overwrite.")
    if args.workers < 1:
        raise RuntimeError("--workers must be at least 1.")
    config_path = ROOT / args.config
    config = _load_config(config_path)
    stage_payload = _stage_payload(config, args.stage)
    base_out = ROOT / args.out
    guard_baseline_output_path(base_out, overwrite_baseline=args.overwrite_baseline)
    if args.stage == "stage_b":
        _require_stage_a_success(base_out)
    contexts = _contexts_from_stage(stage_payload)
    if args.overwrite and _stage_dir(base_out, args.stage).exists():
        shutil.rmtree(_stage_dir(base_out, args.stage))
    stage_out = _stage_dir(base_out, args.stage)
    stage_out.mkdir(parents=True, exist_ok=True)
    defaults = dict(config["defaults"])
    rounds_min = int(args.rounds_min or defaults["rounds_min"])
    rounds_max = int(args.rounds_max or defaults["rounds_max"])
    samples_per_round = int(args.samples_per_round or defaults["samples_per_round"])
    elite_count = int(args.elite_count or defaults["elite_count"])
    candidate_vlm_repeats = int(args.candidate_vlm_repeats or defaults["candidate_vlm_repeats"])
    validation_top_k = int(args.validation_top_k or defaults["validation_top_k"])
    validation_vlm_repeats = int(args.validation_vlm_repeats or defaults["validation_vlm_repeats"])
    paired_validation_repeats = int(args.paired_validation_repeats or defaults["paired_validation_repeats"])
    robust_elite_max_feature_error = float(
        args.robust_elite_max_feature_error
        if args.robust_elite_max_feature_error is not None
        else defaults.get("robust_elite_max_feature_error", 0.08)
    )
    live_estimate = _estimate_live_calls(
        stage_payload,
        rounds_max=rounds_max,
        samples_per_round=samples_per_round,
        candidate_vlm_repeats=candidate_vlm_repeats,
        validation_top_k=validation_top_k,
        validation_vlm_repeats=validation_vlm_repeats,
        paired_validation_repeats=paired_validation_repeats,
    )
    call_budget: GeminiCallBudget | None = None
    if args.evaluator == "gemini":
        if args.max_gemini_calls < 1:
            raise RuntimeError("--max-gemini-calls must be at least 1.")
        if args.workers != 1:
            raise RuntimeError(
                "The Gemini call budget requires --workers 1 so every call is "
                "counted in a single process."
            )
        if int(live_estimate["estimated_total_calls"]) > int(args.max_gemini_calls):
            raise RuntimeError(
                f"Estimated maximum Gemini calls "
                f"({live_estimate['estimated_total_calls']}) exceed the hard "
                f"ceiling of {args.max_gemini_calls}. Reduce the experimental "
                "settings or raise --max-gemini-calls explicitly."
            )
        call_budget = GeminiCallBudget(
            args.max_gemini_calls,
            stage_out / "gemini_call_budget.json",
        )
        set_active_budget(call_budget)
    stage_status = []
    for context in contexts:
        result = run_context(
            context=context,
            out_dir=_context_dir(stage_out, context),
            evaluator_name=args.evaluator,
            model=args.model,
            temperature=args.temperature,
            rounds_min=rounds_min,
            rounds_max=rounds_max,
            samples_per_round=samples_per_round,
            elite_count=elite_count,
            candidate_vlm_repeats=candidate_vlm_repeats,
            validation_top_k=validation_top_k,
            validation_vlm_repeats=validation_vlm_repeats,
            paired_validation_repeats=paired_validation_repeats,
            covariance_shrinkage=args.covariance_shrinkage,
            covariance_smoothing=args.covariance_smoothing,
            mean_smoothing=args.mean_smoothing,
            minimum_eigenvalue=args.minimum_eigenvalue,
            maximum_eigenvalue=args.maximum_eigenvalue,
            covariance_history_rounds=args.covariance_history_rounds,
            round_weight_decay=args.round_weight_decay,
            plateau_patience=args.plateau_patience,
            minimum_reward_improvement=args.minimum_reward_improvement,
            maximum_action_std_for_convergence=args.maximum_action_std_for_convergence,
            realisation_penalty_weight=args.realisation_penalty_weight,
            max_feature_error_threshold=args.max_feature_error_threshold,
            robust_elite_max_feature_error=robust_elite_max_feature_error,
            seed=args.seed,
            resume=args.resume,
            workers=args.workers,
        )
        stage_status.append(result)
    _write_json(
        stage_out / "stage_status.json",
        {
            "stage": args.stage,
            "all_successful": all(
                _successful_outcome(item["outcome"]) for item in stage_status
            ),
            "saved_run_settings": {
                "evaluator": args.evaluator,
                "seed": args.seed,
                "rounds_min": rounds_min,
                "rounds_max": rounds_max,
                "samples_per_round": samples_per_round,
                "elite_count": elite_count,
                "candidate_vlm_repeats": candidate_vlm_repeats,
                "validation_top_k": validation_top_k,
                "validation_vlm_repeats": validation_vlm_repeats,
                "paired_validation_repeats": paired_validation_repeats,
                "max_feature_error_threshold": args.max_feature_error_threshold,
                "robust_elite_max_feature_error": robust_elite_max_feature_error,
                "workers": args.workers,
                "resume": bool(args.resume),
                "max_gemini_calls": (
                    int(args.max_gemini_calls) if args.evaluator == "gemini" else None
                ),
            },
            "estimated_live_calls": live_estimate,
            "gemini_call_budget": (
                call_budget.summary() if call_budget is not None else None
            ),
            "resume_summary": {
                "resume_used": bool(args.resume),
                "contexts_with_resume_events": int(
                    sum(1 for item in stage_status if int(item.get("resume_events", 0)) > 0)
                ),
                "total_resume_events": int(sum(int(item.get("resume_events", 0)) for item in stage_status)),
                "total_recovered_candidates": int(sum(int(item.get("recovered_candidates", 0)) for item in stage_status)),
                "total_recomputed_candidates": int(sum(int(item.get("recomputed_candidates", 0)) for item in stage_status)),
                "total_worker_failures": int(sum(int(item.get("worker_failures", 0)) for item in stage_status)),
            },
            "contexts": stage_status,
        },
    )
    print(stage_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
