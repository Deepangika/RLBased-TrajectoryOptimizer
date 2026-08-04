"""Fixed-profile baseline preservation and loading helpers."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from laban_rl.affect import VAD_KEYS
from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import LabanOptimisationResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "baseline_fixed_profiles"
BASELINE_RESULTS_DIR = BASELINE_ROOT / "full_gemini_live"
BASELINE_MANIFEST_PATH = BASELINE_ROOT / "baseline_manifest.json"
BASELINE_REPORT_PATH = BASELINE_RESULTS_DIR / "full_experiment_report.json"
BASELINE_MATRIX_PATH = PROJECT_ROOT / "configs" / "full_gemini_matrix.json"
BASELINE_EXPERIMENT_NAME = "fixed_profile_baseline_v1"
BASELINE_PROTECTED_SENTINEL = BASELINE_ROOT.resolve()


def is_protected_baseline_path(path: str | Path) -> bool:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate == BASELINE_PROTECTED_SENTINEL or BASELINE_PROTECTED_SENTINEL in candidate.parents


def guard_baseline_output_path(
    path: str | Path,
    *,
    overwrite_baseline: bool = False,
    purpose: str = "output",
) -> None:
    if is_protected_baseline_path(path) and not overwrite_baseline:
        raise RuntimeError(
            f"Refusing to use protected baseline {purpose} path {Path(path)!s}. "
            "Pass --overwrite-baseline only for deliberate developer maintenance."
        )


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-5, 1.0 - 1e-5)
    return np.log(clipped) - np.log1p(-clipped)


def _condition_key(gesture: str, target_state: str) -> str:
    return f"{gesture}::{target_state}"


@dataclass(frozen=True)
class BaselineCondition:
    condition_id: str
    gesture: str
    target_state: str
    target_vad: dict[str, float]
    original_affect_hypothesis: dict[str, float]
    projected_feasible_initialisation: dict[str, float] | None
    requested_optimization: dict[str, float]
    achieved_profile: dict[str, float]
    baseline_observed_vad: dict[str, float]
    baseline_vad_reward: float
    styled_preference_rate: float
    reference_preference_rate: float
    neither_rate: float
    projection_error: float | None
    realisation_error: float
    baseline_candidate_id: str
    reference_candidate_id: str


def _candidate_ids(condition_id: str) -> tuple[str, str]:
    return (
        f"{condition_id}__styled",
        f"{condition_id}__reference",
    )


def load_baseline_report(
    path: Path = BASELINE_REPORT_PATH,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline report was not found at {path}. Preserve the baseline first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_conditions(
    path: Path = BASELINE_REPORT_PATH,
) -> dict[str, BaselineCondition]:
    payload = load_baseline_report(path)
    conditions: dict[str, BaselineCondition] = {}
    for row in payload["conditions"]:
        gesture = str(row["gesture"])
        target_state = str(row["target_state"])
        condition_id = str(row["condition_id"])
        styled_candidate_id, reference_candidate_id = _candidate_ids(condition_id)
        conditions[_condition_key(gesture, target_state)] = BaselineCondition(
            condition_id=condition_id,
            gesture=gesture,
            target_state=target_state,
            target_vad={
                axis: float(row["target_vad"][axis])
                for axis in VAD_KEYS
            },
            original_affect_hypothesis={
                key: float(row["profiles"]["original_affect_derived"][key])
                for key in FEATURE_KEYS
            },
            projected_feasible_initialisation=(
                {
                    key: float(row["profiles"]["projected_feasible"][key])
                    for key in FEATURE_KEYS
                }
                if isinstance(row["profiles"].get("projected_feasible"), Mapping)
                else None
            ),
            requested_optimization={
                key: float(row["profiles"]["requested_optimization"][key])
                for key in FEATURE_KEYS
            },
            achieved_profile={
                key: float(row["profiles"]["achieved"][key])
                for key in FEATURE_KEYS
            },
            baseline_observed_vad={
                axis: float(row["vad"]["styled_mean"][axis])
                for axis in VAD_KEYS
            },
            baseline_vad_reward=float(row["vad"]["styled_mean_reward"]),
            styled_preference_rate=float(
                row["paired_preference"]["styled_preference_rate"]
            ),
            reference_preference_rate=float(
                row["paired_preference"]["reference_preference_rate"]
            ),
            neither_rate=float(row["paired_preference"]["neither_rate"]),
            projection_error=(
                None
                if row["realisation"]["projection_error_e_proj"] is None
                else float(row["realisation"]["projection_error_e_proj"])
            ),
            realisation_error=float(row["realisation"]["realisation_error_e_real"]),
            baseline_candidate_id=styled_candidate_id,
            reference_candidate_id=reference_candidate_id,
        )
    return conditions


def baseline_condition(gesture: str, target_state: str) -> BaselineCondition:
    try:
        return load_baseline_conditions()[_condition_key(gesture, target_state)]
    except KeyError as exc:
        raise KeyError(
            f"No preserved baseline found for {(gesture, target_state)!r}."
        ) from exc


def baseline_optimizer_overrides(
    gesture: str,
    target_state: str,
    *,
    matrix_path: Path = BASELINE_MATRIX_PATH,
) -> dict[str, Any]:
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        if case["gesture"] == gesture and case["target"]["state"] == target_state:
            return dict(
                case.get("candidate_profiles", {})
                .get("styled", {})
                .get("optimizer_overrides", {})
            )
    raise KeyError(
        f"No baseline matrix entry found for {(gesture, target_state)!r}."
    )


def load_motion_snapshot(
    snapshot_dir: str | Path,
    *,
    gesture: str,
    target_state: str,
) -> LabanOptimisationResult:
    snapshot_dir = Path(snapshot_dir)
    arrays_path = snapshot_dir / "paired_motion_snapshot.npz"
    metadata_path = snapshot_dir / "paired_motion_snapshot.json"
    if not arrays_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing baseline snapshot in {snapshot_dir}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path) as arrays:
        q_ref = np.asarray(arrays["q_ref"], dtype=float)
        q_var = np.asarray(arrays["q_var"], dtype=float)
        action = np.asarray(arrays["action"], dtype=float)
    return LabanOptimisationResult(
        gesture=gesture,
        target_state=target_state,
        requested_profile={
            key: float(metadata["requested_profile"][key]) for key in FEATURE_KEYS
        },
        achieved_profile={
            key: float(metadata["achieved_profile"][key]) for key in FEATURE_KEYS
        },
        achieved_profile_clipped={
            key: float(metadata["achieved_profile_clipped"][key])
            for key in FEATURE_KEYS
        },
        inner_reward=float(metadata["inner_reward"]),
        inner_loss=float(metadata["inner_loss"]),
        action_coefficients=action,
        q_ref=q_ref,
        q_var=q_var,
        output_dir=snapshot_dir,
        raw_result={
            "reward_info": dict(metadata.get("reward_info", {})),
            "snapshot_identity": metadata.get("case_identity"),
        },
    )


def load_baseline_motion(
    gesture: str,
    target_state: str,
    *,
    candidate: str,
) -> LabanOptimisationResult:
    condition = baseline_condition(gesture, target_state)
    if candidate == "styled":
        candidate_id = condition.baseline_candidate_id
    elif candidate == "reference":
        candidate_id = condition.reference_candidate_id
    else:
        raise ValueError("candidate must be 'styled' or 'reference'.")
    return load_motion_snapshot(
        BASELINE_RESULTS_DIR / "candidates" / candidate_id,
        gesture=gesture,
        target_state=target_state,
    )


def estimate_initial_covariance_from_baseline(
    gesture: str,
    *,
    diagonal_variance: float = 0.10,
    min_eigenvalue: float = 1e-3,
) -> np.ndarray:
    conditions = [
        condition
        for condition in load_baseline_conditions().values()
        if condition.gesture == gesture
    ]
    if len(conditions) < 2:
        return np.eye(len(FEATURE_KEYS), dtype=float) * diagonal_variance
    profiles = np.asarray(
        [
            _logit(
                np.asarray(
                    [
                        (
                            condition.projected_feasible_initialisation or
                            condition.requested_optimization
                        )[key]
                        for key in FEATURE_KEYS
                    ],
                    dtype=float,
                )
            )
            for condition in conditions
        ],
        dtype=float,
    )
    covariance = np.cov(profiles, rowvar=False, ddof=1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floored = np.maximum(eigenvalues, float(min_eigenvalue))
    return eigenvectors @ np.diag(floored) @ eigenvectors.T


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
