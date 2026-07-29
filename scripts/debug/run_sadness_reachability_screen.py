#!/usr/bin/env python3
"""Low-cost Point-Sadness expressive reachability diagnostic.

Repository destination:
    scripts/debug/run_sadness_reachability_screen.py

The script deliberately does *not* change CEM or the production inner
optimiser. It:

1. realises the current and an exaggerated five-feature Sadness profile using
   the existing Tuned-B inner optimiser;
2. derives bounded endpoint-down, contraction, and retreat variants from the
   successfully realised current profile;
3. checks kinematic validity and recomputes the achieved Laban features;
4. renders one MP4 and GIF per variant;
5. optionally evaluates every feasible frozen video repeatedly with Gemini;
6. writes CSV/JSON summaries and comparison plots.

The derived variants are diagnostic probes, not inner-optimiser solutions.
Their purpose is to establish whether an omitted motion cue makes sadness
perceptually reachable before expanding the CEM action space.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import robust_laban_normalisation_balanced_3gestures as laban

from laban_rl.config import FEATURE_KEYS, JointLimits
from laban_rl.features import compute_raw_and_norm_features
from laban_rl.io_utils import load_ranges_or_default
from laban_rl.optimiser_api import optimise_laban_target
from laban_rl.perceptual_bandit.environment import Context
from laban_rl.perceptual_bandit.gemini_evaluator import GeminiProVideoEvaluator
from laban_rl.perceptual_bandit.variant_video import render_variant_only_mp4
from laban_rl.rewards import compute_joint_limit_penalty


CURRENT_SADNESS_PROFILE = {
    "weight": 0.12,
    "time": 0.15,
    "flow_boundness": 0.35,
    "space_indirectness": 0.25,
    "shape_arcness": 0.25,
}

# Intentionally stronger than the informed profile. This is a reachability
# probe, not a claim about an empirically correct Laban encoding of sadness.
EXAGGERATED_SADNESS_PROFILE = {
    "weight": 0.03,
    "time": 0.05,
    "flow_boundness": 0.80,
    "space_indirectness": 0.15,
    "shape_arcness": 0.10,
}

STATE_LABELS = ("anger", "disgust", "fear", "happiness", "sadness", "surprise")


@dataclass(frozen=True)
class VariantSpec:
    name: str
    cue_type: str
    q_var: np.ndarray
    requested_profile: Mapping[str, float] | None
    source: str


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _forward_kinematics(q: np.ndarray, l1: float, l2: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    shoulder = q[:, 0]
    elbow = q[:, 1]
    return np.column_stack(
        (
            l1 * np.cos(shoulder) + l2 * np.cos(shoulder + elbow),
            l1 * np.sin(shoulder) + l2 * np.sin(shoulder + elbow),
        )
    )


def _continuous_two_link_ik(
    wrist: np.ndarray,
    q_anchor: np.ndarray,
    *,
    l1: float,
    l2: float,
) -> tuple[np.ndarray, float]:
    """Solve planar IK while following the elbow branch nearest q_anchor."""
    wrist = np.asarray(wrist, dtype=float)
    q_anchor = np.asarray(q_anchor, dtype=float)
    q_out = np.empty_like(q_anchor)
    projected = wrist.copy()
    projection_max = 0.0

    r_min = abs(l1 - l2) + 1e-6
    r_max = l1 + l2 - 1e-6

    for index, (x_raw, y_raw) in enumerate(wrist):
        radius = float(math.hypot(x_raw, y_raw))
        if radius < 1e-12:
            direction = np.array([1.0, 0.0])
        else:
            direction = np.array([x_raw, y_raw]) / radius

        clipped_radius = float(np.clip(radius, r_min, r_max))
        projected[index] = direction * clipped_radius
        projection_max = max(projection_max, abs(clipped_radius - radius))
        x, y = projected[index]

        cos_q2 = np.clip(
            (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2),
            -1.0,
            1.0,
        )
        elbow_magnitude = float(np.arccos(cos_q2))
        candidates = []
        for q2 in (elbow_magnitude, -elbow_magnitude):
            q1 = float(
                np.arctan2(y, x)
                - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
            )
            candidate = np.array([q1, q2], dtype=float)
            reference = q_anchor[index] if index == 0 else q_out[index - 1]
            candidate += 2.0 * np.pi * np.round((reference - candidate) / (2.0 * np.pi))
            candidates.append(candidate)

        reference = q_anchor[index] if index == 0 else q_out[index - 1]
        q_out[index] = min(
            candidates,
            key=lambda candidate: float(np.linalg.norm(candidate - reference)),
        )

    return q_out, projection_max


def _apply_downward_endpoint(
    q_base: np.ndarray,
    *,
    l1: float,
    l2: float,
    displacement_m: float,
) -> tuple[np.ndarray, dict[str, float]]:
    wrist = _forward_kinematics(q_base, l1, l2)
    u = np.linspace(0.0, 1.0, len(wrist))
    modified = wrist.copy()
    modified[:, 1] -= float(displacement_m) * _smoothstep01(u)
    q_new, projection = _continuous_two_link_ik(modified, q_base, l1=l1, l2=l2)
    return q_new, {
        "requested_downward_endpoint_m": float(displacement_m),
        "ik_workspace_projection_max_m": projection,
    }


def _apply_contraction(
    q_base: np.ndarray,
    *,
    l1: float,
    l2: float,
    final_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    wrist = _forward_kinematics(q_base, l1, l2)
    u = np.linspace(0.0, 1.0, len(wrist))
    scale = 1.0 + (float(final_scale) - 1.0) * _smoothstep01(u)
    modified = wrist[0] + scale[:, None] * (wrist - wrist[0])
    q_new, projection = _continuous_two_link_ik(modified, q_base, l1=l1, l2=l2)
    return q_new, {
        "requested_final_amplitude_scale": float(final_scale),
        "ik_workspace_projection_max_m": projection,
    }


def _apply_terminal_retreat(
    q_base: np.ndarray,
    *,
    l1: float,
    l2: float,
    retreat_m: float,
    onset_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    wrist = _forward_kinematics(q_base, l1, l2)
    u = np.linspace(0.0, 1.0, len(wrist))
    onset_index = min(len(wrist) - 2, max(1, int(round(onset_fraction * (len(wrist) - 1)))))
    outward = wrist[onset_index] - wrist[0]
    norm = float(np.linalg.norm(outward))
    if norm < 1e-9:
        outward = wrist[-1] - wrist[0]
        norm = float(np.linalg.norm(outward))
    if norm < 1e-9:
        raise RuntimeError("Cannot define retreat direction for a zero-extent path.")
    outward /= norm

    phase = _smoothstep01((u - float(onset_fraction)) / (1.0 - float(onset_fraction)))
    modified = wrist - float(retreat_m) * phase[:, None] * outward[None, :]
    q_new, projection = _continuous_two_link_ik(modified, q_base, l1=l1, l2=l2)
    return q_new, {
        "requested_retreat_m": float(retreat_m),
        "retreat_onset_fraction": float(onset_fraction),
        "ik_workspace_projection_max_m": projection,
    }


def _path_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _kinematic_diagnostics(
    q_ref: np.ndarray,
    q_var: np.ndarray,
    *,
    arm: Any,
    ranges: Mapping[str, tuple[float, float]],
    requested_profile: Mapping[str, float] | None,
) -> dict[str, Any]:
    filter_config = laban.FilterConfig(enabled=True, cutoff_hz=5.0, order=4)
    _, achieved_clipped = compute_raw_and_norm_features(
        q_var, arm, filter_config, ranges
    )
    # Recompute unclipped values so invalid overshoot is visible.
    raw, _ = compute_raw_and_norm_features(q_var, arm, filter_config, ranges)
    achieved = laban.normalise_laban_features(
        features=raw,
        normalisation_ranges=ranges,
        clip=False,
    )

    ref_path = _forward_kinematics(q_ref, arm.l1, arm.l2)
    var_path = _forward_kinematics(q_var, arm.l1, arm.l2)
    ref_length = _path_length(ref_path)
    var_length = _path_length(var_path)
    path_ratio = var_length / max(ref_length, 1e-12)
    endpoint_error = float(np.linalg.norm(var_path[-1] - ref_path[-1]))
    max_joint_delta = float(np.max(np.abs(q_var - q_ref)))
    joint_limit_error = float(
        compute_joint_limit_penalty(q_var, JointLimits())
    )

    finite = bool(
        np.all(np.isfinite(q_var))
        and all(np.isfinite(float(achieved[key])) for key in FEATURE_KEYS)
    )
    path_preserved = bool(0.70 <= path_ratio <= 1.30)
    joint_limits_satisfied = bool(joint_limit_error <= 1e-6)

    per_feature_error: dict[str, float] = {}
    rmse = None
    max_feature_error = None
    strict_feature_realisable = None
    if requested_profile is not None:
        per_feature_error = {
            key: abs(float(achieved[key]) - float(requested_profile[key]))
            for key in FEATURE_KEYS
        }
        rmse = float(np.sqrt(np.mean(np.square(list(per_feature_error.values())))))
        max_feature_error = float(max(per_feature_error.values()))
        strict_feature_realisable = bool(max_feature_error <= 0.10)

    return {
        "finite": finite,
        "path_preserved": path_preserved,
        "joint_limits_satisfied": joint_limits_satisfied,
        "physically_acceptable": bool(finite and path_preserved and joint_limits_satisfied),
        "reference_path_length_m": ref_length,
        "variant_path_length_m": var_length,
        "path_length_ratio": path_ratio,
        "endpoint_error_m": endpoint_error,
        "max_joint_delta_rad": max_joint_delta,
        "joint_limit_error": joint_limit_error,
        "achieved_profile": {key: float(achieved[key]) for key in FEATURE_KEYS},
        "achieved_profile_clipped": {
            key: float(achieved_clipped[key]) for key in FEATURE_KEYS
        },
        "per_feature_abs_error": per_feature_error,
        "feature_rmse": rmse,
        "max_abs_feature_error": max_feature_error,
        "strict_feature_realisable": strict_feature_realisable,
    }


def _save_gif_from_mp4(mp4_path: Path, gif_path: Path) -> None:
    reader = imageio.get_reader(mp4_path)
    try:
        frames = [frame for frame in reader]
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps", 20.0))
    finally:
        reader.close()
    imageio.mimsave(gif_path, frames, duration=1.0 / fps, loop=0)


def _evaluate_frozen_video(
    *,
    evaluator: GeminiProVideoEvaluator,
    context: Context,
    q_ref: np.ndarray,
    q_var: np.ndarray,
    output_dir: Path,
    repeats: int,
) -> dict[str, Any]:
    proxy = SimpleNamespace(
        q_ref=np.asarray(q_ref, dtype=float),
        q_var=np.asarray(q_var, dtype=float),
        output_dir=output_dir,
    )
    records: list[dict[str, float]] = []
    reasoning: list[str | None] = []
    for repeat_index in range(repeats):
        evaluation = evaluator.evaluate(context, proxy)
        probabilities = dict(evaluation.probabilities)
        records.append(probabilities)
        assessment = evaluator.last_assessment
        reasoning.append(
            None if assessment is None else assessment.reasoning_summary
        )
        print(
            f"  repeat {repeat_index + 1}/{repeats}: "
            f"winner={max(probabilities, key=probabilities.get)} "
            f"P(sadness)={probabilities['sadness']:.3f}"
        )

    means = {
        label: float(np.mean([record[label] for record in records]))
        for label in STATE_LABELS
    }
    winners = [max(record, key=record.get) for record in records]
    margins = [
        record["sadness"]
        - max(value for label, value in record.items() if label != "sadness")
        for record in records
    ]
    sadness_rewards = [
        0.5 * record["sadness"] + 0.5 * margin
        for record, margin in zip(records, margins)
    ]
    return {
        "completed_repeats": len(records),
        "raw_probabilities": records,
        "reasoning_summaries": reasoning,
        "mean_probabilities": means,
        "mean_target_probability": means["sadness"],
        "mean_margin": float(np.mean(margins)),
        "target_classification_rate": float(
            np.mean([winner == "sadness" for winner in winners])
        ),
        "winner_counts": {
            label: winners.count(label) for label in sorted(set(winners))
        },
        "perceptual_reward_std": float(np.std(sadness_rewards)),
    }


def _plot_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    names = [row["variant"] for row in rows]
    sadness = [
        np.nan
        if row.get("mean_sadness_probability") is None
        else row["mean_sadness_probability"]
        for row in rows
    ]
    classification = [
        np.nan
        if row.get("sadness_classification_rate") is None
        else row["sadness_classification_rate"]
        for row in rows
    ]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, sadness, width, label="Mean Sadness probability")
    ax.bar(x + width / 2, classification, width, label="Sadness classification rate")
    ax.axhline(0.30, color="C3", linestyle="--", linewidth=1.5, label="Diagnostic P threshold")
    ax.axhline(0.40, color="C4", linestyle=":", linewidth=1.5, label="Diagnostic classification threshold")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Proportion")
    ax.set_title("Point–Sadness expressive reachability screen")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sadness_reachability_comparison.png", dpi=200)
    plt.close(fig)

    feature_matrix = np.asarray(
        [[row[f"achieved_{key}"] for key in FEATURE_KEYS] for row in rows],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(feature_matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xticks(np.arange(len(FEATURE_KEYS)))
    ax.set_xticklabels(FEATURE_KEYS, rotation=25, ha="right")
    for i in range(feature_matrix.shape[0]):
        for j in range(feature_matrix.shape[1]):
            value = feature_matrix[i, j]
            colour = "white" if value < 0.25 or value > 0.75 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=colour)
    fig.colorbar(im, ax=ax, label="Achieved normalised feature")
    ax.set_title("Achieved features of diagnostic variants")
    fig.tight_layout()
    fig.savefig(output_dir / "sadness_reachability_features.png", dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded Point-Sadness expressive reachability screen."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/sadness_reachability_point_v1"),
    )
    parser.add_argument(
        "--screen",
        choices=("initial", "combined"),
        default="initial",
        help=(
            "'initial' runs the original six-variant screen. 'combined' runs "
            "four stronger probes using endpoint lowering, earlier retreat, "
            "and moderate contraction."
        ),
    )
    parser.add_argument(
        "--only-variant",
        default=None,
        help=(
            "Evaluate and save only one named variant from the selected screen. "
            "Useful for a larger frozen follow-up without paying to re-evaluate "
            "the other diagnostic variants."
        ),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maxiter", type=int, default=45)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--local-maxiter", type=int, default=100)
    parser.add_argument("--de-mutation", type=float, default=0.5)
    parser.add_argument("--de-recombination", type=float, default=0.65)
    parser.add_argument("--downward-endpoint-m", type=float, default=0.03)
    parser.add_argument("--contraction-scale", type=float, default=0.70)
    parser.add_argument("--retreat-m", type=float, default=0.03)
    parser.add_argument("--retreat-onset", type=float, default=0.70)
    parser.add_argument(
        "--strong-downward-endpoint-m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--strong-retreat-m",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--strong-retreat-onset",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--moderate-contraction-scale",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--ranges",
        default="configs/normalisation_ranges_balanced_3gestures_by_gesture.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and validate all motions without calling Gemini.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if not 0.0 < args.contraction_scale <= 1.0:
        raise ValueError("--contraction-scale must be in (0, 1].")
    if not 0.0 < args.moderate_contraction_scale <= 1.0:
        raise ValueError("--moderate-contraction-scale must be in (0, 1].")
    if not 0.0 <= args.retreat_onset < 1.0:
        raise ValueError("--retreat-onset must be in [0, 1).")
    if not 0.0 <= args.strong_retreat_onset < 1.0:
        raise ValueError("--strong-retreat-onset must be in [0, 1).")
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.out} is not empty. Use a new --out path or add --overwrite."
        )
    args.out.mkdir(parents=True, exist_ok=True)

    optimiser_overrides = {
        "maxiter": args.maxiter,
        "popsize": args.popsize,
        "local_maxiter": args.local_maxiter,
        "de_mutation": args.de_mutation,
        "de_recombination": args.de_recombination,
        "seed": args.seed,
    }

    print("Realising current Point-Sadness profile...")
    current_result = optimise_laban_target(
        gesture="point",
        target_state="sadness",
        target_profile=CURRENT_SADNESS_PROFILE,
        out_dir=args.out / "current_laban" / "optimiser_outputs",
        optimiser_overrides=optimiser_overrides,
    )
    exaggerated_result = None
    if args.screen == "initial":
        print("Realising exaggerated five-feature profile...")
        exaggerated_result = optimise_laban_target(
            gesture="point",
            target_state="sadness",
            target_profile=EXAGGERATED_SADNESS_PROFILE,
            out_dir=args.out / "exaggerated_laban" / "optimiser_outputs",
            optimiser_overrides=optimiser_overrides,
        )

    q_ref = np.asarray(current_result.q_ref, dtype=float)
    q_current = np.asarray(current_result.q_var, dtype=float)
    arm = laban.ArmConfig(
        n_points=len(q_ref), duration=2.0, l1=0.30, l2=0.25
    )
    ranges_path = Path(args.ranges)
    if not ranges_path.exists():
        ranges_path = PROJECT_ROOT / ranges_path
    ranges = load_ranges_or_default(str(ranges_path), gesture="point")

    if args.screen == "initial":
        q_down, down_metadata = _apply_downward_endpoint(
            q_current,
            l1=arm.l1,
            l2=arm.l2,
            displacement_m=args.downward_endpoint_m,
        )
        q_contract, contract_metadata = _apply_contraction(
            q_current,
            l1=arm.l1,
            l2=arm.l2,
            final_scale=args.contraction_scale,
        )
        q_retreat, retreat_metadata = _apply_terminal_retreat(
            q_current,
            l1=arm.l1,
            l2=arm.l2,
            retreat_m=args.retreat_m,
            onset_fraction=args.retreat_onset,
        )

        variants = [
            VariantSpec("reference", "reference_control", q_ref, None, "reference"),
            VariantSpec(
                "current_laban",
                "five_feature_baseline",
                q_current,
                CURRENT_SADNESS_PROFILE,
                "inner_optimiser",
            ),
            VariantSpec(
                "exaggerated_laban",
                "five_feature_exaggeration",
                np.asarray(exaggerated_result.q_var, dtype=float),
                EXAGGERATED_SADNESS_PROFILE,
                "inner_optimiser",
            ),
            VariantSpec(
                "downward_endpoint",
                "endpoint_height",
                q_down,
                CURRENT_SADNESS_PROFILE,
                "bounded_cartesian_diagnostic",
            ),
            VariantSpec(
                "contracted_reach",
                "amplitude_contraction",
                q_contract,
                CURRENT_SADNESS_PROFILE,
                "bounded_cartesian_diagnostic",
            ),
            VariantSpec(
                "terminal_retreat",
                "event_level_retreat",
                q_retreat,
                CURRENT_SADNESS_PROFILE,
                "bounded_cartesian_diagnostic",
            ),
        ]
        cue_metadata = {
            "reference": {},
            "current_laban": {},
            "exaggerated_laban": {},
            "downward_endpoint": down_metadata,
            "contracted_reach": contract_metadata,
            "terminal_retreat": retreat_metadata,
        }
    else:
        q_strong_down, strong_down_metadata = _apply_downward_endpoint(
            q_current,
            l1=arm.l1,
            l2=arm.l2,
            displacement_m=args.strong_downward_endpoint_m,
        )
        q_early_retreat, early_retreat_metadata = _apply_terminal_retreat(
            q_current,
            l1=arm.l1,
            l2=arm.l2,
            retreat_m=args.strong_retreat_m,
            onset_fraction=args.strong_retreat_onset,
        )

        # Compose cues sequentially. Each transform is solved back to a
        # continuous joint trajectory before the next cue is applied.
        q_down_retreat_stage, combined_down_metadata = _apply_downward_endpoint(
            q_current,
            l1=arm.l1,
            l2=arm.l2,
            displacement_m=args.strong_downward_endpoint_m,
        )
        q_down_retreat, combined_retreat_metadata = _apply_terminal_retreat(
            q_down_retreat_stage,
            l1=arm.l1,
            l2=arm.l2,
            retreat_m=args.strong_retreat_m,
            onset_fraction=args.strong_retreat_onset,
        )

        q_all_stage_1, all_contract_metadata = _apply_contraction(
            q_current,
            l1=arm.l1,
            l2=arm.l2,
            final_scale=args.moderate_contraction_scale,
        )
        q_all_stage_2, all_down_metadata = _apply_downward_endpoint(
            q_all_stage_1,
            l1=arm.l1,
            l2=arm.l2,
            displacement_m=args.strong_downward_endpoint_m,
        )
        q_all, all_retreat_metadata = _apply_terminal_retreat(
            q_all_stage_2,
            l1=arm.l1,
            l2=arm.l2,
            retreat_m=args.strong_retreat_m,
            onset_fraction=args.strong_retreat_onset,
        )

        variants = [
            VariantSpec(
                "strong_downward",
                "strong_endpoint_height",
                q_strong_down,
                CURRENT_SADNESS_PROFILE,
                "bounded_cartesian_diagnostic",
            ),
            VariantSpec(
                "early_retreat",
                "strong_early_retreat",
                q_early_retreat,
                CURRENT_SADNESS_PROFILE,
                "bounded_cartesian_diagnostic",
            ),
            VariantSpec(
                "downward_plus_retreat",
                "combined_height_and_retreat",
                q_down_retreat,
                CURRENT_SADNESS_PROFILE,
                "bounded_cartesian_diagnostic",
            ),
            VariantSpec(
                "downward_retreat_contracted",
                "combined_height_retreat_contraction",
                q_all,
                CURRENT_SADNESS_PROFILE,
                "bounded_cartesian_diagnostic",
            ),
        ]
        cue_metadata = {
            "strong_downward": strong_down_metadata,
            "early_retreat": early_retreat_metadata,
            "downward_plus_retreat": {
                "downward": combined_down_metadata,
                "retreat": combined_retreat_metadata,
            },
            "downward_retreat_contracted": {
                "contraction": all_contract_metadata,
                "downward": all_down_metadata,
                "retreat": all_retreat_metadata,
            },
        }

    if args.only_variant is not None:
        available_names = [spec.name for spec in variants]
        if args.only_variant not in available_names:
            raise ValueError(
                f"--only-variant {args.only_variant!r} is not available for "
                f"--screen {args.screen!r}. Available variants: "
                + ", ".join(available_names)
            )
        variants = [
            spec for spec in variants if spec.name == args.only_variant
        ]

    evaluator = None
    if not args.dry_run:
        evaluator = GeminiProVideoEvaluator(
            model=args.model,
            temperature=args.temperature,
            keep_uploaded_files=False,
        )
    context = Context(gesture="point", target_state="sadness")

    rows: list[dict[str, Any]] = []
    full_results: dict[str, Any] = {
        "experiment": "point_sadness_expressive_reachability",
        "screen": args.screen,
        "only_variant": args.only_variant,
        "diagnostic_not_training": True,
        "repeats_per_variant": 0 if args.dry_run else args.repeats,
        "decision_rule": {
            "mean_sadness_probability_at_least": 0.30,
            "or_classification_rate_at_least": 0.40,
        },
        "optimiser_configuration": optimiser_overrides,
        "variants": {},
    }

    try:
        for spec in variants:
            print(f"\nProcessing {spec.name}...")
            variant_dir = args.out / spec.name
            variant_dir.mkdir(parents=True, exist_ok=True)
            diagnostics = _kinematic_diagnostics(
                q_ref,
                spec.q_var,
                arm=arm,
                ranges=ranges,
                requested_profile=spec.requested_profile,
            )
            np.savez(
                variant_dir / "frozen_variant.npz",
                q_ref=q_ref,
                q_var=spec.q_var,
            )
            mp4_path = render_variant_only_mp4(
                q_ref,
                spec.q_var,
                variant_dir / "variant_only_vlm.mp4",
                duration_seconds=arm.duration,
                overwrite=True,
            )
            _save_gif_from_mp4(mp4_path, variant_dir / "variant_only.gif")

            perceptual = None
            # Feature strictness applies only to profiles produced by the inner
            # optimiser. Derived cue variants intentionally alter their achieved
            # profile; physical safety is the relevant gate for the screen.
            if evaluator is not None and diagnostics["physically_acceptable"]:
                perceptual = _evaluate_frozen_video(
                    evaluator=evaluator,
                    context=context,
                    q_ref=q_ref,
                    q_var=spec.q_var,
                    output_dir=variant_dir,
                    repeats=args.repeats,
                )

            result = {
                "variant": spec.name,
                "cue_type": spec.cue_type,
                "source": spec.source,
                "requested_profile": spec.requested_profile,
                "cue_parameters": cue_metadata[spec.name],
                "diagnostics": diagnostics,
                "perceptual": perceptual,
                "mp4": str(mp4_path),
                "gif": str(variant_dir / "variant_only.gif"),
            }
            full_results["variants"][spec.name] = result

            mean_sadness = (
                None if perceptual is None
                else perceptual["mean_target_probability"]
            )
            class_rate = (
                None if perceptual is None
                else perceptual["target_classification_rate"]
            )
            promising = bool(
                perceptual is not None
                and (mean_sadness >= 0.30 or class_rate >= 0.40)
            )
            row = {
                "variant": spec.name,
                "cue_type": spec.cue_type,
                "source": spec.source,
                "physically_acceptable": diagnostics["physically_acceptable"],
                "strict_feature_realisable": diagnostics["strict_feature_realisable"],
                "feature_rmse": diagnostics["feature_rmse"],
                "max_abs_feature_error": diagnostics["max_abs_feature_error"],
                "path_length_ratio": diagnostics["path_length_ratio"],
                "endpoint_error_m": diagnostics["endpoint_error_m"],
                "mean_sadness_probability": mean_sadness,
                "sadness_classification_rate": class_rate,
                "mean_margin": (
                    None if perceptual is None else perceptual["mean_margin"]
                ),
                "promising": promising,
            }
            for key in FEATURE_KEYS:
                row[f"achieved_{key}"] = diagnostics["achieved_profile"][key]
            rows.append(row)

            with (variant_dir / "diagnostic_result.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(_jsonable(result), handle, indent=2)
    finally:
        if evaluator is not None:
            evaluator.close()

    csv_path = args.out / "reachability_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "reachability_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_jsonable(full_results), handle, indent=2)

    _plot_results(rows, args.out)
    promising = [row["variant"] for row in rows if row["promising"]]
    print("\n" + "=" * 88)
    print("POINT-SADNESS REACHABILITY SCREEN COMPLETE")
    print("=" * 88)
    print(f"Output: {args.out.resolve()}")
    if args.dry_run:
        print("Dry run: motion generation and physical checks only; no Gemini calls.")
    elif promising:
        print("Promising variants: " + ", ".join(promising))
    else:
        print("No variant crossed the predeclared diagnostic threshold.")


if __name__ == "__main__":
    main()