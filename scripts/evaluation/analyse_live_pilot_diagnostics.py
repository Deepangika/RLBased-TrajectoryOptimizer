"""Offline diagnostics for the two live Gemini Stage A pilot runs.

Reads ONLY saved artifacts — it makes zero API calls and runs no new
learning. It produces:

- corrected (taxonomy v2) stage summaries for each run;
- six analyses (reward decomposition, search behaviour, training-to-
  validation robustness, gesture preservation / motion quality, evaluator
  language coding, metric agreement);
- integrity checks that fail loudly;
- JSON + CSV + Markdown reports and diagnostic plots in ``--out``.

Usage::

    uv run python scripts/evaluation/analyse_live_pilot_diagnostics.py \
        --beckon-fear-run outputs/experiments/outer_learning_beckon_fear_live_flash_v1 \
        --wave-sadness-run outputs/experiments/outer_learning_wave_sadness_live_flash_v1 \
        --baseline-root outputs/experiments/baseline_fixed_profiles \
        --out outputs/experiments/live_pilot_diagnostics_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from laban_rl.perceptual_bandit.baseline import load_baseline_motion  # noqa: E402
from laban_rl.perceptual_bandit.outcome_taxonomy import (  # noqa: E402
    assess_perceptual_outcome,
    classify_paired_preference,
)
from laban_rl.perceptual_bandit.outer_learning import weighted_vad_reward  # noqa: E402

EXPECTED_MODEL = "gemini-2.5-flash"
EXPECTED_TEMPERATURE = 0.2
VALIDATION_FEATURE_ERROR_LIMIT = 0.10
EXPECTED_ROBUST_MARGIN = 0.08
VAD_WEIGHTS = {"valence": 0.20, "arousal": 0.40, "dominance": 0.40}
ARM_DURATION_SECONDS = 2.0
ARM_L1 = 0.30
ARM_L2 = 0.25

COLOURS = {
    "reference": "0.45",
    "baseline": "tab:blue",
    "learned": "tab:orange",
    "target": "tab:green",
}

# Transparent keyword scheme for evaluator-language coding (no LLM).
LANGUAGE_CODES: dict[str, list[str]] = {
    "erratic_jittery": [
        "erratic", "jitter", "jerky", "trembl", "shak", "twitch",
        "frantic", "uncontrolled", "chaotic",
    ],
    "smooth_gentle": ["smooth", "gentle", "graceful", "fluid", "flowing", "soft"],
    "slow_heavy": ["slow", "sluggish", "heavy", "droop", "listless", "lethargic"],
    "fast_energetic": ["rapid", "fast", "quick", "energetic", "vigorous"],
    "hesitant_uncertain": ["hesitant", "hesitat", "uncertain", "tentative", "pause"],
    "distortion_incoherence": [
        "distort", "unnatural", "unrecogni", "incoheren", "unravel",
        "convoluted", "complex", "loop", "erratic path", "disorgani",
    ],
}


class IntegrityFailure(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_context_dir(run_root: Path) -> Path:
    stage = run_root / "live_stage_a"
    manifests = sorted(stage.glob("*/*/run_manifest.json"))
    if len(manifests) != 1:
        raise IntegrityFailure(
            f"Expected exactly one context under {stage}, found "
            f"{[str(m.parent) for m in manifests]}"
        )
    return manifests[0].parent


def load_run(run_root: Path) -> dict[str, Any]:
    stage = run_root / "live_stage_a"
    ctx_dir = find_context_dir(run_root)
    data = {
        "run_root": run_root,
        "context_dir": ctx_dir,
        "manifest": _read_json(ctx_dir / "run_manifest.json"),
        "stage_status": _read_json(stage / "stage_status.json"),
        "stopping_reason": _read_json(ctx_dir / "stopping_reason.json"),
        "profile_comparison": _read_json(ctx_dir / "profile_comparison.json"),
        "independent_validation": _read_json(ctx_dir / "independent_validation.json"),
        "paired_vs_baseline": _read_json(ctx_dir / "paired_learned_vs_baseline.json"),
        "paired_vs_reference": _read_json(ctx_dir / "paired_learned_vs_reference.json"),
        "initial_profile": _read_json(ctx_dir / "initial_profile.json"),
        "learned_profile": _read_json(ctx_dir / "learned_profile.json"),
        "selected_profile": _read_json(ctx_dir / "selected_validated_profile.json"),
        "round_history": _read_csv(ctx_dir / "round_history.csv"),
        "sample_history": _read_csv(ctx_dir / "sample_history.csv"),
        "mean_history": _read_csv(ctx_dir / "mean_history.csv"),
        "budget": _read_json(stage / "gemini_call_budget.json"),
    }
    with np.load(ctx_dir / "covariance_history.npz") as arrays:
        data["covariance_history"] = [
            np.asarray(arrays[key], dtype=float) for key in sorted(arrays.keys())
        ]
    context = data["manifest"]["context"]
    data["gesture"] = context["gesture"]
    data["state"] = context["target_state"]
    data["label"] = f"{data['gesture']}-{data['state']}"
    return data


def selected_validation_row(run: dict[str, Any]) -> dict[str, Any] | None:
    selected_id = run["selected_profile"].get("sample_id")
    for row in run["independent_validation"]["candidates"]:
        if row.get("sample_id") == selected_id:
            return row
    return None


def selected_variant_npz(run: dict[str, Any]) -> Path:
    row = selected_validation_row(run)
    if row is None:
        raise IntegrityFailure(f"{run['label']}: no selected validation row.")
    expected = {key: float(value) for key, value in row["profile"].items()}
    for cand_dir in sorted((run["context_dir"] / "validation").glob("candidate_*")):
        resolved_path = cand_dir / "resolved_target_profile.json"
        npz = cand_dir / "best_variant.npz"
        if not resolved_path.exists() or not npz.exists():
            continue
        profile = _read_json(resolved_path)["laban_profile"]
        if all(
            abs(float(profile[key]) - expected[key]) < 1e-9 for key in expected
        ):
            return npz
    raise IntegrityFailure(
        f"{run['label']}: no validation candidate dir matches the selected "
        f"profile {expected}"
    )


# ---------------------------------------------------------------------------
# Analysis 1 — reward decomposition
# ---------------------------------------------------------------------------


def reward_decomposition(run: dict[str, Any]) -> dict[str, Any]:
    comp = run["profile_comparison"]
    target = comp["canonical_target_vad"]
    baseline_vad = comp["initial_observed_vad"]
    learned_vad = comp["learned_observed_vad"]

    def axis_terms(observed: dict[str, float]) -> dict[str, float]:
        return {
            axis: VAD_WEIGHTS[axis] * abs(float(observed[axis]) - float(target[axis]))
            for axis in ("valence", "arousal", "dominance")
        }

    baseline_terms = axis_terms(baseline_vad)
    learned_terms = axis_terms(learned_vad)
    deltas = {
        axis: learned_terms[axis] - baseline_terms[axis]
        for axis in baseline_terms
    }
    dominant_axis = max(deltas, key=lambda axis: deltas[axis])
    return {
        "target_vad": target,
        "baseline_observed_vad": baseline_vad,
        "learned_observed_vad": learned_vad,
        "baseline_weighted_axis_error": baseline_terms,
        "learned_weighted_axis_error": learned_terms,
        "axis_error_delta_learned_minus_baseline": deltas,
        "axis_driving_regression": dominant_axis,
        "baseline_reward_check": weighted_vad_reward(baseline_vad, target),
        "learned_reward_check": weighted_vad_reward(learned_vad, target),
    }


# ---------------------------------------------------------------------------
# Analysis 2 — search behaviour per round
# ---------------------------------------------------------------------------


def search_behaviour(run: dict[str, Any]) -> dict[str, Any]:
    rounds = []
    for row in run["round_history"]:
        rounds.append(
            {
                "round": int(row["round"]),
                "best_reward": float(row["best_reward"]),
                "mean_reward": float(row["mean_reward"]),
                "feasible_candidates": int(row["feasible_candidates"]),
                "robust_candidates": int(row["robust_candidates"]),
                "marginal_candidates": int(row["marginal_candidates"]),
                "action_std": float(row["action_std"]),
            }
        )
    total = len(run["sample_history"])
    feasible = sum(1 for r in run["sample_history"] if r["feasible"] == "True")
    return {
        "rounds": rounds,
        "total_candidates": total,
        "feasible_candidates": feasible,
        "feasible_fraction": feasible / total if total else None,
        "best_reward_trend": [r["best_reward"] for r in rounds],
        "action_std_trend": [r["action_std"] for r in rounds],
    }


# ---------------------------------------------------------------------------
# Analysis 3 — training-to-validation robustness
# ---------------------------------------------------------------------------


def validation_robustness(run: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in run["independent_validation"]["candidates"]:
        training_reward = row.get("training_reward")
        validation_reward = row.get("validation_reward")
        rows.append(
            {
                "sample_id": row["sample_id"],
                "rank": int(row["rank"]),
                "training_reward": (
                    float(training_reward) if training_reward is not None else None
                ),
                "validation_reward": (
                    float(validation_reward) if validation_reward is not None else None
                ),
                "reward_gap_validation_minus_training": (
                    float(validation_reward) - float(training_reward)
                    if training_reward is not None and validation_reward is not None
                    else None
                ),
                "training_max_abs_feature_error": row.get("training_max_abs_feature_error"),
                "validation_max_abs_feature_error": row.get("validation_max_abs_feature_error"),
                "feasibility_survived_seed_change": bool(
                    row.get("feasibility_survived_seed_change")
                ),
                "ineligibility_reasons": list(row.get("ineligibility_reasons", [])),
            }
        )
    survived = sum(1 for r in rows if r["feasibility_survived_seed_change"])
    return {
        "shortlist_size": len(rows),
        "survived_seed_change": survived,
        "survival_fraction": survived / len(rows) if rows else None,
        "candidates": rows,
    }


# ---------------------------------------------------------------------------
# Analysis 4 — gesture preservation / motion quality
# ---------------------------------------------------------------------------


def _wrist_path(q: np.ndarray) -> np.ndarray:
    shoulder = q[:, 0]
    elbow = q[:, 1]
    x = ARM_L1 * np.cos(shoulder) + ARM_L2 * np.cos(shoulder + elbow)
    y = ARM_L1 * np.sin(shoulder) + ARM_L2 * np.sin(shoulder + elbow)
    return np.stack([x, y], axis=1)


def motion_metrics(q_ref: np.ndarray, q_var: np.ndarray) -> dict[str, float]:
    dt = ARM_DURATION_SECONDS / (len(q_var) - 1)
    vel = np.diff(q_var, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    wrist = _wrist_path(q_var)
    path_len = float(np.sum(np.linalg.norm(np.diff(wrist, axis=0), axis=1)))
    return {
        "rms_joint_velocity": float(np.sqrt(np.mean(vel**2))),
        "rms_joint_acceleration": float(np.sqrt(np.mean(acc**2))),
        "rms_joint_jerk": float(np.sqrt(np.mean(jerk**2))),
        "wrist_path_length": path_len,
        "mean_abs_deviation_from_reference": float(np.mean(np.abs(q_var - q_ref))),
        "endpoint_error": float(np.linalg.norm(q_var[-1] - q_ref[-1])),
    }


def gesture_preservation(run: dict[str, Any]) -> dict[str, Any]:
    gesture, state = run["gesture"], run["state"]
    reference = load_baseline_motion(gesture, state, candidate="reference")
    baseline = load_baseline_motion(gesture, state, candidate="styled")
    npz_path = selected_variant_npz(run)
    with np.load(npz_path, allow_pickle=True) as arrays:
        q_ref = np.asarray(arrays["q_ref"], dtype=float)
        q_var = np.asarray(arrays["q_var"], dtype=float)
    return {
        "learned_variant_npz": str(npz_path),
        "reference": motion_metrics(reference.q_ref, reference.q_var),
        "baseline_styled": motion_metrics(baseline.q_ref, baseline.q_var),
        "learned_selected": motion_metrics(q_ref, q_var),
        "note": (
            "Descriptive kinematic metrics only; no thresholds are applied "
            "and no claim of acceptability is made."
        ),
    }


# ---------------------------------------------------------------------------
# Analysis 5 — evaluator language coding
# ---------------------------------------------------------------------------


def code_text(text: str) -> list[str]:
    lowered = text.lower()
    return [
        code
        for code, keywords in LANGUAGE_CODES.items()
        if any(keyword in lowered for keyword in keywords)
    ]


def language_coding(run: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    for name in ("paired_vs_baseline", "paired_vs_reference"):
        payload = run[name]
        counts = {code: 0 for code in LANGUAGE_CODES}
        excerpts: list[dict[str, Any]] = []
        n_texts = 0
        for record in payload.get("records", []):
            for obs in record.get("observations", []):
                text = obs.get("reasoning_summary")
                if not text:
                    continue
                n_texts += 1
                codes = code_text(text)
                for code in codes:
                    counts[code] += 1
                excerpts.append(
                    {
                        "choice": obs.get("choice"),
                        "codes": codes,
                        "excerpt": text[:300],
                    }
                )
        comparisons[name] = {
            "coded_texts": n_texts,
            "code_counts": counts,
            "observations": excerpts,
        }
    return {
        "scheme": {code: kws for code, kws in LANGUAGE_CODES.items()},
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Analysis 6 + corrected summaries — metric agreement
# ---------------------------------------------------------------------------


def corrected_summary(run: dict[str, Any]) -> dict[str, Any]:
    comp = run["profile_comparison"]
    ctx_status = run["stage_status"]["contexts"][0]
    stop = run["stopping_reason"]["stopping_reason"]
    assessment = assess_perceptual_outcome(
        execution_success=stop not in ("evaluator_failure", "call_budget_exhausted"),
        selection_success=comp.get("selection_status") == "selected",
        selected_validation_reward=ctx_status.get("selected_validation_reward"),
        baseline_validation_reward=ctx_status.get("baseline_validation_reward"),
        paired_vs_baseline=run["paired_vs_baseline"],
        paired_vs_reference=run["paired_vs_reference"],
        comparison_status=comp.get("comparison_status"),
    )
    return {
        "context": {"gesture": run["gesture"], "target_state": run["state"]},
        "legacy_outcome": comp.get("outcome"),
        "legacy_all_successful": run["stage_status"].get("all_successful"),
        "corrected_assessment": assessment,
        "stopping_reason": stop,
    }


def metric_agreement(runs: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for run, summary in zip(runs, summaries):
        a = summary["corrected_assessment"]
        table.append(
            {
                "context": run["label"],
                "absolute_reward_delta": a["absolute_reward_delta_vs_baseline"],
                "absolute_improves": a["absolute_reward_improvement"],
                "paired_vs_baseline": a["paired_improvement_vs_baseline"],
                "paired_vs_baseline_counts": a["paired_vs_baseline_counts"],
                "paired_vs_reference": a["paired_improvement_vs_reference"],
                "paired_vs_reference_counts": a["paired_vs_reference_counts"],
                "metrics_agree": (
                    a["absolute_reward_improvement"]
                    == (a["paired_improvement_vs_baseline"] == "strong_preference_improvement")
                    if a["absolute_reward_improvement"] is not None
                    else None
                ),
                "legacy_outcome": summary["legacy_outcome"],
                "corrected_outcome": a["perceptual_outcome"],
            }
        )
    return table


# ---------------------------------------------------------------------------
# Integrity checks (fail loudly)
# ---------------------------------------------------------------------------


def integrity_checks(runs: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    labels = [run["label"] for run in runs]
    check(len(set(labels)) == len(labels), f"Context separation violated: {labels}")

    for run in runs:
        label = run["label"]
        manifest = run["manifest"]
        check(
            manifest.get("model") == EXPECTED_MODEL,
            f"{label}: model {manifest.get('model')!r} != {EXPECTED_MODEL!r}",
        )
        check(
            float(manifest.get("temperature")) == EXPECTED_TEMPERATURE,
            f"{label}: temperature {manifest.get('temperature')} != {EXPECTED_TEMPERATURE}",
        )
        check(
            float(manifest.get("robust_elite_max_feature_error")) == EXPECTED_ROBUST_MARGIN,
            f"{label}: robust margin {manifest.get('robust_elite_max_feature_error')} "
            f"!= {EXPECTED_ROBUST_MARGIN}",
        )

        history = run["covariance_history"]
        check(len(history) >= 2, f"{label}: covariance history too short")
        for index, cov in enumerate(history):
            check(
                bool(np.allclose(cov, cov.T, atol=1e-10)),
                f"{label}: covariance entry {index} not symmetric",
            )
            eigenvalues = np.linalg.eigvalsh(cov)
            check(
                bool(np.all(eigenvalues > 0.0)),
                f"{label}: covariance entry {index} not positive definite "
                f"(min eig {eigenvalues.min():.3e})",
            )
        check(
            not np.allclose(history[0], history[-1]),
            f"{label}: initial and final covariance are identical",
        )
        initial_saved = np.asarray(
            run["initial_profile"]["initial_cem_covariance"], dtype=float
        )
        check(
            not np.allclose(initial_saved, history[-1]),
            f"{label}: saved initial covariance equals final covariance "
            "(mutation regression)",
        )

        row = selected_validation_row(run)
        check(row is not None, f"{label}: selected candidate missing from validation")
        if row is not None:
            check(
                bool(row["feasible"]) and bool(row["valid_realisation"]),
                f"{label}: selected candidate not validation-feasible",
            )
            check(
                float(row["validation_max_abs_feature_error"])
                <= VALIDATION_FEATURE_ERROR_LIMIT,
                f"{label}: selected candidate validation feature error "
                f"{row['validation_max_abs_feature_error']} > "
                f"{VALIDATION_FEATURE_ERROR_LIMIT}",
            )

        # Call accounting.
        budget = run["budget"]
        categories = dict(budget.get("calls_by_category", {}))
        check(
            sum(categories.values()) == int(budget["total_calls"]),
            f"{label}: per-category calls {categories} do not sum to "
            f"total_calls={budget['total_calls']}",
        )
        check(
            int(budget["total_calls"]) <= int(budget["limit"]),
            f"{label}: total_calls exceeds limit",
        )
        feasible_training = sum(
            1 for r in run["sample_history"] if r["feasible"] == "True"
        )
        repeats = int(run["manifest"]["candidate_vlm_repeats"])
        expected_min_search = feasible_training * repeats
        search_calls = int(categories.get("search", 0))
        check(
            search_calls >= expected_min_search,
            f"{label}: search calls {search_calls} < feasible×repeats "
            f"{expected_min_search} (retries make it >=, never <)",
        )
        val_repeats = int(run["manifest"]["validation_vlm_repeats"])
        survivors = sum(
            1
            for r in run["independent_validation"]["candidates"]
            if r.get("feasibility_survived_seed_change")
        )
        validation_calls = int(categories.get("validation", 0))
        check(
            validation_calls >= survivors * val_repeats,
            f"{label}: validation calls {validation_calls} < "
            f"survivors×repeats {survivors * val_repeats}",
        )

        # Raw paired counts must reproduce labels.
        for name in ("paired_vs_baseline", "paired_vs_reference"):
            payload = run[name]
            if not payload.get("records"):
                continue
            counts = payload["records"][0]["choice_counts"]
            label_check = classify_paired_preference(
                counts["styled"],
                counts["reference"],
                counts["neither"],
                counts["styled"] + counts["reference"] + counts["neither"],
            )
            check(
                label_check in (
                    "strong_preference_improvement",
                    "weak_or_inconclusive_preference",
                    "no_preference_improvement",
                ),
                f"{label}: {name} counts {counts} produced invalid label",
            )
    return failures


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_plots(runs: list[dict[str, Any]], analyses: dict[str, Any], out: Path) -> list[str]:
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(fig: plt.Figure, name: str) -> None:
        path = plot_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))

    # 1. Reward per round.
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in runs:
        rounds = analyses[run["label"]]["search_behaviour"]["rounds"]
        x = [r["round"] for r in rounds]
        ax.plot(x, [r["best_reward"] for r in rounds], marker="o", label=f"{run['label']} best")
        ax.plot(x, [r["mean_reward"] for r in rounds], marker="x", linestyle="--", label=f"{run['label']} mean")
    ax.set_xlabel("round"); ax.set_ylabel("training reward"); ax.legend(fontsize=8)
    ax.set_title("Training reward per round")
    save(fig, "reward_per_round.png")

    # 2. Feasibility per round.
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in runs:
        rounds = analyses[run["label"]]["search_behaviour"]["rounds"]
        ax.plot(
            [r["round"] for r in rounds],
            [r["feasible_candidates"] for r in rounds],
            marker="o",
            label=run["label"],
        )
    ax.set_xlabel("round"); ax.set_ylabel("feasible candidates"); ax.legend()
    ax.set_title("Strictly feasible candidates per round")
    save(fig, "feasible_per_round.png")

    # 3. CEM mean evolution.
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        history = run["mean_history"]
        features = [k for k in history[0].keys() if k != "round"]
        for feature in features:
            ax.plot(
                [int(r["round"]) for r in history],
                [float(r[feature]) for r in history],
                marker="o",
                label=feature,
            )
        ax.set_title(f"CEM mean — {run['label']}"); ax.set_xlabel("round")
        ax.legend(fontsize=7)
    save(fig, "cem_mean_evolution.png")

    # 4. Covariance diagonal evolution.
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        history = run["covariance_history"]
        diag = np.stack([np.diag(cov) for cov in history])
        for j in range(diag.shape[1]):
            ax.plot(diag[:, j], marker="o", label=f"dim {j}")
        ax.set_title(f"Covariance diagonal — {run['label']}")
        ax.set_xlabel("history entry"); ax.legend(fontsize=7)
    save(fig, "covariance_diagonal_evolution.png")

    # 5. VAD positions vs target.
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4.5), squeeze=False)
    for ax, run in zip(axes[0], runs):
        decomposition = analyses[run["label"]]["reward_decomposition"]
        for key, colour in (
            ("baseline_observed_vad", COLOURS["baseline"]),
            ("learned_observed_vad", COLOURS["learned"]),
            ("target_vad", COLOURS["target"]),
        ):
            vad = decomposition[key]
            ax.scatter(vad["valence"], vad["arousal"], color=colour, s=90,
                       label=key.replace("_", " "))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("valence"); ax.set_ylabel("arousal")
        ax.set_title(f"VAD (V-A plane) — {run['label']}"); ax.legend(fontsize=7)
    save(fig, "vad_positions.png")

    # 6. Training vs validation reward.
    fig, ax = plt.subplots(figsize=(6, 5))
    for run in runs:
        rows = analyses[run["label"]]["validation_robustness"]["candidates"]
        xs = [r["training_reward"] for r in rows if r["validation_reward"] is not None]
        ys = [r["validation_reward"] for r in rows if r["validation_reward"] is not None]
        ax.scatter(xs, ys, label=run["label"], s=70)
    lims = [0.4, 1.0]
    ax.plot(lims, lims, color="0.6", linestyle=":")
    ax.set_xlabel("training reward"); ax.set_ylabel("validation reward"); ax.legend()
    ax.set_title("Training vs independent validation reward")
    save(fig, "training_vs_validation_reward.png")

    # 7. Paired preference counts.
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        groups = ("vs baseline", "vs reference")
        payloads = (run["paired_vs_baseline"], run["paired_vs_reference"])
        width = 0.25
        for offset, (choice, colour) in enumerate(
            (("styled", COLOURS["learned"]), ("reference", COLOURS["reference"]),
             ("neither", "0.75"))
        ):
            values = [p["records"][0]["choice_counts"][choice] for p in payloads]
            ax.bar(
                [i + offset * width for i in range(len(groups))],
                values, width=width,
                label=f"learned wins" if choice == "styled" else choice,
                color=colour,
            )
        ax.set_xticks([i + width for i in range(len(groups))])
        ax.set_xticklabels(groups)
        ax.set_title(f"Paired preference — {run['label']}"); ax.legend(fontsize=8)
    save(fig, "paired_preference_counts.png")

    # 8. Motion metric comparison.
    metrics = ["rms_joint_velocity", "rms_joint_jerk", "wrist_path_length",
               "mean_abs_deviation_from_reference"]
    fig, axes = plt.subplots(1, len(runs), figsize=(7 * len(runs), 4.5), squeeze=False)
    for ax, run in zip(axes[0], runs):
        preservation = analyses[run["label"]]["gesture_preservation"]
        width = 0.25
        for offset, (variant, colour) in enumerate(
            (("reference", COLOURS["reference"]),
             ("baseline_styled", COLOURS["baseline"]),
             ("learned_selected", COLOURS["learned"]))
        ):
            values = [preservation[variant][m] for m in metrics]
            ax.bar([i + offset * width for i in range(len(metrics))], values,
                   width=width, label=variant, color=colour)
        ax.set_xticks([i + width for i in range(len(metrics))])
        ax.set_xticklabels(metrics, rotation=20, fontsize=7)
        ax.set_title(f"Motion metrics — {run['label']}"); ax.legend(fontsize=8)
    save(fig, "motion_metric_comparison.png")

    # 9. Reward decomposition per axis.
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        decomposition = analyses[run["label"]]["reward_decomposition"]
        axes_names = ["valence", "arousal", "dominance"]
        width = 0.35
        ax.bar(
            [i for i in range(3)],
            [decomposition["baseline_weighted_axis_error"][a] for a in axes_names],
            width=width, label="baseline", color=COLOURS["baseline"],
        )
        ax.bar(
            [i + width for i in range(3)],
            [decomposition["learned_weighted_axis_error"][a] for a in axes_names],
            width=width, label="learned", color=COLOURS["learned"],
        )
        ax.set_xticks([i + width / 2 for i in range(3)])
        ax.set_xticklabels(axes_names)
        ax.set_ylabel("weighted |error|")
        ax.set_title(f"Weighted VAD axis error — {run['label']}"); ax.legend(fontsize=8)
    save(fig, "reward_decomposition.png")

    return written


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def write_csv_tables(analyses: dict[str, Any], agreement: list[dict[str, Any]], out: Path) -> None:
    with (out / "metric_agreement.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["context", "absolute_reward_delta", "absolute_improves",
             "paired_vs_baseline", "baseline_counts", "paired_vs_reference",
             "reference_counts", "metrics_agree", "legacy_outcome", "corrected_outcome"]
        )
        for row in agreement:
            bc, rc = row["paired_vs_baseline_counts"], row["paired_vs_reference_counts"]
            writer.writerow(
                [row["context"], row["absolute_reward_delta"], row["absolute_improves"],
                 row["paired_vs_baseline"],
                 f"{bc['learned_wins']}/{bc['opponent_wins']}/{bc['neither']}",
                 row["paired_vs_reference"],
                 f"{rc['learned_wins']}/{rc['opponent_wins']}/{rc['neither']}",
                 row["metrics_agree"], row["legacy_outcome"], row["corrected_outcome"]]
            )
    with (out / "round_behaviour.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["context", "round", "best_reward", "mean_reward",
                         "feasible_candidates", "robust_candidates", "action_std"])
        for label, payload in analyses.items():
            for row in payload["search_behaviour"]["rounds"]:
                writer.writerow([label, row["round"], row["best_reward"],
                                 row["mean_reward"], row["feasible_candidates"],
                                 row["robust_candidates"], row["action_std"]])
    with (out / "motion_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        metric_names = ["rms_joint_velocity", "rms_joint_acceleration", "rms_joint_jerk",
                        "wrist_path_length", "mean_abs_deviation_from_reference",
                        "endpoint_error"]
        writer.writerow(["context", "variant", *metric_names])
        for label, payload in analyses.items():
            for variant in ("reference", "baseline_styled", "learned_selected"):
                metrics = payload["gesture_preservation"][variant]
                writer.writerow([label, variant, *[metrics[m] for m in metric_names]])


def write_markdown(
    runs: list[dict[str, Any]],
    analyses: dict[str, Any],
    summaries: list[dict[str, Any]],
    agreement: list[dict[str, Any]],
    plots: list[str],
    out: Path,
) -> None:
    lines = [
        "# Live pilot diagnostics (offline)",
        "",
        "All figures below derive from saved artifacts. Zero API calls were "
        "made. Mock/machine-evaluator metrics are not human perceptual "
        "evidence.",
        "",
        "## Corrected outcomes (taxonomy v2)",
        "",
        "| Context | Legacy outcome | Corrected outcome | Abs Δ | Paired vs baseline | Paired vs reference |",
        "|---|---|---|---|---|---|",
    ]
    for row in agreement:
        bc, rc = row["paired_vs_baseline_counts"], row["paired_vs_reference_counts"]
        lines.append(
            f"| {row['context']} | {row['legacy_outcome']} | {row['corrected_outcome']} "
            f"| {row['absolute_reward_delta']:+.3f} "
            f"| {row['paired_vs_baseline']} ({bc['learned_wins']}/{bc['repeats']}) "
            f"| {row['paired_vs_reference']} ({rc['learned_wins']}/{rc['repeats']}) |"
        )
    lines += ["", "## Reward decomposition", ""]
    for run in runs:
        d = analyses[run["label"]]["reward_decomposition"]
        lines += [
            f"### {run['label']}",
            "",
            f"- Axis driving regression: **{d['axis_driving_regression']}**",
            f"- Baseline weighted axis errors: {json.dumps(d['baseline_weighted_axis_error'])}",
            f"- Learned weighted axis errors: {json.dumps(d['learned_weighted_axis_error'])}",
            "",
        ]
    lines += ["## Training-to-validation robustness", ""]
    for run in runs:
        r = analyses[run["label"]]["validation_robustness"]
        lines.append(
            f"- {run['label']}: {r['survived_seed_change']}/{r['shortlist_size']} "
            "shortlist candidates survived the validation seed change."
        )
    lines += ["", "## Evaluator language coding", ""]
    for run in runs:
        coding = analyses[run["label"]]["language_coding"]["comparisons"]
        for name, payload in coding.items():
            top = sorted(payload["code_counts"].items(), key=lambda kv: -kv[1])[:3]
            lines.append(f"- {run['label']} {name}: top codes {top}")
    lines += ["", "## Plots", ""]
    lines += [f"- `{p}`" for p in plots]
    (out / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beckon-fear-run", type=Path, required=True)
    parser.add_argument("--wave-sadness-run", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.baseline_root / "baseline_manifest.json").exists():
        raise IntegrityFailure(
            f"Baseline root {args.baseline_root} missing baseline_manifest.json"
        )
    runs = [load_run(args.beckon_fear_run), load_run(args.wave_sadness_run)]

    failures = integrity_checks(runs)
    if failures:
        for failure in failures:
            print(f"INTEGRITY FAILURE: {failure}", file=sys.stderr)
        raise SystemExit(2)

    analyses: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    for run in runs:
        analyses[run["label"]] = {
            "reward_decomposition": reward_decomposition(run),
            "search_behaviour": search_behaviour(run),
            "validation_robustness": validation_robustness(run),
            "gesture_preservation": gesture_preservation(run),
            "language_coding": language_coding(run),
        }
        summary = corrected_summary(run)
        summaries.append(summary)
        corrected_path = (
            run["run_root"] / "live_stage_a" / "corrected_stage_status.json"
        )
        corrected_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

    agreement = metric_agreement(runs, summaries)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "diagnostics.json").write_text(
        json.dumps(
            {
                "runs": [
                    {"label": run["label"], "run_root": str(run["run_root"])}
                    for run in runs
                ],
                "analyses": analyses,
                "corrected_summaries": summaries,
                "metric_agreement": agreement,
                "integrity": "all_checks_passed",
                "api_calls_made": 0,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    plots = make_plots(runs, analyses, args.out)
    write_csv_tables(analyses, agreement, args.out)
    write_markdown(runs, analyses, summaries, agreement, plots, args.out)
    print(f"Diagnostics written to {args.out} ({len(plots)} plots). Zero API calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
