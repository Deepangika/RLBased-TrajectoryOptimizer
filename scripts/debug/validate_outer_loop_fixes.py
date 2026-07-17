from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import LabanOptimisationResult
from laban_rl.perceptual_bandit.cem import CEMOptimizer
import laban_rl.perceptual_bandit.environment as environment_module
from laban_rl.perceptual_bandit.environment import (
    Context,
    EnvironmentRewardConfig,
    PerceptualBanditEnvironment,
    PerceptualEvaluation,
)

OUT = ROOT / "outputs/outer_loop_fix_validation"
OUT.mkdir(parents=True, exist_ok=True)


def vector(profile):
    return np.asarray([profile[key] for key in FEATURE_KEYS], dtype=float)


def run_synthetic(seed: int, noise_std: float, rounds: int = 20):
    target = np.asarray([0.80, 0.20, 0.65, 0.25, 0.70])
    cem = CEMOptimizer(
        {key: 0.5 for key in FEATURE_KEYS}, initial_width=0.18,
        seed=seed, smoothing=0.7, min_std=0.03, min_elites=3,
    )
    noise_rng = np.random.default_rng(10000 + seed)
    initial_rmse = float(np.sqrt(np.mean((vector(cem.get_mean_profile()) - target) ** 2)))
    best_observed = (-np.inf, None)
    for _ in range(rounds):
        profiles = cem.sample_batch(10)
        candidates = []
        for profile in profiles:
            true_reward = -float(np.mean((vector(profile) - target) ** 2))
            observed = true_reward + float(noise_rng.normal(0.0, noise_std))
            candidates.append((profile, observed))
            if observed > best_observed[0]:
                best_observed = (observed, profile)
        cem.update_elites(candidates, elite_fraction=0.4)
        cem.decay_exploration(0.98)
    final_rmse = float(np.sqrt(np.mean((vector(cem.get_mean_profile()) - target) ** 2)))
    selected_true_rmse = float(np.sqrt(np.mean((vector(best_observed[1]) - target) ** 2)))
    return initial_rmse, final_rmse, selected_true_rmse, cem


def validate_synthetic():
    results = {}
    for noise in (0.0, 0.05, 0.15):
        rows = [run_synthetic(seed, noise)[:3] for seed in range(100)]
        rows = np.asarray(rows)
        results[str(noise)] = {
            "initial_median_rmse": float(np.median(rows[:, 0])),
            "final_median_rmse": float(np.median(rows[:, 1])),
            "final_p90_rmse": float(np.quantile(rows[:, 1], 0.9)),
            "selected_best_true_median_rmse": float(np.median(rows[:, 2])),
            "fraction_improved": float(np.mean(rows[:, 1] < rows[:, 0])),
        }
    return results


def validate_resume():
    target = np.asarray([0.7, 0.3, 0.6, 0.2, 0.8])

    def advance(cem, rounds):
        for _ in range(rounds):
            profiles = cem.sample_batch(10)
            candidates = [(p, -float(np.mean((vector(p) - target) ** 2))) for p in profiles]
            cem.update_elites(candidates, 0.4)
            cem.decay_exploration(0.98)

    continuous = CEMOptimizer(seed=23)
    advance(continuous, 12)
    interrupted = CEMOptimizer(seed=23)
    advance(interrupted, 5)
    state = interrupted.state_dict()
    resumed = CEMOptimizer(seed=999)
    resumed.load_state_dict(state)
    advance(resumed, 7)
    return {
        "mean_exact_match": continuous.get_mean_profile() == resumed.get_mean_profile(),
        "std_exact_match": continuous.get_std_profile() == resumed.get_std_profile(),
        "round_counter_match": continuous.round_counter == resumed.round_counter == 12,
    }


class FixedEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, context, optimisation_result):
        self.calls += 1
        return PerceptualEvaluation({
            "friendly": 0.60, "calm": 0.15, "confident": 0.10,
            "hesitant": 0.05, "confused": 0.05, "angry": 0.05,
        })


def fake_result(*, path_ratio=1.0, joint_error=0.0, achieved=None):
    requested = {key: 0.5 for key in FEATURE_KEYS}
    achieved = achieved or requested
    return LabanOptimisationResult(
        gesture="wave", target_state="friendly", requested_profile=requested,
        achieved_profile=achieved, achieved_profile_clipped={key: np.clip(v, 0, 1) for key, v in achieved.items()},
        inner_reward=-0.1, inner_loss=0.1, action_coefficients=np.zeros(2),
        q_ref=np.zeros((4, 2)), q_var=np.zeros((4, 2)), output_dir=OUT,
        raw_result={"reward_info": {"path_length_ratio": path_ratio, "joint_limit_error": joint_error}},
    )


def validate_environment():
    original = environment_module.optimise_laban_target
    evaluator = FixedEvaluator()
    config = EnvironmentRewardConfig(repeat_evaluations=3, stability_penalty_weight=0.25)
    env = PerceptualBanditEnvironment(evaluator=evaluator, reward_config=config)
    context = Context("wave", "friendly")
    profile = {key: 0.5 for key in FEATURE_KEYS}
    try:
        environment_module.optimise_laban_target = lambda **_: fake_result(path_ratio=0.5)
        bad_path = env.step(context=context, action_profile=profile, out_dir=OUT)
        calls_after_bad_path = evaluator.calls

        environment_module.optimise_laban_target = lambda **_: fake_result(joint_error=1e-3)
        bad_joint = env.step(context=context, action_profile=profile, out_dir=OUT)
        calls_after_bad_joint = evaluator.calls

        achieved = dict(profile); achieved["weight"] = 1.2
        environment_module.optimise_laban_target = lambda **_: fake_result(achieved=achieved)
        valid = env.step(context=context, action_profile=profile, out_dir=OUT)
    finally:
        environment_module.optimise_laban_target = original

    expected_rmse = float(np.sqrt((0.7**2) / len(FEATURE_KEYS)))
    return {
        "bad_path_rejected": not bad_path.valid_realisation,
        "bad_path_reason": bad_path.failure_reason,
        "bad_joint_rejected": not bad_joint.valid_realisation,
        "bad_joint_reason": bad_joint.failure_reason,
        "vlm_not_called_for_invalid": calls_after_bad_path == 0 and calls_after_bad_joint == 0,
        "unclipped_rmse_used": bool(np.isclose(valid.realisation_rmse, expected_rmse)),
        "valid_repeat_count": len(valid.perceptual_evaluations),
        "target_classification_rate": valid.target_classification_rate,
        "winner_agreement_rate": valid.winner_agreement_rate,
        "finite_entropy": bool(np.isfinite(valid.mean_probability_entropy)),
    }


def validate_current_batch_elites():
    cem = CEMOptimizer(seed=3, smoothing=1.0)
    old = [({key: 0.9 for key in FEATURE_KEYS}, 100.0) for _ in range(4)]
    cem.update_elites(old, 0.4)
    new = [({key: 0.1 + i * 0.01 for key in FEATURE_KEYS}, float(i)) for i in range(10)]
    cem.update_elites(new, 0.4)
    return {
        "historical_elites_removed": all(elite.reward < 100.0 for elite in cem.elites),
        "elite_count": len(cem.elites),
        "mean_moved_to_current_batch": all(value < 0.25 for value in cem.get_mean_profile().values()),
    }


report = {
    "synthetic_convergence": validate_synthetic(),
    "checkpoint_resume": validate_resume(),
    "environment_guards": validate_environment(),
    "elite_archive": validate_current_batch_elites(),
}
(OUT / "validation_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

checks = [
    report["synthetic_convergence"]["0.0"]["fraction_improved"] >= 0.99,
    report["synthetic_convergence"]["0.05"]["fraction_improved"] >= 0.90,
    all(report["checkpoint_resume"].values()),
    report["environment_guards"]["bad_path_rejected"],
    report["environment_guards"]["bad_joint_rejected"],
    report["environment_guards"]["vlm_not_called_for_invalid"],
    report["environment_guards"]["unclipped_rmse_used"],
    report["elite_archive"]["historical_elites_removed"],
]
if not all(checks):
    raise SystemExit("One or more outer-loop validations failed.")
