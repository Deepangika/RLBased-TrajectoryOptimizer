from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = [ROOT / "outputs/experiment_cem_wave_friendly", ROOT / "outputs/experiment_cem_point_friendly"]
OUT = ROOT / "outputs/cem_outer_loop_audit"
OUT.mkdir(parents=True, exist_ok=True)


def load_samples(run: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((run / "rounds").glob("round_*_sample_*/sample_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["environment_result"]
        evaluations = result.get("perceptual_evaluations") or []
        target = result["context"]["target_state"]
        target_ps = [float(e[target]) for e in evaluations]
        winners = [max(e, key=e.get) for e in evaluations]
        rows.append({
            "run": run.name,
            "gesture": result["context"]["gesture"],
            "target": target,
            "round": int(path.parent.name.split("_")[1]),
            "sample": int(path.parent.name.split("_")[3]),
            "outer_reward": float(result["outer_reward"]),
            "target_probability": float(result["mean_target_probability"]),
            "margin": float(result["mean_margin"]),
            "realisation_rmse": float(result["realisation_rmse"]),
            "perceptual_reward_std": float(result["perceptual_reward_std"]),
            "repeat_target_probability_range": max(target_ps) - min(target_ps) if target_ps else np.nan,
            "repeat_target_probability_std": np.std(target_ps) if target_ps else np.nan,
            "repeat_winner_agreement": len(set(winners)) == 1 if winners else False,
            "target_wins": sum(w == target for w in winners),
            "repeat_count": len(winners),
            "failure_reason": result.get("failure_reason"),
        })
    return pd.DataFrame(rows)


all_samples = pd.concat([load_samples(run) for run in RUNS], ignore_index=True)
all_samples.to_csv(OUT / "sample_level_audit.csv", index=False)

summary = []
for run in RUNS:
    hist = pd.read_csv(run / "training_history.csv")
    x = all_samples[all_samples.run == run.name]
    best = x.loc[x.outer_reward.idxmax()]
    first = x[x["round"] <= 5]
    last = x[x["round"] >= x["round"].max() - 4]
    means = hist[[c for c in hist if c.startswith("mean_") and c not in {
        "mean_sample_reward", "mean_target_probability", "mean_realisation_rmse"}]]
    frozen_transitions = int(np.sum(np.all(np.isclose(np.diff(means.to_numpy(), axis=0), 0.0, atol=1e-12), axis=1)))
    summary.append({
        "run": run.name,
        "gesture": x.gesture.iloc[0],
        "target": x.target.iloc[0],
        "rounds": int(x["round"].max()),
        "samples": len(x),
        "first5_mean_reward": first.outer_reward.mean(),
        "last5_mean_reward": last.outer_reward.mean(),
        "mean_reward_change": last.outer_reward.mean() - first.outer_reward.mean(),
        "first5_target_probability": first.target_probability.mean(),
        "last5_target_probability": last.target_probability.mean(),
        "target_probability_change": last.target_probability.mean() - first.target_probability.mean(),
        "first5_realisation_rmse": first.realisation_rmse.mean(),
        "last5_realisation_rmse": last.realisation_rmse.mean(),
        "best_reward": best.outer_reward,
        "best_round": int(best["round"]),
        "best_target_probability": best.target_probability,
        "best_margin": best.margin,
        "best_realisation_rmse": best.realisation_rmse,
        "best_repeat_probability_range": best.repeat_target_probability_range,
        "best_repeat_winner_agreement": bool(best.repeat_winner_agreement),
        "all_repeat_winner_agreement_rate": x.repeat_winner_agreement.mean(),
        "target_classification_rate": x.target_wins.sum() / x.repeat_count.sum(),
        "mean_repeat_probability_range": x.repeat_target_probability_range.mean(),
        "frozen_distribution_transitions": frozen_transitions,
        "reward_probability_correlation": x.outer_reward.corr(x.target_probability),
        "reward_rmse_correlation": x.outer_reward.corr(x.realisation_rmse),
        "evaluator_failure_count": int(x.failure_reason.notna().sum()),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT / "run_level_audit.csv", index=False)

# Inspect whether the saved summaries label cumulative-best statistics as mean reward.
integrity = []
for run in RUNS:
    result = json.loads((run / "results_summary.json").read_text(encoding="utf-8"))
    hist = pd.read_csv(run / "training_history.csv")
    integrity.append({
        "run": run.name,
        "reported_mean_reward_all_rounds": result["mean_reward_all_rounds"],
        "actual_mean_sample_reward_all_rounds": hist.mean_sample_reward.mean(),
        "reported_mean_reward_last5": result["mean_reward_last_5_rounds"],
        "actual_mean_sample_reward_last5": hist.mean_sample_reward.tail(5).mean(),
        "reported_value_is_cumulative_best_mean": np.isclose(
            result["mean_reward_all_rounds"], hist.best_round_reward.mean()
        ),
    })
pd.DataFrame(integrity).to_csv(OUT / "summary_integrity_check.csv", index=False)

print(summary_df.to_string(index=False))
print("\nSummary integrity")
print(pd.DataFrame(integrity).to_string(index=False))
