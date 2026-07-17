from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/revised_optimizer_comparison/revised_four_states_120_runs.csv"
OUT = ROOT / "outputs/final_optimizer_metrics"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GESTURES = ["wave", "reach", "point"]
STATES = ["confident", "friendly", "calm", "confused"]
FEATURES = ["weight", "time", "flow_boundness", "space_indirectness", "shape_arcness"]
FLABELS = ["Weight", "Time", "Flow", "Space", "Shape"]

df = pd.read_csv(SOURCE)

success = np.zeros((3, 4))
rmse = np.zeros((3, 4))
acceptable = np.zeros((3, 4))
for gi, gesture in enumerate(GESTURES):
    for si, state in enumerate(STATES):
        x = df[(df.gesture == gesture) & (df.state == state)]
        success[gi, si] = 100 * x.realised_all_features.mean()
        acceptable[gi, si] = 100 * x.fully_acceptable.mean()
        rmse[gi, si] = x.unclipped_rmse.median()


def annotate(ax, values, formatter, threshold=None):
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            dark = threshold is not None and values[i, j] >= threshold
            ax.text(j, i, formatter(values[i, j]), ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white" if dark else "#172033")


# 1. Success heatmap
fig, ax = plt.subplots(figsize=(8.6, 4.3))
im = ax.imshow(success, cmap="YlGn", vmin=0, vmax=100, aspect="auto")
annotate(ax, success, lambda v: f"{v:.0f}%", threshold=60)
ax.set_xticks(range(4), [s.title() for s in STATES])
ax.set_yticks(range(3), [g.title() for g in GESTURES])
ax.set_title("Final optimiser target-realisation success", loc="left", fontsize=14, fontweight="bold")
ax.set_xlabel("Target state")
fig.colorbar(im, ax=ax, fraction=.04, pad=.03, label="Successful runs (%)")
ax.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig_final_success_heatmap.png", dpi=260, bbox_inches="tight")
plt.close(fig)

# 2. Median RMSE heatmap
fig, ax = plt.subplots(figsize=(8.6, 4.3))
im = ax.imshow(rmse, cmap="YlOrRd", vmin=0, vmax=max(.05, rmse.max()), aspect="auto")
annotate(ax, rmse, lambda v: f"{v:.4f}")
ax.set_xticks(range(4), [s.title() for s in STATES])
ax.set_yticks(range(3), [g.title() for g in GESTURES])
ax.set_title("Median normalised feature RMSE", loc="left", fontsize=14, fontweight="bold")
ax.set_xlabel("Target state")
fig.colorbar(im, ax=ax, fraction=.04, pad=.03, label="Median RMSE (lower is better)")
ax.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig_final_rmse_heatmap.png", dpi=260, bbox_inches="tight")
plt.close(fig)

# 3. Per-feature median error by gesture
rows = []
for gesture in GESTURES:
    x = df[df.gesture == gesture]
    for feature, label in zip(FEATURES, FLABELS):
        rows.append({"gesture": gesture, "feature": label,
                     "median_error": x[f"unclipped_error_{feature}"].median()})
feature_df = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
for ax, gesture in zip(axes, GESTURES):
    x = feature_df[feature_df.gesture == gesture]
    ax.bar(x.feature, x.median_error, color="#2a9d8f")
    ax.axhline(.10, color="#c44536", linestyle="--", linewidth=1.4, label="Success threshold")
    ax.set_title(gesture.title(), fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Median absolute normalised error")
axes[-1].legend(frameon=False, fontsize=8)
fig.suptitle("Final optimiser per-feature accuracy", x=.06, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(FIG / "fig_final_feature_errors.png", dpi=260, bbox_inches="tight")
plt.close(fig)

# 4. Reliability summary
metrics = pd.DataFrame({
    "metric": ["Feature success", "Fully acceptable", "Finite features", "Within calibrated range"],
    "percentage": [100 * df.realised_all_features.mean(), 100 * df.fully_acceptable.mean(),
                   100 * df.valid_features.mean(), 100 * (df.normalisation_clipped_count == 0).mean()],
})
fig, ax = plt.subplots(figsize=(8.8, 4.3))
bars = ax.barh(metrics.metric, metrics.percentage, color=["#287271", "#2a9d8f", "#70a288", "#9cc5a1"])
for bar, value in zip(bars, metrics.percentage):
    ax.text(value - 1.2, bar.get_y() + bar.get_height()/2, f"{value:.1f}%",
            ha="right", va="center", color="white", fontweight="bold")
ax.set_xlim(0, 100)
ax.set_xlabel("Runs satisfying criterion (%)")
ax.set_title("Final optimiser reliability over 120 runs", loc="left", fontsize=14, fontweight="bold")
ax.grid(axis="x", alpha=.25)
ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig_final_reliability.png", dpi=260, bbox_inches="tight")
plt.close(fig)

# Paper-ready summaries
gesture_summary = df.groupby("gesture").agg(
    runs=("seed", "size"),
    success_rate=("realised_all_features", "mean"),
    fully_acceptable_rate=("fully_acceptable", "mean"),
    median_rmse=("unclipped_rmse", "median"),
    mean_rmse=("unclipped_rmse", "mean"),
    maximum_rmse=("unclipped_rmse", "max"),
).reset_index()
state_summary = df.groupby("state").agg(
    runs=("seed", "size"),
    success_rate=("realised_all_features", "mean"),
    fully_acceptable_rate=("fully_acceptable", "mean"),
    median_rmse=("unclipped_rmse", "median"),
    mean_rmse=("unclipped_rmse", "mean"),
    maximum_rmse=("unclipped_rmse", "max"),
).reset_index()
condition_summary = []
for gesture in GESTURES:
    for state in STATES:
        x = df[(df.gesture == gesture) & (df.state == state)]
        condition_summary.append({
            "gesture": gesture, "state": state, "runs": len(x),
            "successful_runs": int(x.realised_all_features.sum()),
            "success_rate": x.realised_all_features.mean(),
            "fully_acceptable_runs": int(x.fully_acceptable.sum()),
            "median_rmse": x.unclipped_rmse.median(), "mean_rmse": x.unclipped_rmse.mean(),
        })
gesture_summary.to_csv(OUT / "final_gesture_summary.csv", index=False)
state_summary.to_csv(OUT / "final_state_summary.csv", index=False)
pd.DataFrame(condition_summary).to_csv(OUT / "final_condition_summary.csv", index=False)
feature_df.to_csv(OUT / "final_feature_error_summary.csv", index=False)
print({"runs": len(df), "successes": int(df.realised_all_features.sum()),
       "fully_acceptable": int(df.fully_acceptable.sum()),
       "median_rmse": float(df.unclipped_rmse.median()),
       "invalid": int((~df.valid_features).sum()),
       "out_of_range": int((df.normalisation_clipped_count > 0).sum())})
