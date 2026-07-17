from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs" / "full_realisability_matrix"
FIG = DATA / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GESTURES = ["wave", "reach", "point"]
STATES = ["confident", "calm", "hesitant", "friendly", "confused", "angry"]
FEATURES = ["weight", "time", "flow_boundness", "space_indirectness", "shape_arcness"]
LABELS = ["Weight", "Time", "Flow", "Space", "Shape"]

rows = []
for gesture in GESTURES:
    with (DATA / f"matrix_{gesture}.csv").open(encoding="utf-8") as handle:
        rows.extend(list(csv.DictReader(handle)))

combined = DATA / "full_realisability_matrix.csv"
with combined.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def subset(gesture, state):
    return [r for r in rows if r["gesture"] == gesture and r["state"] == state]


summary_rows = []
success = np.zeros((len(GESTURES), len(STATES)))
median_rmse = np.zeros_like(success)
clip_rate = np.zeros_like(success)
acceptable = np.zeros_like(success)

for gi, gesture in enumerate(GESTURES):
    for si, state in enumerate(STATES):
        group = subset(gesture, state)
        realised = np.array([r["realised_all_features"] == "True" for r in group])
        full = np.array([r["fully_acceptable"] == "True" for r in group])
        rmses = np.array([float(r["unclipped_rmse"]) for r in group])
        clipped = np.array([int(r["normalisation_clipped_count"]) for r in group])
        success[gi, si] = realised.mean()
        acceptable[gi, si] = full.mean()
        median_rmse[gi, si] = np.median(rmses)
        clip_rate[gi, si] = np.mean(clipped > 0)
        feature_mae = {
            f"mae_{key}": float(np.mean([float(r[f"unclipped_error_{key}"]) for r in group]))
            for key in FEATURES
        }
        best = min(group, key=lambda r: float(r["unclipped_rmse"]))
        summary_rows.append({
            "gesture": gesture,
            "state": state,
            "realised_runs": int(realised.sum()),
            "runs": len(group),
            "success_rate": float(realised.mean()),
            "fully_acceptable_rate": float(full.mean()),
            "median_unclipped_rmse": float(np.median(rmses)),
            "mean_unclipped_rmse": float(np.mean(rmses)),
            "best_unclipped_rmse": float(best["unclipped_rmse"]),
            "best_seed": int(best["seed"]),
            "runs_with_clipping": int(np.sum(clipped > 0)),
            "mean_path_length_ratio": float(np.mean([float(r["path_length_ratio"]) for r in group])),
            **feature_mae,
        })

summary_csv = DATA / "realisability_summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
    writer.writeheader()
    writer.writerows(summary_rows)


def heatmap(matrix, title, cbar_label, filename, fmt, vmin=0, vmax=1, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(10.2, 4.3))
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(STATES)), [s.title() for s in STATES])
    ax.set_yticks(range(len(GESTURES)), [g.title() for g in GESTURES])
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=14)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > (vmin + vmax) / 2 else "#172033"
            ax.text(j, i, fmt.format(value), ha="center", va="center", color=color, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(cbar_label)
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


heatmap(success, "Target realisation reliability across five seeds", "Successful run proportion", "fig1_success_rate_heatmap.png", "{:.0%}", cmap="YlGn")
heatmap(median_rmse, "Median target error using unclipped normalised features", "Median RMSE", "fig2_median_rmse_heatmap.png", "{:.3f}", vmax=max(1.0, float(np.nanpercentile(median_rmse, 90))), cmap="magma_r")
heatmap(clip_rate, "Frequency of out-of-range feature saturation", "Runs with ≥1 clipped feature", "fig3_clipping_rate_heatmap.png", "{:.0%}", cmap="OrRd")

# Mean per-feature absolute error, grouped by gesture across all states/seeds.
feature_matrix = np.zeros((len(GESTURES), len(FEATURES)))
for gi, gesture in enumerate(GESTURES):
    group = [r for r in rows if r["gesture"] == gesture]
    for fi, feature in enumerate(FEATURES):
        feature_matrix[gi, fi] = np.mean([float(r[f"unclipped_error_{feature}"]) for r in group])

fig, ax = plt.subplots(figsize=(9.2, 4.3))
im = ax.imshow(feature_matrix, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(FEATURES)), LABELS)
ax.set_yticks(range(len(GESTURES)), [g.title() for g in GESTURES])
ax.set_title("Mean absolute target error by feature", loc="left", fontsize=14, fontweight="bold", pad=14)
for i in range(feature_matrix.shape[0]):
    for j in range(feature_matrix.shape[1]):
        ax.text(j, i, f"{feature_matrix[i,j]:.3f}", ha="center", va="center", color="#172033", fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="Mean absolute error")
ax.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "fig4_feature_error_heatmap.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# Best achieved versus target profile for each gesture/state.
fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharey=True)
x = np.arange(len(FEATURES))
for ax, state in zip(axes.flat, STATES):
    width = 0.22
    target = np.array([float(subset("wave", state)[0][f"target_{k}"]) for k in FEATURES])
    ax.plot(x, target, color="#111827", marker="o", linewidth=2.2, label="Target")
    for gi, gesture in enumerate(GESTURES):
        best = min(subset(gesture, state), key=lambda r: float(r["unclipped_rmse"]))
        vals = np.array([float(best[f"unclipped_{k}"]) for k in FEATURES])
        ax.plot(x, vals, marker="o", linewidth=1.6, label=gesture.title())
    ax.set_title(state.title(), loc="left", fontweight="bold")
    ax.set_xticks(x, LABELS, rotation=20)
    ax.axhspan(0, 1, color="#f3f4f6", zorder=-5)
    ax.set_ylim(-0.08, 1.15)
    ax.grid(axis="y", alpha=0.25)
axes[0, 0].legend(ncol=2, frameon=False, fontsize=9)
fig.suptitle("Best target match found for each gesture–state combination", x=0.08, ha="left", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIG / "fig5_best_profiles.png", dpi=220, bbox_inches="tight")
plt.close(fig)

overall = {
    "total_runs": len(rows),
    "realised_runs": sum(r["realised_all_features"] == "True" for r in rows),
    "fully_acceptable_runs": sum(r["fully_acceptable"] == "True" for r in rows),
    "runs_with_clipping": sum(int(r["normalisation_clipped_count"]) > 0 for r in rows),
}
(DATA / "overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
print(json.dumps(overall, indent=2))
print(summary_csv)
