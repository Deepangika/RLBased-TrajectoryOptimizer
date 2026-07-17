from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs" / "full_realisability_matrix"
NEW = ROOT / "outputs" / "robust_four_states_extra"
REGION = ROOT / "outputs" / "robust_four_states_region"
OUT = ROOT / "outputs" / "robust_four_states_analysis"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GESTURES = ["wave", "reach", "point"]
STATES = ["confident", "friendly", "calm", "confused"]
FEATURES = ["weight", "time", "flow_boundness", "space_indirectness", "shape_arcness"]
FLABELS = ["Weight", "Time", "Flow", "Space", "Shape"]
COLORS = {"wave": "#1f77b4", "reach": "#f28e2b", "point": "#2ca02c"}


def read(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


rows = []
for gesture in GESTURES:
    rows += [r for r in read(OLD / f"matrix_{gesture}.csv") if r["state"] in STATES]
    rows += read(NEW / f"matrix_{gesture}.csv")

combined_path = OUT / "robust_four_states_120_runs.csv"
with combined_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)


def group(gesture, state):
    return [r for r in rows if r["gesture"] == gesture and r["state"] == state]


summary = []
success = np.zeros((3, 4))
clip_rate = np.zeros((3, 4))
median_rmse = np.zeros((3, 4))
feature_median = np.zeros((3, 4, 5))
feature_success_median = np.full((3, 4, 5), np.nan)

for gi, gesture in enumerate(GESTURES):
    for si, state in enumerate(STATES):
        gg = group(gesture, state)
        valid = [r for r in gg if r["valid_features"] == "True"]
        succeeded = [r for r in valid if r["realised_all_features"] == "True"]
        rmses = np.array([float(r["unclipped_rmse"]) for r in valid])
        success[gi, si] = len(succeeded) / len(gg)
        clip_rate[gi, si] = np.mean([int(r["normalisation_clipped_count"]) > 0 for r in gg])
        median_rmse[gi, si] = np.median(rmses) if len(rmses) else np.nan
        all_errors = np.array([[float(r[f"unclipped_error_{k}"]) for k in FEATURES] for r in valid])
        feature_median[gi, si] = np.median(all_errors, axis=0)
        if succeeded:
            succ_errors = np.array([[float(r[f"unclipped_error_{k}"]) for k in FEATURES] for r in succeeded])
            feature_success_median[gi, si] = np.median(succ_errors, axis=0)
        summary.append({
            "gesture": gesture,
            "state": state,
            "successful_runs": len(succeeded),
            "runs": len(gg),
            "success_rate": len(succeeded) / len(gg),
            "invalid_runs": len(gg) - len(valid),
            "runs_with_clipping": sum(int(r["normalisation_clipped_count"]) > 0 for r in gg),
            "median_rmse": float(np.median(rmses)) if len(rmses) else np.nan,
            "q1_rmse": float(np.quantile(rmses, .25)) if len(rmses) else np.nan,
            "q3_rmse": float(np.quantile(rmses, .75)) if len(rmses) else np.nan,
            "best_rmse": float(np.min(rmses)) if len(rmses) else np.nan,
            **{f"median_error_{k}": float(np.median(all_errors[:, fi])) for fi, k in enumerate(FEATURES)},
            **{f"successful_median_error_{k}": float(feature_success_median[gi, si, fi]) for fi, k in enumerate(FEATURES)},
        })

summary_path = OUT / "robust_four_states_summary.csv"
with summary_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader(); writer.writerows(summary)


def heatmap(data, title, filename, fmt="{:.0%}", cmap="YlGn", vmin=0, vmax=1, cbar=""):
    fig, ax = plt.subplots(figsize=(8.7, 4.2))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(4), [s.title() for s in STATES])
    ax.set_yticks(range(3), [g.title() for g in GESTURES])
    ax.set_title(title, loc="left", fontweight="bold", fontsize=14, pad=13)
    midpoint = (vmin + vmax) / 2
    for i in range(3):
        for j in range(4):
            ax.text(j, i, fmt.format(data[i,j]), ha="center", va="center", fontweight="bold",
                    color="white" if data[i,j] > midpoint else "#172033")
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.03); cb.set_label(cbar)
    ax.spines[:].set_visible(False); fig.tight_layout()
    fig.savefig(FIG / filename, dpi=240, bbox_inches="tight"); plt.close(fig)


heatmap(success, "Target realisation rate across ten random seeds", "fig1_success_rate_10_seeds.png", cbar="Successful run proportion")
heatmap(clip_rate, "Frequency of out-of-range feature saturation", "fig2_clipping_rate_10_seeds.png", cmap="OrRd", cbar="Runs with ≥1 out-of-range feature")

# RMSE distributions; logarithmic axis keeps both exact and extreme failures visible.
fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True)
for ax, gesture in zip(axes, GESTURES):
    data = []
    for state in STATES:
        vals = [max(float(r["unclipped_rmse"]), 1e-6) for r in group(gesture, state) if r["valid_features"] == "True"]
        data.append(vals)
    bp = ax.boxplot(data, patch_artist=True, widths=.62, showfliers=True)
    for box in bp["boxes"]: box.set_facecolor(COLORS[gesture]); box.set_alpha(.65)
    ax.set_yscale("log"); ax.axhline(.10, color="#b22222", linestyle="--", linewidth=1.3, label="0.10 tolerance")
    ax.set_xticks(range(1,5), [s.title() for s in STATES], rotation=24)
    ax.set_title(gesture.title(), fontweight="bold"); ax.grid(axis="y", alpha=.25)
axes[0].set_ylabel("Unclipped feature RMSE (log scale)")
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Final target error across ten optimisation seeds", x=.07, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(FIG / "fig3_rmse_boxplots.png", dpi=240, bbox_inches="tight"); plt.close(fig)

# Per-feature median error panels.
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
for gi, (ax, gesture) in enumerate(zip(axes, GESTURES)):
    im = ax.imshow(feature_median[gi], cmap="Blues", vmin=0, vmax=max(.5, np.nanpercentile(feature_median, 85)), aspect="auto")
    ax.set_xticks(range(5), FLABELS, rotation=28)
    ax.set_yticks(range(4), [s.title() for s in STATES])
    ax.set_title(gesture.title(), fontweight="bold")
    for i in range(4):
        for j in range(5): ax.text(j, i, f"{feature_median[gi,i,j]:.2f}", ha="center", va="center", fontsize=8, fontweight="bold")
fig.colorbar(im, ax=axes, fraction=.018, pad=.055, label="Median absolute error")
fig.suptitle("Median final per-feature error across ten seeds", x=.06, ha="left", fontsize=14, fontweight="bold")
fig.subplots_adjust(left=.08, right=.86, bottom=.18, top=.84, wspace=.22)
fig.savefig(FIG / "fig4_per_feature_error.png", dpi=240, bbox_inches="tight"); plt.close(fig)

# Achievable-region samples from two local scales.
region = {}
region_summary = []
targets = {}
for state in STATES:
    exemplar = next(r for r in rows if r["state"] == state)
    targets[state] = np.array([float(exemplar[f"target_{k}"]) for k in FEATURES])

for gesture in GESTURES:
    rr = []
    for sigma in [0.03, 0.07]:
        rr += read(REGION / f"achievable_region_{gesture}_sigma_{sigma:.2f}.csv")
    compatible = [r for r in rr if r["preservation_compatible"] == "True"]
    arr = np.array([[float(r[f"unclipped_{k}"]) for k in FEATURES] for r in compatible])
    region[gesture] = arr
    for fi, key in enumerate(FEATURES):
        region_summary.append({
            "gesture": gesture, "feature": key, "n_compatible": len(arr),
            "q02_5": float(np.quantile(arr[:,fi], .025)), "median": float(np.median(arr[:,fi])),
            "q97_5": float(np.quantile(arr[:,fi], .975)), "min": float(np.min(arr[:,fi])), "max": float(np.max(arr[:,fi])),
        })

with (OUT / "achievable_region_quantiles.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(region_summary[0])); writer.writeheader(); writer.writerows(region_summary)

# Marginal 95% feature bands with target overlays.
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
x = np.arange(5)
markers = ["o", "s", "^", "D"]
for ax, gesture in zip(axes, GESTURES):
    arr = region[gesture]
    lo, med, hi = np.quantile(arr, [.025,.5,.975], axis=0)
    ax.fill_between(x, lo, hi, color=COLORS[gesture], alpha=.22, label="Sampled 95% marginal range")
    ax.plot(x, med, color=COLORS[gesture], linewidth=2, label="Sample median")
    for marker, state in zip(markers, STATES):
        ax.scatter(x, targets[state], s=24, marker=marker, label=state.title())
    ax.set_xticks(x, FLABELS, rotation=25); ax.set_title(f"{gesture.title()} (n={len(arr):,})", fontweight="bold")
    ax.axhspan(0,1,color="#f4f5f7",zorder=-5); ax.grid(axis="y",alpha=.2)
axes[0].set_ylabel("Unclipped normalised feature value")
axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
fig.suptitle("Approximate preservation-compatible feature region", x=.06, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(FIG / "fig5_achievable_region_bands.png", dpi=240, bbox_inches="tight"); plt.close(fig)

# Joint feature region via a common PCA projection.
all_region = np.vstack([region[g] for g in GESTURES])
mean = all_region.mean(axis=0); scale = all_region.std(axis=0) + 1e-8
z = (all_region - mean) / scale
_, _, vt = np.linalg.svd(z, full_matrices=False)
components = vt[:2]
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True, sharey=True)
for ax, gesture in zip(axes, GESTURES):
    proj = ((region[gesture] - mean) / scale) @ components.T
    ax.scatter(proj[:,0], proj[:,1], s=5, alpha=.18, color=COLORS[gesture], rasterized=True)
    for marker, state in zip(markers, STATES):
        tp = ((targets[state] - mean) / scale) @ components.T
        ax.scatter(tp[0], tp[1], s=55, marker=marker, edgecolor="black", linewidth=.5, label=state.title())
    ax.set_title(gesture.title(), fontweight="bold"); ax.grid(alpha=.2); ax.set_xlabel("Feature-space PC1")
axes[0].set_ylabel("Feature-space PC2"); axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("Joint structure of the sampled achievable feature region", x=.06, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0,0,1,.94]); fig.savefig(FIG / "fig6_achievable_region_pca.png", dpi=240, bbox_inches="tight"); plt.close(fig)

overall = {
    "runs": len(rows),
    "successful": sum(r["realised_all_features"] == "True" for r in rows),
    "invalid": sum(r["valid_features"] != "True" for r in rows),
    "with_clipping": sum(int(r["normalisation_clipped_count"]) > 0 for r in rows),
    "region_compatible_counts": {g: len(region[g]) for g in GESTURES},
}
(OUT / "robust_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
print(json.dumps(overall, indent=2))
print(summary_path)
