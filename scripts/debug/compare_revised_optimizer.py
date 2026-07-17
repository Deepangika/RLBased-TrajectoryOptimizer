from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "robust_four_states_analysis" / "robust_four_states_120_runs.csv"
REVISED_DIR = ROOT / "outputs" / "revised_four_states_120"
OUT = ROOT / "outputs" / "revised_optimizer_comparison"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GESTURES = ["wave", "reach", "point"]
STATES = ["confident", "friendly", "calm", "confused"]
FEATURES = ["weight", "time", "flow_boundness", "space_indirectness", "shape_arcness"]
FLABELS = ["Weight", "Time", "Flow", "Space", "Shape"]


def read(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


baseline = read(BASE)
revised = []
for gesture in GESTURES:
    revised += read(REVISED_DIR / f"matrix_{gesture}.csv")

with (OUT / "revised_four_states_120_runs.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(revised[0])); writer.writeheader(); writer.writerows(revised)


def subset(rows, gesture, state):
    return [r for r in rows if r["gesture"] == gesture and r["state"] == state]


summary = []
before_success = np.zeros((3,4)); after_success = np.zeros((3,4))
before_clip = np.zeros((3,4)); after_clip = np.zeros((3,4))
before_rmse = np.zeros((3,4)); after_rmse = np.zeros((3,4))
before_feat = np.zeros((3,4,5)); after_feat = np.zeros((3,4,5))

for gi,g in enumerate(GESTURES):
    for si,s in enumerate(STATES):
        b = subset(baseline,g,s); a = subset(revised,g,s)
        bv = [r for r in b if r["valid_features"]=="True"]
        av = [r for r in a if r["valid_features"]=="True"]
        before_success[gi,si] = np.mean([r["realised_all_features"]=="True" for r in b])
        after_success[gi,si] = np.mean([r["realised_all_features"]=="True" for r in a])
        before_clip[gi,si] = np.mean([int(r["normalisation_clipped_count"])>0 for r in b])
        after_clip[gi,si] = np.mean([int(r["normalisation_clipped_count"])>0 for r in a])
        before_rmse[gi,si] = np.median([float(r["unclipped_rmse"]) for r in bv])
        after_rmse[gi,si] = np.median([float(r["unclipped_rmse"]) for r in av])
        before_feat[gi,si] = np.median([[float(r[f"unclipped_error_{k}"]) for k in FEATURES] for r in bv],axis=0)
        after_feat[gi,si] = np.median([[float(r[f"unclipped_error_{k}"]) for k in FEATURES] for r in av],axis=0)
        summary.append({
            "gesture":g,"state":s,
            "baseline_successes":int(before_success[gi,si]*10),
            "revised_successes":int(after_success[gi,si]*10),
            "success_change_percentage_points":100*(after_success[gi,si]-before_success[gi,si]),
            "baseline_median_rmse":before_rmse[gi,si],
            "revised_median_rmse":after_rmse[gi,si],
            "baseline_clipped_runs":int(before_clip[gi,si]*10),
            "revised_clipped_runs":int(after_clip[gi,si]*10),
            **{f"baseline_median_error_{k}":before_feat[gi,si,fi] for fi,k in enumerate(FEATURES)},
            **{f"revised_median_error_{k}":after_feat[gi,si,fi] for fi,k in enumerate(FEATURES)},
        })

with (OUT / "before_after_summary.csv").open("w",newline="",encoding="utf-8") as handle:
    writer=csv.DictWriter(handle,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)


def annotate(ax,data,fmt):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j,i,fmt(data[i,j]),ha="center",va="center",fontweight="bold",fontsize=9,
                    color="white" if data[i,j]>.55 else "#172033")


# Paired success heatmaps.
fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
for ax,data,title in zip(axes,[before_success,after_success],["Baseline","Revised optimiser"]):
    im=ax.imshow(data,vmin=0,vmax=1,cmap="YlGn",aspect="auto");annotate(ax,data,lambda x:f"{x:.0%}")
    ax.set_xticks(range(4),[s.title() for s in STATES]);ax.set_yticks(range(3),[g.title() for g in GESTURES]);ax.set_title(title,fontweight="bold")
fig.colorbar(im,ax=axes,fraction=.025,pad=.03,label="Successful run proportion")
fig.suptitle("Target realisation reliability before and after stabilisation",x=.06,ha="left",fontsize=14,fontweight="bold")
fig.subplots_adjust(left=.08,right=.9,bottom=.13,top=.82,wspace=.18);fig.savefig(FIG/"fig1_success_before_after.png",dpi=240,bbox_inches="tight");plt.close(fig)

# Success improvement in percentage points.
delta=100*(after_success-before_success)
fig,ax=plt.subplots(figsize=(8.8,4.2));im=ax.imshow(delta,cmap="BrBG",vmin=-100,vmax=100,aspect="auto")
ax.set_xticks(range(4),[s.title() for s in STATES]);ax.set_yticks(range(3),[g.title() for g in GESTURES]);ax.set_title("Change in target realisation rate",loc="left",fontsize=14,fontweight="bold",pad=12)
for i in range(3):
    for j in range(4):ax.text(j,i,f"{delta[i,j]:+.0f} pp",ha="center",va="center",fontweight="bold")
fig.colorbar(im,ax=ax,fraction=.04,pad=.03,label="Percentage-point change");ax.spines[:].set_visible(False);fig.tight_layout();fig.savefig(FIG/"fig2_success_improvement.png",dpi=240,bbox_inches="tight");plt.close(fig)

# Median RMSE paired dot chart on log scale.
fig,axes=plt.subplots(1,3,figsize=(12.5,4.5),sharey=True)
x=np.arange(4)
for gi,(ax,g) in enumerate(zip(axes,GESTURES)):
    b=np.maximum(before_rmse[gi],1e-7);a=np.maximum(after_rmse[gi],1e-7)
    for j in range(4):ax.plot([j,j],[b[j],a[j]],color="#9ca3af",linewidth=1.3)
    ax.scatter(x,b,label="Baseline",color="#d95f02",s=52);ax.scatter(x,a,label="Revised",color="#1b9e77",s=52)
    ax.set_yscale("log");ax.axhline(.1,color="#b22222",linestyle="--",linewidth=1);ax.set_xticks(x,[s.title() for s in STATES],rotation=24);ax.set_title(g.title(),fontweight="bold");ax.grid(axis="y",alpha=.2)
axes[0].set_ylabel("Median unclipped RMSE (log scale)");axes[0].legend(frameon=False,fontsize=8)
fig.suptitle("Median target error before and after stabilisation",x=.06,ha="left",fontsize=14,fontweight="bold");fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(FIG/"fig3_rmse_before_after.png",dpi=240,bbox_inches="tight");plt.close(fig)

# Overall per-feature median error before/after by gesture.
b_over=np.median(before_feat,axis=1);a_over=np.median(after_feat,axis=1)
fig,axes=plt.subplots(1,3,figsize=(12.5,4.2),sharey=True)
width=.36
for gi,(ax,g) in enumerate(zip(axes,GESTURES)):
    ax.bar(x:=np.arange(5)-width/2,b_over[gi],width,label="Baseline",color="#d95f02",alpha=.8)
    ax.bar(np.arange(5)+width/2,a_over[gi],width,label="Revised",color="#1b9e77",alpha=.85)
    ax.set_xticks(range(5),FLABELS,rotation=25);ax.set_title(g.title(),fontweight="bold");ax.grid(axis="y",alpha=.2)
axes[0].set_ylabel("Median absolute feature error");axes[0].legend(frameon=False,fontsize=8)
fig.suptitle("Per-feature error reduction across the four states",x=.06,ha="left",fontsize=14,fontweight="bold");fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(FIG/"fig4_feature_error_before_after.png",dpi=240,bbox_inches="tight");plt.close(fig)

# Clipping paired heatmaps.
fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
for ax,data,title in zip(axes,[before_clip,after_clip],["Baseline","Revised optimiser"]):
    im=ax.imshow(data,vmin=0,vmax=1,cmap="OrRd",aspect="auto");annotate(ax,data,lambda x:f"{x:.0%}")
    ax.set_xticks(range(4),[s.title() for s in STATES]);ax.set_yticks(range(3),[g.title() for g in GESTURES]);ax.set_title(title,fontweight="bold")
fig.colorbar(im,ax=axes,fraction=.025,pad=.03,label="Runs with ≥1 out-of-range feature")
fig.suptitle("Out-of-range feature saturation before and after stabilisation",x=.06,ha="left",fontsize=14,fontweight="bold")
fig.subplots_adjust(left=.08,right=.9,bottom=.13,top=.82,wspace=.18);fig.savefig(FIG/"fig5_clipping_before_after.png",dpi=240,bbox_inches="tight");plt.close(fig)

overall={
    "baseline_successes":sum(r["realised_all_features"]=="True" for r in baseline),
    "revised_successes":sum(r["realised_all_features"]=="True" for r in revised),
    "baseline_fully_acceptable":sum(r["fully_acceptable"]=="True" for r in baseline),
    "revised_fully_acceptable":sum(r["fully_acceptable"]=="True" for r in revised),
    "baseline_clipped_runs":sum(int(r["normalisation_clipped_count"])>0 for r in baseline),
    "revised_clipped_runs":sum(int(r["normalisation_clipped_count"])>0 for r in revised),
    "baseline_invalid_runs":sum(r["valid_features"]!="True" for r in baseline),
    "revised_invalid_runs":sum(r["valid_features"]!="True" for r in revised),
}
(OUT/"overall_before_after.json").write_text(json.dumps(overall,indent=2),encoding="utf-8")
print(json.dumps(overall,indent=2))
