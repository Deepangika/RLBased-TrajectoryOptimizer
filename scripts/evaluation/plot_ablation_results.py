"""Plots and Gemini-language coding for the 2x2 ablation (descriptive only).

Reads the frozen Condition A archives plus the six live ablation runs and
produces:
- per-context 2x2 bar panels (reward improvement, feasibility, paired
  fractions, selected-motion RMS jerk, calls used);
- contrast bar charts (target effect, smoothness effect, interaction);
- mean-displacement and covariance summaries per condition;
- a distortion-language coding of every Gemini paired ``reasoning_summary``
  (simple term counting, reported per condition, no interpretation).

Zero Gemini calls. Outputs go to --out (default
outputs/experiments/ablation_analysis_live).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

RUN_DIRS = {
    "beckon_fear": {
        "A": "outputs/experiments/outer_learning_beckon_fear_live_flash_v1",
        "B": "outputs/experiments/ablation_beckon_fear_B_v1",
        "C": "outputs/experiments/ablation_beckon_fear_C_v1",
        "D": "outputs/experiments/ablation_beckon_fear_D_v1",
    },
    "wave_sadness": {
        "A": "outputs/experiments/outer_learning_wave_sadness_live_flash_v1",
        "B": "outputs/experiments/ablation_wave_sadness_B_v1",
        "C": "outputs/experiments/ablation_wave_sadness_C_v1",
        "D": "outputs/experiments/ablation_wave_sadness_D_v1",
    },
}
GESTURE_STATE = {"beckon_fear": ("beckon", "fear"), "wave_sadness": ("wave", "sadness")}
CONDITION_LABELS = {
    "A": "A: original",
    "B": "B: recal target",
    "C": "C: tight smooth",
    "D": "D: both",
}

DISTORTION_TERMS = [
    "trembl", "shak", "erratic", "jerk", "jitter", "unstable", "instabilit",
    "distort", "unnatural", "abrupt", "twitch", "spasm", "wobbl", "chaotic",
    "irregular", "stutter",
]


def context_dir(cond_dir: Path, key: str) -> Path:
    gesture, state = GESTURE_STATE[key]
    return cond_dir / "live_stage_a" / gesture / state


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def mean_displacement(ctx_dir: Path) -> dict:
    hist = np.genfromtxt(ctx_dir / "mean_history.csv", delimiter=",", names=True)
    rows = np.atleast_1d(hist)
    names = [n for n in rows.dtype.names if n not in ("round",)]
    first = np.array([rows[0][n] for n in names], dtype=float)
    last = np.array([rows[-1][n] for n in names], dtype=float)
    return {
        "rounds": int(len(rows)),
        "displacement_norm": float(np.linalg.norm(last - first)),
    }


def covariance_summary(ctx_dir: Path) -> dict:
    data = np.load(ctx_dir / "covariance_history.npz")
    keys = sorted(data.files)
    first, last = np.asarray(data[keys[0]]), np.asarray(data[keys[-1]])
    eig = np.linalg.eigvalsh(last)
    return {
        "initial_trace": float(np.trace(first)),
        "final_trace": float(np.trace(last)),
        "final_min_eig": float(eig.min()),
        "spd": bool(eig.min() > 0),
    }


def language_coding(ctx_dir: Path) -> dict:
    counts: Counter[str] = Counter()
    n_obs = 0
    for name in ("paired_learned_vs_baseline.json", "paired_learned_vs_reference.json"):
        path = ctx_dir / name
        if not path.exists():
            continue
        payload = load_json(path)
        for record in payload.get("records", []):
            for obs in record.get("observations", []):
                text = (obs.get("reasoning_summary") or "").lower()
                if not text:
                    continue
                n_obs += 1
                for term in DISTORTION_TERMS:
                    counts[term] += len(re.findall(term, text))
    total = sum(counts.values())
    return {
        "observations_with_text": n_obs,
        "distortion_term_mentions": total,
        "mentions_per_observation": (total / n_obs) if n_obs else None,
        "top_terms": counts.most_common(6),
    }


def bar_panel(ax, values: dict, title: str, fmt: str = "{:+.3f}") -> None:
    conds = ["A", "B", "C", "D"]
    ys = [values.get(c) for c in conds]
    xs = np.arange(len(conds))
    colors = ["#888888", "#1f77b4", "#ff7f0e", "#2ca02c"]
    plotted = [(x, y, c) for x, y, c in zip(xs, ys, colors) if y is not None]
    ax.bar([p[0] for p in plotted], [p[1] for p in plotted],
           color=[p[2] for p in plotted])
    for x, y, _ in plotted:
        ax.text(x, y, fmt.format(y), ha="center",
                va="bottom" if y >= 0 else "top", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(conds)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title(title, fontsize=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", default="outputs/experiments/ablation_analysis_live/ablation_analysis.json")
    parser.add_argument("--out", default="outputs/experiments/ablation_analysis_live")
    args = parser.parse_args()

    analysis = load_json(ROOT / args.analysis)
    out_dir = ROOT / args.out
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    extras: dict = {"contexts": {}}
    for key, conds in RUN_DIRS.items():
        block = analysis["contexts"][key]
        values = {m: block["contrasts"][m]["values"] for m in block["contrasts"]}

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        bar_panel(axes[0, 0], values["reward_improvement_vs_fixed_baseline"],
                  "Reward improvement vs fixed baseline")
        bar_panel(axes[0, 1], values["feasible_proportion"],
                  "Strictly feasible proportion", fmt="{:.2f}")
        bar_panel(axes[0, 2], values["selected_rms_jerk"],
                  "Selected-motion RMS jerk", fmt="{:.0f}")
        bar_panel(axes[1, 0], values["paired_vs_baseline_learned_fraction"],
                  "Paired vs baseline: learned wins", fmt="{:.1f}")
        bar_panel(axes[1, 1], values["paired_vs_reference_learned_fraction"],
                  "Paired vs reference: learned wins", fmt="{:.1f}")
        calls = {c: block["conditions"][c].get("gemini_calls_used") for c in conds}
        bar_panel(axes[1, 2], calls, "Gemini calls used", fmt="{:.0f}")
        fig.suptitle(f"{key} — 2x2 ablation (descriptive; machine evaluator only)")
        fig.tight_layout()
        fig.savefig(plots / f"{key}_condition_panel.png", dpi=150)
        plt.close(fig)

        # contrast chart
        metrics = [m for m, v in block["contrasts"].items() if v.get("status") == "complete"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.25
        xs = np.arange(len(metrics))
        for i, effect in enumerate(("target_recalibration_effect",
                                    "tightened_smoothness_effect", "interaction")):
            raw = [block["contrasts"][m][effect] for m in metrics]
            scale = [abs(block["contrasts"][m]["values"]["A"]) or 1.0 for m in metrics]
            ax.bar(xs + (i - 1) * width, np.array(raw) / np.array(scale), width,
                   label=effect.replace("_", " "))
        ax.set_xticks(xs)
        ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=7)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.set_ylabel("effect / |A| (scale-normalised)")
        ax.legend(fontsize=8)
        ax.set_title(f"{key} — factorial contrasts")
        fig.tight_layout()
        fig.savefig(plots / f"{key}_contrasts.png", dpi=150)
        plt.close(fig)

        ctx_extra = {}
        for cond, rel in conds.items():
            ctx_dir = context_dir(ROOT / rel, key)
            entry = {}
            try:
                entry["mean"] = mean_displacement(ctx_dir)
                entry["covariance"] = covariance_summary(ctx_dir)
            except Exception as exc:  # archived A layout may differ
                entry["history_error"] = str(exc)
            entry["language"] = language_coding(ctx_dir)
            stop = ctx_dir / "stopping_reason.json"
            if stop.exists():
                sr = load_json(stop)
                entry["stop_reason"] = sr.get("stopping_reason") or sr.get("reason")
            ctx_extra[cond] = entry
        extras["contexts"][key] = ctx_extra

        # language plot
        fig, ax = plt.subplots(figsize=(7, 4))
        rates = {c: (ctx_extra[c]["language"]["mentions_per_observation"] or 0.0)
                 for c in conds}
        bar_panel(ax, rates, f"{key} — distortion-term mentions per paired observation",
                  fmt="{:.2f}")
        fig.tight_layout()
        fig.savefig(plots / f"{key}_language_coding.png", dpi=150)
        plt.close(fig)

    with open(out_dir / "ablation_extras.json", "w", encoding="utf-8") as fh:
        json.dump(extras, fh, indent=2, default=str)
    print(out_dir / "ablation_extras.json")
    print(plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
