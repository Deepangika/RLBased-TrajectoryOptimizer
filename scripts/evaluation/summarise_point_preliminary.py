"""Consolidate point preliminary Stage A results across four affective states.

Reads the live run outputs for point-{anger,disgust,happiness,surprise},
builds one comparison table (learned vs re-evaluated fixed baseline vs
re-evaluated reference under a common target/scoring yardstick), and writes
JSON + markdown plus a summary plot into docs/point_preliminary_v1/.

All rewards are Gemini machine-evaluator scores, not human perceptual
evidence.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
STATES = ("anger", "disgust", "happiness", "surprise")
OUT_DOCS = REPO / "docs" / "point_preliminary_v1"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _feasibility(sample_history: Path) -> dict:
    total = feasible = 0
    failures: dict[str, int] = {}
    with sample_history.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            if str(row.get("feasible", "")).lower() == "true":
                feasible += 1
            else:
                reason = row.get("feasibility_reason") or row.get("status") or "infeasible"
                failures[reason] = failures.get(reason, 0) + 1
    return {"candidates_attempted": total, "strictly_feasible": feasible,
            "feasible_proportion": (feasible / total) if total else None,
            "failures_by_reason": failures}


def main() -> int:
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    rows = []
    for state in STATES:
        run = REPO / "outputs" / "experiments" / f"point_prelim_{state}_v1" / "live_stage_a"
        ctx_dir = run / "point" / state
        stage = _load(run / "stage_status.json")["contexts"][0]
        budget = _load(run / "gemini_call_budget.json")
        manifest = _load(ctx_dir / "run_manifest.json")
        reeval = _load(ctx_dir / "baseline_reevaluation.json")
        selected = _load(ctx_dir / "selected_validated_profile.json")
        paired = {}
        for name, key in (("paired_learned_vs_baseline", "vs_baseline"),
                          ("paired_learned_vs_reference", "vs_reference")):
            rec = _load(ctx_dir / f"{name}.json")["records"][0]
            paired[key] = {
                "learned_preference_rate": rec.get("styled_preference_rate"),
                "opponent_preference_rate": rec.get("reference_preference_rate"),
                "neither_rate": rec.get("neither_rate"),
                "choice_counts": rec.get("choice_counts"),
                "mean_confidence": rec.get("mean_confidence"),
            }
        feas = _feasibility(ctx_dir / "sample_history.csv")
        by_cat = budget.get("calls_by_category") or {}
        rows.append({
            "context": f"point-{state}",
            "canonical_target_vad": manifest.get("canonical_target_vad"),
            "effective_target_vad": manifest.get("effective_target_vad"),
            "target_vad_recalibrated": manifest.get("target_vad_recalibrated"),
            "learned_validation_reward": stage.get("selected_validation_reward"),
            "learned_mean_observed_vad": selected.get("mean_observed_vad"),
            "learned_max_abs_feature_error": selected.get("max_abs_feature_error"),
            "archived_baseline_reward": stage.get("baseline_validation_reward"),
            "reevaluated_baseline_reward": stage.get("reevaluated_baseline_reward"),
            "reevaluated_reference_reward": stage.get("reevaluated_reference_reward"),
            "reference_reeval_detail": reeval["candidates"].get("reference"),
            "improvement_over_archived_baseline": stage.get("improvement_over_baseline"),
            "improvement_over_reevaluated_baseline": stage.get("improvement_over_reevaluated_baseline"),
            "paired": paired,
            "feasibility": feas,
            "outcome": stage.get("outcome"),
            "selection_status": stage.get("selection_status"),
            "stopping_reason": stage.get("stopping_reason"),
            "comparison_status": stage.get("comparison_status"),
            "gemini_calls_by_category": by_cat,
            "gemini_calls_total": sum(by_cat.values()),
            "call_ceiling": 65,
        })

    consolidated = {"description": "Point preliminary Stage A consolidated results. "
                    "All rewards are Gemini machine-evaluator scores under the original "
                    "canonical VAD targets; not human perceptual evidence.",
                    "contexts": rows}
    (OUT_DOCS / "consolidated_results.json").write_text(
        json.dumps(consolidated, indent=2), encoding="utf-8")

    def _f(v, nd=3):
        return "—" if v is None else f"{v:+.{nd}f}" if isinstance(v, float) and v < 0 else (
            f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    lines = [
        "# Point preliminary Stage A — consolidated comparison",
        "",
        "All rewards are Gemini 2.5 Flash machine-evaluator scores under the original",
        "canonical VAD targets with strict feasibility 0.10 / robust 0.08. The",
        "re-evaluated baseline/reference columns use the same target and scoring as the",
        "learned candidate (common yardstick). These are not human perceptual results.",
        "",
        "| Context | Learned reward | Re-eval fixed baseline | Re-eval reference | Δ vs re-eval baseline | Paired vs baseline (L/O/N) | Paired vs reference (L/O/N) | Feasible/attempted | Calls (≤65) | Stop reason | Selection |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        pb, pr = r["paired"]["vs_baseline"]["choice_counts"], r["paired"]["vs_reference"]["choice_counts"]
        lines.append(
            "| {ctx} | {lr} | {rb} | {rr} | {d} | {pb} | {pr} | {fe}/{at} | {calls} | {stop} | {sel} |".format(
                ctx=r["context"], lr=_f(r["learned_validation_reward"]),
                rb=_f(r["reevaluated_baseline_reward"]),
                rr=("infeasible (−1.0)" if r["reevaluated_reference_reward"] == -1.0
                    else _f(r["reevaluated_reference_reward"])),
                d=_f(r["improvement_over_reevaluated_baseline"]),
                pb=f"{pb['styled']}/{pb['reference']}/{pb['neither']}",
                pr=f"{pr['styled']}/{pr['reference']}/{pr['neither']}",
                fe=r["feasibility"]["strictly_feasible"], at=r["feasibility"]["candidates_attempted"],
                calls=r["gemini_calls_total"], stop=r["stopping_reason"],
                sel=r["selection_status"]))
    lines += [
        "",
        "Paired columns are Learned/Opponent/Neither choice counts over 3 repeats.",
        "The reference motion fails the strict 0.10 feature-error gate under every",
        "styled target (max error 0.289), so its penalised reward is −1.0 by",
        "construction; its role here is a feasibility control, not a competitive",
        "comparator.",
    ]
    (OUT_DOCS / "consolidated_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(rows))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [r["learned_validation_reward"] for r in rows], w,
           label="Learned (validated)")
    ax.bar([i + w / 2 for i in x], [r["reevaluated_baseline_reward"] for r in rows], w,
           label="Fixed baseline (re-evaluated)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["context"] for r in rows])
    ax.set_ylabel("Gemini VAD reward (common yardstick)")
    ax.set_title("Point preliminary Stage A: learned vs re-evaluated fixed baseline")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DOCS / "learned_vs_baseline_rewards.png", dpi=150)

    for state in STATES:
        src = (REPO / "outputs" / "experiments" / f"point_prelim_{state}_v1" /
               "live_stage_a" / "point" / state / "final_motion.gif")
        if src.exists():
            shutil.copy2(src, OUT_DOCS / f"final_motion_{state}.gif")

    print(f"Wrote consolidated results to {OUT_DOCS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
