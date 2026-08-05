# 2×2 Ablation Report — VAD Target Recalibration × Tightened Inner Smoothness

**Date:** 2026-08-05 · **Evaluator:** gemini-2.5-flash (temp 0.2) · **Scope:** descriptive,
machine-evaluator only. Nothing here is evidence about human affect perception.

## 1. Design

Two frozen factors, applied to the two Stage A contexts (beckon–fear, wave–sadness):

| Condition | Target recalibration | Tightened smoothness (smooth_weight 0.01→0.03) |
|---|---|---|
| A (archived) | – | – |
| B | ✓ | – |
| C | – | ✓ |
| D | ✓ | ✓ |

- **Frozen recalibrated targets** (p10–p90 clip of the archived pre-outer-learning baseline VAD
  envelope; `configs/ablations/recalibrated_targets_v1.json`):
  beckon–fear V .300 / A .755 / D .300; wave–sadness V .490 / A .400 / D .300.
- **Frozen smoothness** (44-run offline grid, zero Gemini;
  `configs/ablations/tightened_smoothness_v1.json`): smooth_weight = 0.03.
- Frozen live protocol identical to Condition A: 5 rounds × 8 samples, 3 elites, 2 candidate
  repeats, validation top-4 × 5 repeats, paired 2 × 5 repeats, seed 7, 110-call/run ceiling,
  660-call aggregate ledger.

## 2. Execution integrity

| Run | Calls | Outcome |
|---|---|---|
| bf-B | 83 | success |
| bf-C | 68 | success (stop: budget) |
| bf-D | 83 | success |
| ws-B | 92 | success |
| ws-C | 93 | success after 2 resumes (transient Google API outages; both failed attempts preserved as `stage_status_failed_attempt{1,2}.json`; 12 checkpointed candidates recovered; deterministic seeds intact) |
| ws-D | 91 | success |

Total new calls ≈ 510 ≤ 660. Every run ≤ 110. All covariances symmetric positive definite;
contexts evolved independently; factor isolation verified in each `run_manifest.json`
(`target_vad_recalibrated`, `extra_optimiser_overrides`). Candidate seeds matched across
conditions by construction.

## 3. Headline numbers as reported by the pipeline

`improvement_over_baseline` (selected validated reward − archived baseline reward):

| | A | B | C | D |
|---|---|---|---|---|
| beckon–fear | −0.096 | +0.002 | −0.168 | −0.008 |
| wave–sadness | −0.038 | +0.186 | −0.004 | +0.174 |

**These B/D values are confounded** (Section 4) and must not be quoted as improvement.

Paired preference vs fixed baseline (learned wins / 5): A: bf 5, ws 2 → B: 0, C: 0, D: 1 in
both contexts. (B/D paired prompts carry the recalibrated VAD lines, so they are not directly
comparable with A/C.)

Selected-motion RMS jerk: bf 7803 (A) → 1145 (B) / 1504 (C) / 1184 (D);
ws 224 (A) → 69 (B) / 76 (C) / 88 (D). All three new conditions eliminate the extreme jerk of
Condition A's selections; recalibration alone was sufficient.

Strictly feasible candidate proportion (of 40): bf .70/.65/.575/.65; ws .65/.725/.575/.65 —
the small feasibility cost of smooth_weight 0.03 matches the frozen grid prediction.

## 4. Yardstick confound and corrected comparison

The pipeline's `baseline_validation_reward` is the **archived** baseline reward, scored under
the **original** target. In B/D the learned candidates were scored under the **recalibrated**
target. The recalibrated targets sit inside the baseline's own observed VAD envelope (that is
how they were frozen), so the reward scale is systematically easier — the "+0.186" for ws-B is
mostly a yardstick change, not learning.

Cross-scoring every selection and the fixed baseline under **both** targets (reward =
1 − weighted L1; weights V .2 / A .4 / D .4; `docs/ablation_v1/cross_scored_rewards.json`):

| beckon–fear | r(original target) | r(recalibrated target) |
|---|---|---|
| fixed baseline | **0.886** | **0.992** |
| A selection | 0.796 | 0.914 |
| B selection | 0.820 | 0.938 |
| C selection | 0.732 | 0.850 |
| D selection | 0.804 | 0.886 |

| wave–sadness | r(original target) | r(recalibrated target) |
|---|---|---|
| fixed baseline | **0.686** | **0.884** |
| A selection | 0.658 | 0.856 |
| B selection | 0.684 | 0.882 |
| C selection | 0.682 | 0.880 |
| D selection | 0.676 | 0.874 |

Under a common yardstick, **no learned selection beats the fixed baseline in any cell**, and
the baseline is near-ceiling under the recalibrated targets by construction. With this
correction, absolute reward and paired preference finally **agree**: the outer learner has not
produced profiles the machine evaluator scores or prefers above the fixed baseline.

## 5. Factorial reading (descriptive)

- **Target recalibration** is the only factor that moves nominal reward improvement toward
  zero/positive — but Section 4 shows this is a property of the easier yardstick, not of
  better motion. Its genuine effect is on **selection behaviour**: recalibrated runs select
  low-jerk candidates (bf jerk −85%) because achievable targets no longer reward extreme
  distortion.
- **Tightened smoothness** barely changes reward (bf −0.041, ws +0.011 nominal), slightly
  reduces feasibility (−0.06 to −0.16), and is largely redundant for jerk once recalibration
  is present.
- **Interaction** terms are small relative to main effects on all metrics except the
  paired-vs-baseline fraction, where counts are too coarse (0–5) to interpret.

## 6. Gemini language coding

Distortion-term rate in paired `reasoning_summary` texts (mentions/observation, 10 obs per
condition; `docs/ablation_v1/ablation_extras.json`):

- beckon–fear: A 3.6, B 3.7, C 3.9, D 3.9 — unchanged. Caveat: for fear, Gemini frequently
  describes trembling/shakiness as *evidence of fear* ("consistent with high arousal and low
  dominance"), so the term count conflates distortion complaints with affect-consistent
  description; it cannot be read as a distortion measure for this context.
- wave–sadness: A 0.5, B 0.3, C 0.3, D 0.5 — low throughout; the gesture-distortion language
  prominent in Condition A's *search-phase* reasoning does not dominate paired texts in any
  condition.

## 7. Conclusions (machine-evaluator scope only)

1. The 2×2 executed cleanly within budget with verified factor isolation and honest failure
   handling (ws-C outage → checkpoint resume).
2. The apparent reward gains under recalibration are a yardstick artefact; cross-scored, the
   fixed affect-derived baseline remains unbeaten in all eight cells.
3. Both knobs successfully remove the extreme-jerk selections of Condition A — the main
   genuine behavioural change.
4. The metric disagreement of Condition A is resolved: with a common yardstick, absolute
   reward and paired preference agree that outer learning has not improved on the baseline.
5. Implication: with a target inside the baseline's own VAD envelope the baseline is
   near-optimal by construction, leaving almost no headroom for the outer learner. Future
   recalibration should not be derived from the baseline distribution it will be compared
   against, or the comparison must re-evaluate the baseline under the same target with live
   calls.

## 8. Artifacts

- Analysis: `docs/ablation_v1/ablation_analysis.json`, `ablation_extras.json`,
  `cross_scored_rewards.json`; plots in `docs/ablation_v1/plots/`.
- Raw runs: `outputs/experiments/ablation_{beckon_fear,wave_sadness}_{B,C,D}_v1/` (not
  committed); Condition A archives unchanged.
- Scripts: `scripts/evaluation/run_ablation.py`, `analyse_ablation_results.py`,
  `plot_ablation_results.py`.
