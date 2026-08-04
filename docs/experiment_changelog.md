# Experiment changelog

## 2026-02 — Live Gemini Stage A pilots (beckon–fear, wave–sadness) + offline diagnostics

Two live pilot runs with an identical frozen protocol (gemini-2.5-flash,
temperature 0.2, 5 rounds x 8 samples, 2 search repeats, top-k 4 x 5
validation repeats, 2 x 5 paired repeats, robust margin 0.08, hard 110-call
budget per run). Verified from saved artifacts by
`scripts/evaluation/analyse_live_pilot_diagnostics.py` (zero API calls):

| Context | Calls | Val reward (learned vs baseline) | Paired vs baseline | Paired vs reference | Corrected outcome (taxonomy v2) |
|---|---|---|---|---|---|
| beckon–fear | 79/110 | 0.790 vs 0.886 (−0.096) | 5/5 | 3/5 (1 neither) | `paired_only_improvement` |
| wave–sadness | 75/110 | 0.648 vs 0.686 (−0.038) | 2/5 (2 neither) | 1/5 | `no_perceptual_improvement` |

Diagnostic findings:

- Metric divergence in beckon–fear (absolute reward down, paired preference
  unanimous); metric agreement in wave–sadness (both negative).
- Reward regression driven by dominance error (beckon–fear) and valence
  error (wave–sadness).
- Learned motions carry much higher RMS joint jerk than baselines
  (beckon ~5x, wave ~3x); evaluator reasoning codes this as
  erratic/trembling — rewarded for fear, penalised for sadness.
- Shortlist seed-change survival was 2/4 in both runs.
- Legacy outcome `relative_preference_improvement` for wave–sadness was not
  supported by the 2/5 raw count; superseded by taxonomy v2
  (`corrected_stage_status.json` written beside each run's
  `stage_status.json`).

Status: balanced-matrix expansion paused pending re-examination of the
perceptual objective and the preservation/smoothness trade-off. All results
are machine-evaluator only; no human perceptual evidence exists.
