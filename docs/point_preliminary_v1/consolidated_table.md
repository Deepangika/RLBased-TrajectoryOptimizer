# Point preliminary Stage A — consolidated comparison

All rewards are Gemini 2.5 Flash machine-evaluator scores under the original
canonical VAD targets with strict feasibility 0.10 / robust 0.08. The
re-evaluated baseline/reference columns use the same target and scoring as the
learned candidate (common yardstick). These are not human perceptual results.

| Context | Learned reward | Re-eval fixed baseline | Re-eval reference | Δ vs re-eval baseline | Paired vs baseline (L/O/N) | Paired vs reference (L/O/N) | Feasible/attempted | Calls (≤65) | Stop reason | Selection |
|---|---|---|---|---|---|---|---|---|---|---|
| point-anger | 0.683 | 0.737 | infeasible (−1.0) | -0.054 | 0/2/1 | 3/0/0 | 11/18 | 37 | perceptual_success | selected |
| point-disgust | 0.827 | 0.870 | infeasible (−1.0) | -0.044 | 0/0/3 | 3/0/0 | 12/18 | 42 | perceptual_success | selected |
| point-happiness | 0.823 | 0.773 | infeasible (−1.0) | 0.050 | 1/1/1 | 2/1/0 | 13/18 | 44 | maximum_budget_reached | selected |
| point-surprise | 0.783 | 0.717 | infeasible (−1.0) | 0.067 | 0/0/3 | 0/1/2 | 14/18 | 46 | perceptual_success | selected |

Paired columns are Learned/Opponent/Neither choice counts over 3 repeats.
The reference motion fails the strict 0.10 feature-error gate under every
styled target (max error 0.289), so its penalised reward is −1.0 by
construction; its role here is a feasibility control, not a competitive
comparator.
