# Live Gemini retest: wave -> friendly

Run from the project root in PowerShell. Use a new output directory because
the reward definition and result schema changed after the first pilot.

```powershell
$env:GOOGLE_API_KEY="YOUR_API_KEY"

python scripts/train_cem_contextual_bandit.py `
  --gesture wave `
  --target-state friendly `
  --evaluator gemini `
  --rounds 5 `
  --cem-samples-per-round 6 `
  --cem-elite-fraction 0.5 `
  --repeats 3 `
  --validation-repeats 10 `
  --validation-top-k 3 `
  --maxiter 45 `
  --popsize 5 `
  --local-maxiter 100 `
  --seed 41 `
  --temperature 0.2 `
  --realisation-penalty-weight 0.25 `
  --max-feature-error-threshold 0.10 `
  --max-feature-error-penalty-weight 0.50 `
  --wave-flow-target-weight 0.35 `
  --stability-penalty-weight 0.25 `
  --reward-margin-mode raw `
  --out outputs/live_retest_v2_wave_friendly_seed41
```

This uses a soft realisability rule. Do not add
`--reject-excessive-feature-error` for this experiment; retaining the soft
penalty lets the results expose the reachable feature boundary rather than
hiding failed targets.

The run performs 90 training evaluator calls and 50 independent validation
calls: 10 for the informed initial profile, 10 for the final CEM mean, and 10
for each of the top three sampled candidates. The final sampled profile is
selected by those fresh validation results, not its training reward.

Important outputs:

- `training_history.csv`: includes maximum feature error, feature acceptance,
  classification, agreement and entropy per round.
- `independent_validation.json`: includes all shortlisted validations and the
  final re-ranking decision.
- `selected_validated_profile.json`: the independently selected candidate.
- `results_summary.json`: configuration and complete result summary.

Rerun the identical command without `--overwrite` to resume a checkpoint.
