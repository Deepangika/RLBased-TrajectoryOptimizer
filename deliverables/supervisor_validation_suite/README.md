# Supervisor validation suite

Copy the two supplied Python files into the project's `scripts` directory,
preserving the `debug/evaluate_fixed_profile_holdout.py` script already present
in the current codebase.

## Recommended six-context run

From the project root in PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
python scripts/run_supervisor_validation_suite.py `
  --preset supervisor `
  --seed 41 `
  --out outputs/supervisor_validation_seed41
```

The default panel is:

- wave--friendly and wave--calm;
- reach--confident and reach--angry; and
- point--hesitant and point--confused.

It covers all six affective labels exactly once and each gesture exactly twice.
The planned protocol uses five CEM rounds, six profiles per round, three
training judgements per profile, ten independent judgements for each of the
top-three candidates plus the initial/final-mean diagnostics, and twenty fresh
holdout judgements of the frozen selected profile. This is approximately 960
evaluator calls for the six-context panel.

The runner is resumable. If it is interrupted, issue the same command again.
Completed training and holdout files are reused. Use `--force` only when you
intentionally want to replace completed results.

## Faster rehearsal

To check paths, credentials, and presentation output using three contexts:

```powershell
python scripts/run_supervisor_validation_suite.py `
  --preset quick `
  --seed 41 `
  --out outputs/supervisor_quick_seed41
```

Do not combine the quick and six-context runs in the same output directory.

## Rebuild the report without evaluator calls

```powershell
python scripts/run_supervisor_validation_suite.py `
  --preset supervisor `
  --seed 41 `
  --out outputs/supervisor_validation_seed41 `
  --summarize-only
```

## Saved evidence

Each context directory contains the existing implementation's complete
evidence, including:

- `results_summary.json` and `independent_validation.json`;
- `best_profile.json` and the CEM checkpoint;
- reward, Laban-policy, and realisation-error curves;
- per-candidate requested/achieved feature comparisons;
- joint/path comparisons and the rendered motion videos;
- `fixed_profile_holdout.json`; and
- the frozen holdout trajectory/video under `holdout/`.

The shared `supervisor_summary/` directory contains:

- `SUPERVISOR_REPORT.md`;
- `supervisor_results.json`;
- `context_metrics.csv`;
- target-probability and holdout-classification plots;
- feature-RMSE and maximum-error plots; and
- a gesture--state classification matrix.

Headline success metrics use only the frozen-profile holdout. A context passes
perceptually when its target classification rate is at least 0.70 and its mean
margin is positive. It passes feature realisation when RMSE and the worst
individual feature error are both at most 0.10. Full success additionally
requires physical acceptance.

The suite should be presented as an honest system validation. A failed context
is retained in the report and can reveal a gesture/affect representation or
evaluator limitation even when the inner trajectory optimiser is accurate.
