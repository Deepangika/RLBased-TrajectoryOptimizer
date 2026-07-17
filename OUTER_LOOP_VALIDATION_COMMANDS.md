# Corrected optimiser and CEM: validation commands

Run every command from the project root:

```powershell
cd path\to\laban_rl_pipeline_refactor
```

## 1. Install runtime dependencies

```powershell
python -m pip install -U numpy scipy pandas matplotlib imageio imageio-ffmpeg pydantic google-genai
```

PyTorch is not required by the corrected CEM training script.

For live Gemini evaluation, set the API key in the same PowerShell session:

```powershell
$env:GOOGLE_API_KEY="YOUR_API_KEY"
```

## 2. Check that the corrected code imports

```powershell
python -m compileall -q src scripts
```

## 3. Run the offline outer-loop validation suite

This uses a known synthetic objective and fake optimiser results. It makes no API calls.

```powershell
python scripts/debug/validate_outer_loop_fixes.py
```

Expected output:

- exact checkpoint/resume agreement;
- invalid paths and joint-limit violations rejected;
- no evaluator calls for rejected trajectories;
- unclipped realisation RMSE used;
- historical elites removed from the next CEM update;
- 100% improvement on the deterministic synthetic objective;
- approximately 97% improvement under moderate synthetic noise for the fixed 100-seed test.

Results are written to:

```text
outputs/outer_loop_fix_validation/validation_results.json
```

## 4. Run an actual-inner-optimiser mock smoke test

This exercises the complete CEM-to-inner-optimiser-to-validity-to-reward pipeline without Gemini.

```powershell
python scripts/train_cem_contextual_bandit.py `
  --gesture wave `
  --target-state friendly `
  --evaluator mock `
  --rounds 3 `
  --cem-samples-per-round 6 `
  --cem-elite-fraction 0.5 `
  --repeats 3 `
  --validation-repeats 5 `
  --maxiter 15 `
  --popsize 4 `
  --local-maxiter 50 `
  --seed 41 `
  --out outputs/outer_validation_wave_friendly_mock `
  --overwrite
```

This run independently evaluates the informed initial profile, final distribution mean, and best sampled profile.

## 5. Run the live Gemini pilot

Start with this smaller pilot before launching the full benchmark:

```powershell
python scripts/train_cem_contextual_bandit.py `
  --gesture wave `
  --target-state friendly `
  --evaluator gemini `
  --rounds 5 `
  --cem-samples-per-round 6 `
  --cem-elite-fraction 0.5 `
  --repeats 3 `
  --validation-repeats 5 `
  --maxiter 45 `
  --popsize 5 `
  --local-maxiter 100 `
  --seed 41 `
  --temperature 0.2 `
  --stability-penalty-weight 0.25 `
  --out outputs/corrected_cem_wave_friendly_seed41
```

The saved output directory is automatically resumed if a corrected checkpoint exists. Run the same command again to resume. Do not add `--overwrite` when resuming.

## 6. Run a full-budget single-context experiment

```powershell
python scripts/train_cem_contextual_bandit.py `
  --gesture wave `
  --target-state friendly `
  --evaluator gemini `
  --rounds 15 `
  --cem-samples-per-round 10 `
  --cem-elite-fraction 0.4 `
  --cem-initial-width 0.15 `
  --cem-smoothing 0.7 `
  --cem-min-std 0.03 `
  --cem-min-elites 3 `
  --exploration-decay-rate 0.98 `
  --repeats 3 `
  --validation-repeats 5 `
  --maxiter 45 `
  --popsize 5 `
  --local-maxiter 100 `
  --seed 41 `
  --temperature 0.2 `
  --realisation-penalty-weight 0.25 `
  --stability-penalty-weight 0.25 `
  --reward-margin-mode raw `
  --out outputs/corrected_cem_wave_friendly_seed41
```

## 7. Multi-seed evaluation

Repeat the full-budget command with at least three seeds, for example:

```text
7, 41, 97
```

Use a different output directory for every seed:

```text
outputs/corrected_cem_wave_friendly_seed7
outputs/corrected_cem_wave_friendly_seed41
outputs/corrected_cem_wave_friendly_seed97
```

Do not use `--overwrite` unless intentionally restarting a run from the beginning.

## 8. Audit completed CEM results

The existing audit script reads the configured experiment folders in the script:

```powershell
python scripts/debug/audit_cem_outer_loop.py
```

For a new benchmark matrix, update `RUNS` at the top of that script or extend it to discover your new output folders.

## Important output files

Each completed CEM run contains:

```text
training_history.csv
results_summary.json
independent_validation.json
best_profile.json
latest_checkpoint.pt
reward_curve.png
target_probability_curve.png
realisation_rmse_curve.png
policy_<feature>_curve.png
rounds/
validation/
```

Use `independent_validation.json` when reporting the initial-profile, final-mean, and best-profile results. The training maximum alone should not be treated as final evidence because it may contain selection noise.
