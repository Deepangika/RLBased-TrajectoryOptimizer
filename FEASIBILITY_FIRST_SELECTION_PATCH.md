# Feasibility-first selection patch

This patch changes final CEM selection from "highest validation reward among
top-K" to an incumbent-safe, feasibility-first choice over:

- the informed initial profile;
- the final CEM distribution mean; and
- the independently evaluated top-K sampled candidates.

A candidate is eligible only when it is a valid, physically acceptable
realisation with both RMSE and maximum per-feature error no greater than the
configured tolerance. If no candidate is eligible, the best diagnostic fallback
is saved with `strict_feasibility_satisfied=false`.

## Recover the completed clean seed-7 run

Preview changes without writing:

```powershell
python scripts/evaluation/recover_feasibility_first_selection.py `
  --root outputs/final_clean_evaluation_v1/outer `
  --dry-run
```

Apply the recovery (no Gemini calls are made):

```powershell
python scripts/evaluation/recover_feasibility_first_selection.py `
  --root outputs/final_clean_evaluation_v1/outer
```

The script creates `.pre_feasibility_selection.json` backups once, updates the
three selection/summary files consistently, and refuses to run if a frozen
holdout already exists.

## Validate

```powershell
python -m pytest tests/test_feasibility_first_selection.py `
  tests/test_comprehensive_runner_completion.py -q
```

After recovery, rerun the comprehensive seed-7 command with `--dry-run`. Every
completed context should be reported as `SKIP completed outer run`.
