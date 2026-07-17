# Comprehensive evaluation with the revised Space descriptor

This evaluation replaces the old single-context evidence with two complementary
factorial experiments:

1. **Inner realisability:** 3 gestures × 6 affective states × 5 optimiser seeds
   = 90 optimisation runs, with no VLM calls.
2. **Outer contextual-bandit evaluation:** 3 gestures × 6 states × 3 CEM seeds
   = 54 independently trained and frozen profiles. Each selected profile then
   receives 20 untouched VLM holdout judgements.

All commands must be run from the repository root. Existing completed runs are
skipped, and interrupted CEM runs resume from `latest_checkpoint.pt`.

## 1. Verify dependencies and inspect the planned work

```bash
python -m py_compile scripts/evaluation/run_comprehensive_evaluation.py

python scripts/evaluation/run_comprehensive_evaluation.py \
  --phase all \
  --dry-run \
  --gestures wave reach point \
  --states confident calm hesitant friendly confused angry \
  --seeds 7 41 97
```

The dry run prints every command and an estimated number of live evaluator
calls without running an experiment.

## 2. Run an inexpensive end-to-end mock smoke test

```bash
python scripts/evaluation/run_comprehensive_evaluation.py \
  --phase outer \
  --evaluator mock \
  --gestures wave \
  --states friendly \
  --seeds 7 \
  --rounds 1 \
  --samples-per-round 3 \
  --training-repeats 1 \
  --validation-top-k 1 \
  --validation-repeats 1 \
  --maxiter 5 \
  --popsize 3 \
  --local-maxiter 20 \
  --out outputs/comprehensive_smoke
```

This checks orchestration and file formats only. Mock perceptual scores are not
reportable evidence.

## 3. Run the revised inner-loop matrix

```bash
python scripts/evaluation/run_comprehensive_evaluation.py \
  --phase inner \
  --gestures wave reach point \
  --states confident calm hesitant friendly confused angry \
  --inner-seeds 7 17 27 37 47 \
  --inner-matrix-maxiter 45 \
  --inner-matrix-popsize 5 \
  --inner-matrix-local-maxiter 100 \
  --out outputs/comprehensive_revised_space
```

## 4. Recommended live pilot before the full factorial run

Set the Gemini credentials required by the existing evaluator, then run one
seed for one state across all gestures:

```bash
python scripts/evaluation/run_comprehensive_evaluation.py \
  --phase outer \
  --evaluator gemini \
  --confirm-live-cost \
  --gestures wave reach point \
  --states friendly \
  --seeds 7 \
  --rounds 5 \
  --samples-per-round 6 \
  --training-repeats 3 \
  --validation-top-k 3 \
  --validation-repeats 10 \
  --maxiter 90 \
  --popsize 8 \
  --local-maxiter 300 \
  --temperature 0.2 \
  --out outputs/comprehensive_revised_space
```

Freeze and evaluate the pilot profiles:

```bash
python scripts/evaluation/run_comprehensive_evaluation.py \
  --phase holdout \
  --evaluator gemini \
  --confirm-live-cost \
  --gestures wave reach point \
  --states friendly \
  --seeds 7 \
  --holdout-repeats 20 \
  --temperature 0.2 \
  --out outputs/comprehensive_revised_space
```

## 5. Full multi-gesture, multi-state, multi-seed outer evaluation

```bash
python scripts/evaluation/run_comprehensive_evaluation.py \
  --phase outer \
  --evaluator gemini \
  --confirm-live-cost \
  --gestures wave reach point \
  --states confident calm hesitant friendly confused angry \
  --seeds 7 41 97 \
  --rounds 5 \
  --samples-per-round 6 \
  --training-repeats 3 \
  --validation-top-k 3 \
  --validation-repeats 10 \
  --maxiter 90 \
  --popsize 8 \
  --local-maxiter 300 \
  --temperature 0.2 \
  --out outputs/comprehensive_revised_space
```

Run the untouched holdouts only after all intended selection runs are complete:

```bash
python scripts/evaluation/run_comprehensive_evaluation.py \
  --phase holdout \
  --evaluator gemini \
  --confirm-live-cost \
  --gestures wave reach point \
  --states confident calm hesitant friendly confused angry \
  --seeds 7 41 97 \
  --holdout-repeats 20 \
  --temperature 0.2 \
  --out outputs/comprehensive_revised_space
```

Do not use `--overwrite-holdout` merely because a result is unfavourable. That
flag exists only to recover from a documented invalid or interrupted holdout.

## 6. Generate tables and figures

```bash
python scripts/evaluation/run_comprehensive_evaluation.py \
  --phase report \
  --gestures wave reach point \
  --states confident calm hesitant friendly confused angry \
  --seeds 7 41 97 \
  --out outputs/comprehensive_revised_space
```

The `report` folder contains:

- `overall_metrics.json` — coverage, realisability, perceptual, and full-success metrics;
- `holdout_seed_results.csv` — one row per gesture/state/seed;
- `holdout_context_summary.csv` — mean, SD, and success rate per context;
- `inner_run_results.csv` — revised-descriptor inner matrix;
- classification-rate and target-probability heatmaps;
- realisation-RMSE and full-success heatmaps; and
- aggregate per-feature error plot.

Interpret the 20 repeated VLM calls as repeated judgements nested within one
generated gesture, not as 20 independent optimiser runs. The independent unit
for optimiser repeatability is the seed-level selected profile.
