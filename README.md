# Updated CEM Live Retest V2

Laban-based contextual bandit pipeline for gesture style optimization and perception-aligned training.

## What this project does

- Optimizes gesture style profiles with a CEM outer loop
- Uses a direct inner optimizer for trajectory realization
- Supports mock and Gemini evaluators for perceptual feedback
- Uses valence-arousal-dominance (VAD) as the primary perceptual reward
- Retains Ekman emotion probabilities as secondary diagnostics
- Provides evaluation and debugging scripts for diagnostics and ablations
- Includes wave, reach, point, circle, beckon, and celebratory-pump references

## Project layout

- src/laban_rl
  - Core package modules for features, rewards, trajectories, policy, environment, and optimizer API
- src/laban_rl/perceptual_bandit
  - CEM optimizer, policy logic, perceptual environment, evaluator integration, and candidate selection
- scripts
  - Main training, batch runs, monitoring, plotting, and utility workflows
- scripts/debug
  - Diagnostics and targeted investigation scripts
- scripts/evaluation
  - Evaluation pipelines and DE transfer/ablation workflows
- tests
  - Automated pytest suite for core behavior and regression checks
- configs
  - Normalization and experiment configuration data
- outputs
  - Experiment artifacts and generated results

## Environment setup

1. Create and activate a Python environment (conda or venv)
2. Install dependencies

   pip install -r requirements.txt

3. Optional editable install

   pip install -e .

## Quick start

Run a small CEM training example:

  python scripts/train_cem_contextual_bandit.py \
    --gesture wave \
    --target-state surprise \
    --rounds 1 \
    --cem-samples-per-round 3 \
    --cem-elite-fraction 0.67 \
    --cem-min-elites 2 \
    --cem-initial-width 0.15 \
    --cem-smoothing 0.5 \
    --cem-min-std 0.05 \
    --repeats 2 \
    --validation-repeats 3 \
    --validation-top-k 2 \
    --evaluator gemini \
    --model gemini-2.5-flash \
    --temperature 0.2 \
    --maxiter 45 \
    --popsize 5 \
    --local-maxiter 100 \
    --de-mutation 0.5 \
    --de-recombination 0.65 \
    --max-feature-error-threshold 0.10 \
    --reject-excessive-feature-error \
    --stability-penalty-weight 0 \
    --reward-margin-mode raw \
    --seed 7 \
    --out outputs/live_wave_surprise_diagnostic_v1

The default perceptual objective is weighted VAD distance (`0.2` valence,
`0.4` arousal, `0.4` dominance). Legacy categorical-margin experiments can
be reproduced with `--perceptual-reward-mode categorical`.

If you use Gemini evaluation, set your API key first:

  set GOOGLE_API_KEY=your_key_here

or in PowerShell:

  $env:GOOGLE_API_KEY="your_key_here"

## Testing

Run the test suite:

  python -m pytest -q tests

Current suite status after cleanup: all tests passing.

## Notes

- The repository is experiment-heavy by design; many scripts are operational entry points.
- Automated tests focus on core logic under src/laban_rl and selected workflow behavior.
- Outputs and cache folders can be large and are safe to clean when not needed.
