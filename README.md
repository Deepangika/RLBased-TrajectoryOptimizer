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

Target an arbitrary normalized VAD point directly with all three explicit
coordinates:

  python scripts/train_cem_contextual_bandit.py \
    --gesture point \
    --target-valence 0.72 \
    --target-arousal 0.58 \
    --target-dominance 0.81 \
    --evaluator mock \
    --rounds 2 \
    --cem-samples-per-round 4 \
    --cem-min-elites 2 \
    --repeats 2 \
    --validation-repeats 3 \
    --validation-top-k 2 \
    --out outputs/direct_vad_point_mock

The same direct target works with Gemini by changing `--evaluator mock` to
`--evaluator gemini` and supplying the API key described below. The three VAD
values must each lie in `[0, 1]`; they are an all-or-nothing alternative to
`--target-state`. Direct targets use VAD reward mode, while categorical reward
mode remains available only for named targets.

Every run writes the resolved target to `target_context.json`, each history row,
the checkpoint, and `results_summary.json`. Resuming with a different gesture,
named state, direct VAD point, evaluator, optimizer, or reward configuration is
rejected.

The CEM optimizer operates directly on the resolved VAD target and uses the
nearest named anchor only to choose an informed initial Laban profile. The
separate `ContinuousContextualBanditPolicy` remains a per-named-state policy and
does not generalize across arbitrary VAD coordinates.

If you use Gemini evaluation, set your API key first:

  set GOOGLE_API_KEY=your_key_here

or in PowerShell:

  $env:GOOGLE_API_KEY="your_key_here"

## Testing

Run the test suite:

  python -m pytest -q tests

Current suite status after cleanup: all tests passing.

## Normalization calibration

The gesture-specific range file includes robust 5th-95th percentile ranges for
circle, beckon, and celebratory-pump. They were generated deterministically with:

  python scripts/calibrate_new_gesture_ranges.py \
    --samples 9000 \
    --seed 20260731 \
    --low-percentile 5 \
    --high-percentile 95

The sweep uses the existing balanced sampler (3,000 semantic variants per
gesture), the standard 160-point/2-second arm and 5 Hz Butterworth feature
pipeline, and the parameter bounds in
`robust_laban_normalisation_balanced_3gestures.py`. The calibration command
updates only those three gesture entries; the legacy balanced, wave, reach, and
point ranges remain unchanged.

Circle previously inherited the balanced range, whose shape-arcness lower bound
was above the circle reference value and therefore clipped the reference to
zero. Its dedicated range places the reference inside the calibrated region.
Celebratory-pump also receives five timing bases, a `2.0` timing scale, and
stronger Flow tracking in the CEM inner optimizer; its spatial and path
constraints are unchanged.

## Notes

- The repository is experiment-heavy by design; many scripts are operational entry points.
- Automated tests focus on core logic under src/laban_rl and selected workflow behavior.
- Outputs and cache folders can be large and are safe to clean when not needed.
