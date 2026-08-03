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
  - Main training, calibration, visualization, and utility workflows
- scripts/debug
  - Current realizability, holdout, and achievable-region diagnostics
- scripts/evaluation
  - Paired evaluation, reward ablation, compatibility, and readiness workflows
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
rejected. Resume an interrupted compatible run explicitly with `--resume`; use
`--overwrite` to discard an existing run and start fresh. An explicit resume can
also recover deterministic work and cached repeats if interruption occurred
before the first checkpoint. Checkpoints created before strict feasible-only CEM
updates are rejected because their learned search state cannot be reinterpreted
safely.

CEM updates use only candidates that pass physical checks, feature RMSE and
maximum-error thresholds, and the complete requested perceptual repeat count.
If fewer than `--cem-min-elites` candidates qualify, that round does not update
the distribution. Exploration contracts only after a strictly feasible
incumbent improves. Successful repeats are persisted so retries collect only
missing observations. Training and independent validation use separate cache
namespaces to prevent validation from reusing observations that influenced the
search. Results report the best observed profile separately from the best
strictly feasible profile and split training from validation evaluator counts.

The CEM optimizer operates directly on the resolved VAD target and uses the
nearest named anchor only to choose an informed initial Laban profile. The
separate `ContinuousContextualBanditPolicy` remains a per-named-state policy and
does not generalize across arbitrary VAD coordinates.

### Focused inner-optimizer screening

Before spending Gemini calls on difficult gesture/state pairs, run the
mock-independent inner screens:

```text
python scripts/evaluation/run_focused_inner_screen.py --phase beckon-fear
python scripts/evaluation/run_focused_inner_screen.py --phase beckon-surprise
python scripts/evaluation/run_focused_inner_screen.py --phase wave
```

The runner enforces the `0.10` per-feature threshold and records physical,
path-preservation, smoothness, and per-feature diagnostics for seeds 7, 17, and
27. Each case is persisted atomically and can safely resume. Wave varies Flow
target weight and timing resolution; beckon varies optimizer effort and timing
resolution with moderate Time/Flow tracking and additional Shape tracking for
fear. Raw and seed-aggregated CSV/JSON outputs are written under
`outputs/focused_inner_screen`.

The robust inner-optimizer defaults selected by this screen are applied
automatically during CEM training. Beckon-fear uses five timing bases and the
75-iteration/eight-member budget; beckon-surprise uses six timing bases with the
same budget. The four difficult wave states (anger, disgust, fear, and sadness)
use six timing bases, the 75/eight budget, and Flow target weight 1.5.

### Feasible wave-target projection

For wave anger, disgust, and sadness, the affect-derived five-feature profiles
contain combinations that were not reliably realizable under the strict `0.10`
feature gate. Reproduce the deterministic feasible-region projection with:

```text
python scripts/evaluation/project_wave_feasible_targets.py \
  --out outputs/wave_feasible_projection \
  --config-out configs/wave_feasible_target_projections.json
```

For each state, the workflow screens 128 nearby candidates with seed 7, then
validates the eight nearest feasible candidates with seeds 17 and 27. It
publishes a projection only when the selected profile passes all three seeds,
the path-ratio gate, and the per-feature threshold. CEM uses that projected
profile only as its initialization centre: perceptual scoring still uses the
original named affect and VAD target. Run metadata records both profiles,
projection distance, validation evidence, and the effective inner-optimizer
overrides so resumes and holdout checks remain reproducible.

Render the selected projected wave motions together with the refined wave-fear
and beckon motions for visual review:

```text
python scripts/evaluation/render_refined_motion_review.py \
  --out outputs/refined_motion_review \
  --seed 7
```

The command writes one GIF and feature plot per condition plus
`render_summary.json`. Passing numerical gates establish realizability, not
human interpretability; review the rendered motions before live evaluation.

### Limited Gemini pilot

The preconfigured live pilot contains six affect-diverse original-target
conditions and projected wave anger, sadness, and disgust:

```text
python scripts/evaluation/run_paired_perceptual_experiment.py \
  --matrix configs/limited_gemini_pilot.json \
  --evaluator gemini \
  --repeats 3 \
  --paired-ab \
  --cache outputs/limited_gemini_pilot_cache \
  --out outputs/limited_gemini_pilot
```

Each condition compares the unmodified gesture reference with one styled
motion. The 18 logical entries contain 15 unique clips because the same wave
reference is reused across four target conditions; the content-addressed cache
therefore requires 45 Gemini evaluations rather than 54. Feature-infeasible or
physically invalid styled motions are rejected before evaluator calls.

Projected conditions retain the original named state and VAD as the
perceptual target. Each result separately records the original affect-derived
Laban target, projected feasible target, requested optimization profile, and
actually achieved profile. Restart the identical command to resume missing
repeats without duplicating completed calls.

### Five-pair long-clip diagnostic

After the initial pilot, use the focused five-pair diagnostic instead of a
complete 36-condition run:

```text
python scripts/evaluation/run_paired_perceptual_experiment.py \
  --matrix configs/five_pair_gemini_diagnostic.json \
  --evaluator gemini \
  --repeats 5 \
  --paired-ab \
  --paired-preference-repeats 5 \
  --video-duration-seconds 2 \
  --video-lead-in-seconds 0.5 \
  --video-repetitions 2 \
  --video-inter-repeat-transition-seconds 0.5 \
  --video-final-hold-seconds 0.5 \
  --cache outputs/five_pair_diagnostic_cache \
  --paired-preference-cache outputs/five_pair_preference_cache \
  --out outputs/five_pair_diagnostic
```

The five conditions are celebratory-pump anger, beckon disgust, projected
wave anger, projected wave disgust, and wave surprise. Each 5.5-second MP4
holds the initial pose for 0.5 seconds, plays two natural-speed gesture cycles
separated by a smooth 0.5-second return, and ends with a 0.5-second hold.
Independent VAD/category ratings remain
target-blind. A separate prompt receives the intended affect and asks which
blinded clip expresses it more strongly: A, B, or neither. Display order is
deterministic and balanced across five repeats.

The experiment contains eight unique clips, requiring 40 independent ratings,
plus 25 paired judgments. Both caches persist each successful repeat
immediately. Motion snapshots prevent re-optimization during resume, and
uploaded Gemini files are deleted when collection closes.

Estimate a conservative empirical VAD envelope from a completed independent
result:

```text
python scripts/evaluation/calibrate_empirical_vad_region.py \
  --results outputs/limited_gemini_pilot/paired_results.json \
  --out outputs/achievable_vad_calibration/live_pilot_empirical_vad.json
```

This reports observed clip-mean ranges and each canonical target's nearest
point on the sparse empirical VAD convex hull. It is a diagnostic projection,
not a replacement for canonical targets or human calibration.

## Paired perceptual experiments

The frozen full evaluation matrix contains 36 gesture-state conditions, 42
unique videos, five target-blind VAD repeats per video, and five blinded
A/B/neither judgments per condition. Build it reproducibly and rehearse it
offline before using Gemini:

```text
python scripts/evaluation/build_full_gemini_matrix.py
python scripts/evaluation/run_paired_perceptual_experiment.py \
  --matrix configs/full_gemini_matrix.json \
  --evaluator mock \
  --repeats 5 \
  --paired-ab \
  --paired-preference-repeats 5 \
  --gesture-wide-camera-limits \
  --video-duration-seconds 2.0 \
  --video-lead-in-seconds 0.5 \
  --video-repetitions 2 \
  --video-inter-repeat-transition-seconds 0.5 \
  --video-final-hold-seconds 0.5 \
  --cache outputs/full_gemini_cache \
  --out outputs/full_gemini
```

The matrix rejects drift in model, temperature, prompts, schemas, rendering,
camera policy, or repeat counts. A live run changes only `--evaluator mock` to
`--evaluator gemini` and makes 390 evaluator calls: 210 independent ratings
and 180 paired judgments. The runner rejects any infeasible motion before
either evaluation phase. `full_experiment_report.json` and CSV preserve the
original, projected, requested, and achieved Laban profiles; projection and
realisation errors; VAD changes; preference rates; variability; and path
preservation metrics.

Use the paired runner before human validation instead of running separate VAD
and categorical evaluator passes. It renders each candidate once, stores every
structured repeat observation in a content-addressed cache, and derives both
reward views from exactly those observations. Cache keys include trajectory
content, gesture prompt context, model, temperature, renderer settings, and
prompt/schema versions; API keys and other credentials are never cached.

Create a matrix JSON such as:

```json
{
  "gestures": ["point", "wave"],
  "targets": [
    {"state": "happiness"},
    {
      "label": "custom_vad",
      "vad": {"valence": 0.72, "arousal": 0.58, "dominance": 0.81}
    }
  ],
  "seeds": [7, 19],
  "candidate_profiles": {
    "baseline": {
      "weight": 0.50,
      "time": 0.50,
      "flow_boundness": 0.50,
      "space_indirectness": 0.50,
      "shape_arcness": 0.50
    },
    "candidate_b": {
      "weight": 0.65,
      "time": 0.70,
      "flow_boundness": 0.40,
      "space_indirectness": 0.30,
      "shape_arcness": 0.70
    }
  }
}
```

Run the complete matrix offline with the deterministic mock evaluator:

```text
python scripts/evaluation/run_paired_perceptual_experiment.py \
  --matrix configs/paired_matrix.json \
  --evaluator mock \
  --repeats 5 \
  --cache outputs/paired_cache \
  --out outputs/paired_mock
```

For a focused two-profile A/B check, use the same matrix with exactly two
candidate profiles and add `--paired-ab`. The report compares ranking agreement,
repeat stability, feasibility, and selected candidate identity. **VAD and
categorical reward magnitudes are not directly comparable.** Direct VAD targets
do not require categorical output; categorical rankings are reported only when
the cached evaluator observations provide the optional probabilities.

Change `--evaluator mock` to `--evaluator gemini` for live collection after
setting the API key. Interrupted runs can be restarted with the identical
command: completed candidates are read from `paired_results.json`, and partially
collected repeats resume from the valid cache without duplicate Gemini calls.
Changing clip content or any evaluator/prompt/schema setting creates a new cache
identity; tampered or incompatible entries are rejected.

Each run writes `paired_results.json`, `paired_results.csv`,
`paired_rankings.png`, and `paired_stability.png`. Reliability output includes
per-axis VAD mean/standard deviation, winner agreement, probability entropy and
variability, and confidence variability. ICC(2,1) is emitted only when at least
two clips and two repeats make it identifiable; low-repeat and zero-variance
cases are explicitly marked unavailable rather than represented as NaN.

Secondary category output is explicitly `complete`, `ambiguous`, or `missing`.
The evaluator may report independent per-category intensities without forcing a
winner; normalized six-state probabilities remain available for legacy
diagnostics when intensities have a positive sum. Reports separately expose
distribution coverage and unambiguous coverage. Direct VAD targets never require
category output.

## Reward and feasibility ablations

Rescore an existing paired result without new evaluator calls:

```text
python scripts/evaluation/rescore_paired_ablations.py \
  --paired-results outputs/paired/paired_results.json \
  --configurations configs/ablation_grid.json \
  --out outputs/paired/ablation_comparison.json
```

The configuration file is a JSON array whose `settings` may vary VAD weights,
realization/stability penalties, feature-error thresholds, and feasibility
gates. Output ranks VAD and categorical views independently because their raw
magnitudes are not equivalent. Settings that alter the inner optimizer, require
observations previously skipped by a gate, or lack raw legacy gate measurements
are marked as requiring an inner-optimizer rerun.

Keep `stability_penalty_weight=0` until multi-clip repeated evaluation has
measured test-retest reliability. Non-zero stability ablations require at least
two repeats per clip and an identifiable reliability result. Apply the penalty
only to repeat dispersion; do not also encode that same evaluator uncertainty
in a second confidence penalty. No empirical stability constant is assumed.

## Checkpoint and cache compatibility

Current CEM checkpoints include format, feature-order, state-schema, and complete
named/direct-VAD target fingerprint metadata plus strict outer-loop semantics.
Legacy learned-policy weights without observation-shape metadata are rejected
rather than reinterpreted. Metadata-less and pre-strict CEM checkpoints are also
rejected because their learned search state cannot be migrated safely.

Evaluator cache format and Gemini prompt/schema versions are content-addressed.
Old entries are never silently reused after category-schema changes.

## Pre-human readiness audit

Create a manifest pointing to the calibration JSON, paired matrix/results,
realisability summary, evaluator cache, and recorded full-test status, then run:

```text
python scripts/evaluation/pre_human_readiness.py \
  --manifest configs/pre_human_readiness.json \
  --out outputs/readiness/pre_human_readiness.json
```

The command emits JSON plus a concise terminal report and exits nonzero for
blocking calibration, matrix coverage, realizability, repeat reliability,
ambiguity coverage, cache compatibility, test-status, credential, or artifact
hygiene failures. It does not call Gemini.

If you use Gemini evaluation, set your API key first:

  set GOOGLE_API_KEY=your_key_here

or in PowerShell:

  $env:GOOGLE_API_KEY="your_key_here"

Keep keys in the process environment, never in source, matrix files, prompts, or
cache settings. Remove them after live collection with
`Remove-Item Env:GOOGLE_API_KEY` (PowerShell) or `set GOOGLE_API_KEY=` (cmd).
The readiness audit reports only variable names and suspect file paths; it never
prints credential values.

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
