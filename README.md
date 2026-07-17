# Laban RL Trajectory-Styling Pipeline

This project generates expressive variants of simple planar robot-arm gestures while preserving the original task. A reference **wave**, **reach**, or **point** trajectory is transformed toward an affective Laban profile such as **confident**, **calm**, **hesitant**, or **friendly**.

The repository supports three search strategies:

1. **PPO reinforcement learning** learns a six-dimensional style action.
2. **Random search** provides a simple baseline over the same action space.
3. **Direct spatiotemporal optimisation** searches spatial deformation and timing-warp coefficients directly. This is the most developed experimental implementation in the repository.

## Framework overview

```text
gesture name
    -> reference joint trajectory q_ref
    -> style transformation or coefficient-based deformation
    -> candidate trajectory q_variant
    -> forward kinematics and Laban feature extraction
    -> robust feature normalisation
    -> comparison with an affective target profile
    -> style + task-preservation + feasibility objective
    -> optimiser / PPO update
    -> plots, animation, arrays, history, and summary
```

The arm is represented by shoulder and elbow joint angles over time. Forward kinematics converts these angles to elbow and wrist positions. Each candidate is evaluated using five normalized Laban-inspired features:

| Feature | Interpretation in this implementation |
| --- | --- |
| `weight` | Energy/force proxy derived from motion dynamics |
| `time` | Acceleration-based urgency or suddenness |
| `flow_boundness` | Jerk-based bound versus free-flowing quality |
| `space_indirectness` | Wrist path length relative to displacement |
| `shape_arcness` | Curvature/arc quality of the wrist path |

The target values for each affective style are defined in `src/laban_rl/targets.py`. They are computational targets rather than claims of a complete formal Laban Movement Analysis model.

## Implementation routes

### 1. PPO style-action learning

`LabanTrajectoryStylerEnv` exposes a Gymnasium environment. Its policy produces:

```text
[amplitude, timing, curve, pause, smoothing, envelope_blend]
```

All six values lie in `[-1, 1]`. `apply_style_action()` converts this action into a trajectory by applying a time warp, amplitude envelope, pathway curvature, optional hold, and smoothing. The start pose is retained; endpoint drift is normally handled by a soft reward penalty.

The reward combines Laban target matching with trajectory preservation, endpoint preservation, smoothness, joint-limit feasibility, action regularisation, saturation control, and a small target-dependent action preference.

### 2. Random-search baseline

Random search samples the same six-dimensional style-action space and retains the highest-reward candidate. It is useful for checking whether PPO improves on a transparent non-learning baseline and for debugging the reward landscape.

### 3. Direct spatiotemporal Laban optimisation

The final direct optimiser represents spatial changes with sine-basis coefficients and timing changes with a monotonic time warp. SciPy differential evolution searches these coefficients against a weighted Laban feature error plus preservation constraints.

The objective includes wrist-path proximity, endpoint and direction preservation, path-length control, detour and maximum-deviation penalties, joint smoothness and limits, and coefficient/time-warp regularisation. Endpoint behavior can be selected as:

- `hard`: force the final joint pose to match the reference;
- `soft`: allow a learned endpoint offset with a penalty;
- `free`: allow endpoint variation, normally with zero endpoint weight.

## Files and responsibilities

### Core package

| File | Main responsibility and functions |
| --- | --- |
| `src/laban_rl/config.py` | Shared feature order, gesture names, reward weights, and joint limits (`RewardWeights`, `JointLimits`). |
| `src/laban_rl/targets.py` | Affective `TARGET_PROFILES` for confident, calm, hesitant, and friendly motion. |
| `src/laban_rl/trajectories.py` | Builds wave, reach, and point references through `make_reference_trajectory()`. |
| `src/laban_rl/style_actions.py` | Maps a six-dimensional action to a styled trajectory with `apply_style_action()`; contains time warping, envelope blending, pauses, and smoothing. |
| `src/laban_rl/features.py` | Computes raw/normalized features, converts dictionaries to arrays, and masks unreliable features by gesture. |
| `src/laban_rl/rewards.py` | Computes style error, preservation and feasibility penalties, and the combined reward. |
| `src/laban_rl/envs.py` | Gymnasium `LabanTrajectoryStylerEnv` used for PPO and action-based evaluation. |
| `src/laban_rl/training.py` | Builds and trains the Stable-Baselines3 PPO model. |
| `src/laban_rl/evaluation.py` | Runs random search and evaluates a trained policy. |
| `src/laban_rl/visualisation.py` | Saves feature/joint/path plots, arm GIFs, summaries, and trajectory arrays. |
| `src/laban_rl/io_utils.py` | Loads saved normalization ranges or supplies fallback ranges. |

### Experiment and entry-point scripts

| File | Purpose |
| --- | --- |
| `scripts/train_ppo.py` | Command-line PPO training, evaluation, and output generation. |
| `scripts/run_random_search.py` | Command-line random-search baseline. |
| `scripts/direct_laban_feature_optimizer_spatiotemporal_final.py` | Recommended direct optimiser with spatial bases, time warping, and task-preservation terms. |
| `scripts/direct_laban_feature_optimizer.py` | Earlier spatial-only direct optimiser. |
| `scripts/direct_laban_feature_optimizer_constrained.py` | Earlier constrained direct-optimisation experiment. |
| `scripts/direct_laban_feature_optimizer_spatiotemporal.py` | Initial spatiotemporal version. |
| `scripts/direct_laban_feature_optimizer_spatiotemporal_v2.py` | Intermediate version retained for comparison. |
| `scripts/debug_action_sensitivity_refactor.py` | Sweeps individual action dimensions. |
| `scripts/debug_combo_sensitivity_refactor.py` | Sweeps pairs of action dimensions. |
| `scripts/debug_triple_sensitivity_refactor.py` | Sweeps triples of action dimensions. |
| `robust_laban_normalisation_balanced_3gestures.py` | Generates gestures, computes raw Laban features, derives robust normalization ranges, and provides shared arm/kinematics utilities. |

### Configuration, tests, and results

| Path | Contents |
| --- | --- |
| `configs/normalisation_ranges_balanced_3gestures.json` | Robust feature ranges shared by training and optimisation. |
| `tests/` | Unit and integration tests for trajectories, actions, features, rewards, environment, evaluation, visualization, and training imports. |
| `outputs/` | Saved experiment directories. Each run is kept separate so methods and parameter settings can be compared. |

## Installation

Python 3.10 or newer is required. From the directory containing this README:

```bash
python -m venv .venv
```

Activate the environment, then install the project and dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Running experiments

### Final spatiotemporal optimiser

```bash
python scripts/direct_laban_feature_optimizer_spatiotemporal_final.py \
  --gesture point \
  --target confident \
  --out outputs/final_point_confident_spatiotemporal
```

Useful controls include `--n-spatial-basis`, `--n-timing-basis`, `--endpoint-mode`, `--maxiter`, `--popsize`, and the individual preservation weights. Run the script with `--help` for the complete list.

### PPO

```bash
python scripts/train_ppo.py \
  --gesture point \
  --target confident \
  --timesteps 30000 \
  --out outputs/ppo_runs/point_confident
```

### Random search

```bash
python scripts/run_random_search.py \
  --gesture point \
  --target confident \
  --trials 2000 \
  --out outputs/random_search/point_confident
```

### Real contextual-bandit learning with Gemini video evaluation

Train a Beta-distribution policy on one gesture-state context using Gemini for perceptual evaluation:

```bash
python scripts/train_real_continuous_contextual_bandit.py \
  --gesture point \
  --target-state confident \
  --rounds 30 \
  --repeats 3 \
  --entropy-weight 0.0 \
  --entropy-decay-steps 0 \
  --rmse-normal-threshold 0.05 \
  --rmse-skip-threshold 0.10 \
  --skip-high-rmse-updates \
  --reward-margin-mode raw \
  --out outputs/bandit_point_confident_v1 \
  --overwrite
```

**Key options:**
- `--rmse-normal-threshold`: RMSE below this allows normal policy updates.
- `--rmse-skip-threshold`: RMSE at or above this skips the policy update (prevents action-reward mismatch corruption).
- `--skip-high-rmse-updates`: Enable RMSE gating.
- `--entropy-weight`: Entropy bonus coefficient (0.0 = off, 0.01-0.05 = on). Default: 0.0.
- `--entropy-decay-steps`: Linearly decay entropy to zero over N rounds. 0 = no decay.
- `--reward-margin-mode`: `raw` (actual margin) or `clipped` (max(0, margin)) for reward computation.
- `--allow-default-profile`: Allow missing informed profiles to default to [0.5, 0.5, 0.5, 0.5, 0.5]. Default: error.
- `--overwrite`: Overwrite existing output folder. Default: error if non-empty.

### Recompute robust normalization ranges

```bash
python robust_laban_normalisation_balanced_3gestures.py
```

Because all target comparisons depend on these ranges, use the same normalization file when comparing experiments.

## Perceptual Bandit Training (NEW)

The real contextual-bandit training loop connects a Beta-distribution policy to:
1. A reference gesture trajectory.
2. A spatiotemporal optimiser that deforms the trajectory toward a target Laban profile.
3. A variant-only video renderer.
4. Gemini Pro Vision for perceptual evaluation of the animated gesture.
5. REINFORCE policy updates with optional entropy regularization and RMSE gating.

### Policy Learning with Safety Checks

**RMSE Gating:** When the inner optimiser's achieved profile is far from the requested profile (high `realisation_rmse`), the policy update is skipped to prevent learning from corrupted signals.

**Reward Transparency:** Both `raw` and `clipped` margins are logged. Raw margins reveal true failures (negative margins); clipped margins soften the signal for early training stability.

**Entropy Control:** Entropy bonus is **off by default** (`--entropy-weight 0.0`). Enable optional exploration with `--entropy-weight 0.01` and `--entropy-decay-steps N` to decay entropy to zero after N rounds.

**Informed Initialization:** All 12 gesture-state contexts (wave, reach, point × confident, calm, hesitant, friendly) have informed initial profiles. Missing profiles raise an error unless `--allow-default-profile` is passed.

**Output Safety:** Existing output folders must be explicitly overwritten with `--overwrite`; otherwise an error is raised. This prevents accidental contamination of results.

### Output Artifacts from Bandit Training

| Artifact | Description |
| --- | --- |
| `training_history.csv` | Per-round metrics: outer_reward, outer_reward_clipped, mean_target_probability, mean_margin, mean_margin_clipped, realisation_rmse, rmse_skipped, entropy, effective_entropy_weight, policy_updated, policy_loss. |
| `results_summary.json` | Final summary: best_reward, best_profile, final_policy_mean, final_policy_std, mean_reward_all_rounds, mean_reward_last_5_rounds, num_skipped_rmse_updates, entropy_settings, RMSE gate thresholds. |
| `best_profile.json` | Best sampled profile and associated reward. |
| `latest_checkpoint.pt` | Latest policy, optimizer, and baseline checkpoints. |
| `rounds/round_NNN/round_summary.json` | Per-round details: sampled_profile, policy_mean_after_update, environment_result (with raw/clipped rewards), update_stats. |
| `rounds/round_NNN/optimiser_outputs/` | Outputs from the inner spatiotemporal optimiser for that round. |
| `reward_curve.png` | Outer reward over rounds. |
| `target_probability_curve.png` | Gemini target-state probability over rounds. |
| `realisation_rmse_curve.png` | Requested-to-achieved profile RMSE over rounds. |
| `policy_*_curve.png` | Policy trajectory and exploration for each Laban feature. |

## Testing

Run the complete suite from the project directory:

```bash
pytest
```

For tests specific to the new perceptual-bandit improvements (RMSE gating, entropy decay, reward logging, all 12 profiles):

```bash
pytest tests/test_policy_and_environment_improvements.py -v
```

The tests cover trajectory construction, style-action behavior, Laban feature handling, reward terms, environment operation, policy evaluation, output generation, optional training imports, RMSE gating, entropy logging, and profile validation.

## Extending the framework

- Add an affective style by defining a normalized profile in `targets.py`.
- Add a gesture in the underlying generator, register it in `config.py`, and route it through `make_reference_trajectory()`.
- Change action semantics in `style_actions.py` and adjust reward regularisation in `rewards.py`.
- Change direct-optimisation constraints or basis representations in the final spatiotemporal optimiser.
- Recompute normalization ranges whenever the gesture distribution, arm model, sampling duration, or feature definitions change materially.
# Laban-inspired-RL-Trajectory-Styler
# RLBased-TrajectoryOptimizer
