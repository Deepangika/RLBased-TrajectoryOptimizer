# Simplified inner-optimiser objective

The inner optimiser uses the conceptual objective

\[
\theta^*=\arg\min_\theta
\left(E_{\mathrm{LMA}}+\lambda_pE_{\mathrm{path}}+\lambda_sE_{\mathrm{jerk}}\right).
\]

- `E_LMA`: weighted RMSE between the requested and realised five-dimensional Laban profiles.
- `E_path`: nearest-reference-path mean squared distance in end-effector space.
- `E_jerk`: joint-space jerk measure, retaining the implementation's existing `0.001` scale.

The following are feasibility requirements rather than additional conceptual objectives:

- finite trajectory and Laban values;
- joint limits;
- monotonic temporal progression, guaranteed by construction;
- path-length ratio in `[0.70, 1.30]` by default;
- endpoint error no greater than `0.08 m` by default;
- cosine direction error no greater than `0.25` by default;
- overall and per-dimension Laban errors no greater than the outer-loop realisability tolerance, normally `0.10`.

Differential Evolution still needs a scalar search signal. `constraint_penalty` is therefore a quadratic hinge guide that is zero throughout the feasible region and positive only when a constraint is violated. It is stored separately from `objective_loss`:

\[
\texttt{total\_loss}=\texttt{objective\_loss}+\texttt{constraint\_penalty}.
\]

The outer environment independently rejects motions that fail the physical constraints, and `strict_realisability` additionally checks the overall and maximum per-feature Laban errors.

The default `endpoint_mode="soft"` retains learned endpoint-offset variables. With `endpoint_mode="hard"`, the parameterisation fixes both reference endpoints exactly, so the endpoint constraint is satisfied by construction.

Relevant command-line controls:

```text
--nearest-path-weight 1.5
--smooth-weight 0.01
--minimum-path-length-ratio 0.70
--maximum-path-length-ratio 1.30
--endpoint-tolerance 0.08
--direction-tolerance 0.25
--constraint-penalty-weight 100.0
```

Legacy loss-weight arguments remain accepted so existing experiment configurations load without modification, but duplicated preservation, target-dimension, coefficient, timing, detour, and maximum-deviation terms no longer contribute to the compact objective.
