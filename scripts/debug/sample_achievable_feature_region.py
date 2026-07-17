"""Approximate the feature region induced by the optimiser action space."""
from __future__ import annotations

import argparse
import csv
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from scipy.stats import norm, qmc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laban_rl.config import FEATURE_KEYS, JointLimits
from laban_rl.rewards import compute_joint_limit_penalty


def load_module():
    spec = spec_from_file_location("region_optimiser", ROOT / "scripts" / "direct_laban_feature_optimizer_spatiotemporal_final.py")
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", required=True, choices=["wave", "reach", "point"])
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--out", required=True)
    cli = parser.parse_args()

    m = load_module()
    args = m.build_parser().parse_args([])
    args.gesture = cli.gesture
    arm = m.laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    filt = m.laban.FilterConfig(enabled=True, cutoff_hz=5.0, order=4)
    ranges = m.load_ranges_or_default(str(ROOT / args.ranges), gesture=cli.gesture)
    q_ref = m.make_reference_trajectory(gesture_type=cli.gesture, arm=arm)
    basis = m.make_sine_basis(arm.n_points, args.n_spatial_basis)
    n_vars = m.spatial_variable_count(args.n_spatial_basis, args.endpoint_mode) + args.n_timing_basis

    # Sobol covers the complete bounded action space more evenly than iid random draws.
    power = int(np.ceil(np.log2(cli.samples)))
    unit = qmc.Sobol(d=n_vars, scramble=True, seed=cli.seed).random_base2(power)[:cli.samples]
    # Concentrate samples around the reference (x=0). Uniform sampling in an
    # 18-D cube overwhelmingly selects large simultaneous perturbations and
    # provides almost no preservation-compatible points.
    actions = np.clip(norm.ppf(np.clip(unit, 1e-8, 1.0 - 1e-8)) * cli.sigma, -1.0, 1.0)

    out_dir = Path(cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    limits = JointLimits()

    for index, x in enumerate(actions):
        q_var, q_spatial, warped_u, speed = m.build_variant(
            x, q_ref, basis, args.n_spatial_basis, args.n_timing_basis,
            args.max_delta, args.max_end_delta, args.endpoint_mode, args.time_scale,
            args.max_total_delta,
        )
        raw, clipped = m.compute_raw_and_norm_features(q_var, arm, filt, ranges)
        unclipped = m.laban.normalise_laban_features(raw, ranges, clip=False)
        terms = m.compute_preservation_terms(
            q_ref, q_var, q_spatial, warped_u, speed, arm,
            args.detour_tolerance, args.max_dev_tolerance,
        )
        joint_error = float(compute_joint_limit_penalty(q_var, limits))
        finite = all(np.isfinite(unclipped.get(k, np.nan)) for k in FEATURE_KEYS)
        joint_safe = joint_error <= 1e-8
        preservation_compatible = bool(
            finite and joint_safe
            and 0.70 <= terms["path_length_ratio"] <= 1.30
            and terms["nearest_path_max_dist"] <= 0.08
        )
        row = {
            "gesture": cli.gesture,
            "sample": index,
            "finite": finite,
            "joint_safe": joint_safe,
            "preservation_compatible": preservation_compatible,
            "joint_limit_error": joint_error,
            "path_length_ratio": terms["path_length_ratio"],
            "nearest_path_max_dist": terms["nearest_path_max_dist"],
            "direction_error": terms["direction_error"],
            "endpoint_error": terms["endpoint_error"],
            "normalisation_clipped_count": sum(abs(float(clipped[k]) - float(unclipped[k])) > 1e-12 for k in FEATURE_KEYS if np.isfinite(unclipped[k])),
        }
        for key in FEATURE_KEYS:
            row[f"unclipped_{key}"] = float(unclipped.get(key, np.nan))
            row[f"clipped_{key}"] = float(clipped.get(key, np.nan))
        rows.append(row)
        if (index + 1) % 512 == 0:
            valid_count = sum(r["preservation_compatible"] for r in rows)
            print(f"{cli.gesture}: {index+1}/{cli.samples}, preservation-compatible={valid_count}", flush=True)

    path = out_dir / f"achievable_region_{cli.gesture}_sigma_{cli.sigma:.2f}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
