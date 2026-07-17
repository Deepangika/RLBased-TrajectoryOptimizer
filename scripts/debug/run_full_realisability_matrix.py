"""Run a multi-seed target-realisability matrix without rendering each case."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laban_rl.config import FEATURE_KEYS
from laban_rl.targets import TARGET_PROFILES


def load_optimiser():
    spec = spec_from_file_location(
        "matrix_optimiser",
        ROOT / "scripts" / "direct_laban_feature_optimizer_spatiotemporal_final.py",
    )
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.save_outputs = lambda result, out_path, arm: None
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", required=True, choices=["wave", "reach", "point"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27, 37, 47])
    parser.add_argument(
        "--states",
        nargs="+",
        choices=list(TARGET_PROFILES),
        default=list(TARGET_PROFILES),
    )
    parser.add_argument("--maxiter", type=int, default=45)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--local-maxiter", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def main():
    cli = parse_args()
    module = load_optimiser()
    out_dir = Path(cli.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for state in cli.states:
        target = TARGET_PROFILES[state]
        for seed in cli.seeds:
            args = module.build_parser().parse_args([])
            args.gesture = cli.gesture
            args.target = state
            args.seed = seed
            args.maxiter = cli.maxiter
            args.popsize = cli.popsize
            args.local_maxiter = cli.local_maxiter
            # Include the gesture to prevent diagnostics from later gesture
            # runs overwriting an earlier gesture with the same state/seed.
            args.out = str(out_dir / "cases" / cli.gesture / state / f"seed_{seed}")

            with contextlib.redirect_stdout(io.StringIO()):
                result = module.optimise(args)

            clipped = np.asarray([result["var_norm"].get(k, np.nan) for k in FEATURE_KEYS], float)
            unclipped = np.asarray([result["var_norm_unclipped"].get(k, np.nan) for k in FEATURE_KEYS], float)
            target_array = np.asarray([target[k] for k in FEATURE_KEYS], float)
            clipped_error = np.abs(clipped - target_array)
            unclipped_error = np.abs(unclipped - target_array)
            action = np.asarray(result["action"], float)
            info = result["reward_info"]
            valid = bool(np.all(np.isfinite(unclipped)))
            within = bool(valid and np.all(unclipped_error <= cli.tolerance))
            path_preserved = bool(0.70 <= info["path_length_ratio"] <= 1.30)
            joint_safe = bool(info["joint_limit_error"] <= 1e-8)

            row = {
                "gesture": cli.gesture,
                "state": state,
                "seed": seed,
                "valid_features": valid,
                "realised_all_features": within,
                "path_preserved_0.70_1.30": path_preserved,
                "joint_limits_satisfied": joint_safe,
                "fully_acceptable": bool(within and path_preserved and joint_safe),
                "clipped_rmse": finite_float(np.sqrt(np.mean((clipped - target_array) ** 2))),
                "unclipped_rmse": finite_float(np.sqrt(np.mean((unclipped - target_array) ** 2))),
                "unclipped_max_abs_error": finite_float(np.max(unclipped_error)),
                "saturation_rate_0.95": float(np.mean(np.abs(action) >= 0.95)),
                "normalisation_clipped_count": int(np.sum(np.abs(clipped - unclipped) > 1e-12)),
                "path_length_ratio": info["path_length_ratio"],
                "nearest_path_mse": info["nearest_path_mse"],
                "nearest_path_max_dist": info["nearest_path_max_dist"],
                "endpoint_error": info["endpoint_error"],
                "direction_error": info["direction_error"],
                "joint_limit_error": info["joint_limit_error"],
                "smoothness_error": info["smoothness_error"],
            }
            for i, key in enumerate(FEATURE_KEYS):
                row[f"target_{key}"] = target_array[i]
                row[f"clipped_{key}"] = clipped[i]
                row[f"unclipped_{key}"] = unclipped[i]
                row[f"unclipped_error_{key}"] = unclipped_error[i]
            rows.append(row)
            print(
                f"{cli.gesture:5s} {state:9s} seed={seed:2d} "
                f"rmse={row['unclipped_rmse']:.4f} realised={within} "
                f"clipped_dims={row['normalisation_clipped_count']}",
                flush=True,
            )

    csv_path = out_dir / f"matrix_{cli.gesture}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "gesture": cli.gesture,
        "states": cli.states,
        "seeds": cli.seeds,
        "maxiter": cli.maxiter,
        "popsize": cli.popsize,
        "local_maxiter": cli.local_maxiter,
        "per_feature_tolerance": cli.tolerance,
        "n_runs": len(rows),
    }
    (out_dir / f"metadata_{cli.gesture}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(csv_path, flush=True)


if __name__ == "__main__":
    main()
