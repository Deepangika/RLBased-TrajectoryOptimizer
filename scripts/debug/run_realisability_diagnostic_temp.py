from __future__ import annotations

import contextlib
import csv
import io
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laban_rl.config import FEATURE_KEYS
from laban_rl.targets import TARGET_PROFILES


spec = spec_from_file_location(
    "diagnostic_optimiser",
    ROOT / "scripts" / "direct_laban_feature_optimizer_spatiotemporal_final.py",
)
module = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

# Avoid expensive plots/GIFs for this numerical diagnostic.
module.save_outputs = lambda result, out_path, arm: None

target = TARGET_PROFILES["confident"]
rows = []

for gesture in ["reach", "point"]:
    for mode in ["constrained"]:
        for seed in [7, 27]:
            args = module.build_parser().parse_args([])
            args.gesture = gesture
            args.target = "confident"
            args.seed = seed
            args.maxiter = 45
            args.popsize = 5
            args.local_maxiter = 100
            args.out = str(Path("/tmp/realisability_high") / gesture / mode / str(seed))

            if mode == "feature_only":
                for name in [
                    "preserve_weight", "nearest_path_weight", "endpoint_weight",
                    "direction_weight", "path_length_weight", "detour_weight",
                    "max_dev_weight", "smooth_weight", "joint_limit_weight",
                    "coeff_weight", "time_coeff_weight", "time_warp_weight",
                    "time_roughness_weight",
                ]:
                    setattr(args, name, 0.0)

            with contextlib.redirect_stdout(io.StringIO()):
                result = module.optimise(args)

            achieved = np.asarray([result["var_norm"][k] for k in FEATURE_KEYS], dtype=float)
            requested = np.asarray([target[k] for k in FEATURE_KEYS], dtype=float)
            errors = np.abs(achieved - requested)
            action = np.asarray(result["action"], dtype=float)
            info = result["reward_info"]
            row = {
                "gesture": gesture,
                "mode": mode,
                "seed": seed,
                "rmse": float(np.sqrt(np.mean((achieved - requested) ** 2))),
                "max_abs_error": float(np.max(errors)),
                "all_within_0.10": bool(np.all(errors <= 0.10)),
                "saturation_rate_0.95": float(np.mean(np.abs(action) >= 0.95)),
                "joint_limit_error": info["joint_limit_error"],
                "path_length_ratio": info["path_length_ratio"],
                "nearest_path_mse": info["nearest_path_mse"],
            }
            for i, key in enumerate(FEATURE_KEYS):
                row[f"achieved_{key}"] = achieved[i]
                row[f"error_{key}"] = errors[i]
            rows.append(row)
            print(gesture, mode, seed, f"rmse={row['rmse']:.4f}", f"maxerr={row['max_abs_error']:.4f}")

out = ROOT / "outputs" / "realisability_diagnostic_confident_high_budget_extra_seeds.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(out)
