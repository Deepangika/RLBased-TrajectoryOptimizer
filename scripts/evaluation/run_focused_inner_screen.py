"""Screen difficult gesture/affect pairs with the inner optimiser only."""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import sys
import uuid
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laban_rl.config import FEATURE_KEYS
from laban_rl.targets import TARGET_PROFILES
from scripts.train_cem_contextual_bandit import _build_optimiser_overrides

WAVE_STATES = ("anger", "disgust", "fear", "sadness")
SEEDS = (7, 17, 27)
WAVE_FLOW_TARGET_WEIGHTS = (0.35, 0.75, 1.0, 1.5)
TIMING_BASES = (4, 5, 6)
BECKON_TIMING_BASES = (5, 6)
BECKON_MAXITERS = (45, 75)
BECKON_POPSIZES = (5, 8)
TOLERANCE = 0.10
SCREEN_VERSION = 2


def load_optimiser():
    path = ROOT / "scripts" / "direct_laban_feature_optimizer_spatiotemporal_final.py"
    spec = spec_from_file_location("focused_inner_optimiser", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load optimiser: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.save_outputs = lambda result, out_path, arm: None
    return module


def wave_overrides(
    *,
    seed: int,
    flow_target_weight: float,
    timing_basis: int,
) -> dict[str, Any]:
    args = SimpleNamespace(
        maxiter=45,
        popsize=5,
        local_maxiter=100,
        de_mutation=0.5,
        de_recombination=0.65,
        seed=seed,
        wave_flow_target_weight=flow_target_weight,
    )
    overrides = _build_optimiser_overrides(args, "wave")
    overrides["n_timing_basis"] = timing_basis
    return overrides


def beckon_overrides(
    *,
    state: str,
    seed: int,
    maxiter: int,
    popsize: int,
    timing_basis: int,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "maxiter": maxiter,
        "popsize": popsize,
        "local_maxiter": 100,
        "de_mutation": 0.5,
        "de_recombination": 0.65,
        "seed": seed,
        "n_timing_basis": timing_basis,
        "time_target_weight": 0.10,
        "flow_boundness_target_weight": 0.10,
    }
    if state == "fear":
        overrides["shape_arcness_target_weight"] = 0.06
    return overrides


def build_cases(phase: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if phase in ("wave", "all"):
        for state, flow_weight, timing_basis, seed in product(
            WAVE_STATES,
            WAVE_FLOW_TARGET_WEIGHTS,
            TIMING_BASES,
            SEEDS,
        ):
            cases.append(
                {
                    "gesture": "wave",
                    "state": state,
                    "seed": seed,
                    "flow_target_weight": flow_weight,
                    "timing_basis": timing_basis,
                    "maxiter": 45,
                    "popsize": 5,
                    "overrides": wave_overrides(
                        seed=seed,
                        flow_target_weight=flow_weight,
                        timing_basis=timing_basis,
                    ),
                }
            )
    beckon_states = []
    if phase in ("beckon-fear", "all"):
        beckon_states.append("fear")
    if phase in ("beckon-surprise", "all"):
        beckon_states.append("surprise")
    for state, maxiter, popsize, timing_basis, seed in product(
        beckon_states,
        BECKON_MAXITERS,
        BECKON_POPSIZES,
        BECKON_TIMING_BASES,
        SEEDS,
    ):
        cases.append(
            {
                "gesture": "beckon",
                "state": state,
                "seed": seed,
                "flow_target_weight": 0.10,
                "timing_basis": timing_basis,
                "maxiter": maxiter,
                "popsize": popsize,
                "overrides": beckon_overrides(
                    state=state,
                    seed=seed,
                    maxiter=maxiter,
                    popsize=popsize,
                    timing_basis=timing_basis,
                ),
            }
        )
    return cases


def case_id(case: dict[str, Any]) -> str:
    return (
        f"v{SCREEN_VERSION}-{case['gesture']}-{case['state']}"
        f"-flow_{case['flow_target_weight']:g}"
        f"-timing_{case['timing_basis']}"
        f"-iter_{case['maxiter']}"
        f"-pop_{case['popsize']}"
        f"-seed_{case['seed']}"
    )


def finite(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def evaluate_case(module: Any, case: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    parser_args = module.build_parser().parse_args([])
    parser_args.gesture = case["gesture"]
    parser_args.target = case["state"]
    parser_args.out = str(out_dir / "optimiser_outputs")
    for name, value in case["overrides"].items():
        setattr(parser_args, name, value)

    with contextlib.redirect_stdout(io.StringIO()):
        result = module.optimise(parser_args)

    target = TARGET_PROFILES[case["state"]]
    achieved = result["var_norm_unclipped"]
    errors = {
        key: abs(float(achieved[key]) - float(target[key]))
        for key in FEATURE_KEYS
    }
    values = np.asarray(list(achieved.values()), dtype=float)
    reward_info = result["reward_info"]
    path_ratio = float(reward_info["path_length_ratio"])
    joint_limit_error = float(reward_info["joint_limit_error"])
    valid = bool(np.all(np.isfinite(values)))
    feature_acceptable = bool(valid and max(errors.values()) <= TOLERANCE)
    path_preserved = bool(0.70 <= path_ratio <= 1.30)
    joint_safe = bool(joint_limit_error <= 1e-8)
    row = {
        key: value for key, value in case.items() if key != "overrides"
    }
    row.update(
        {
            "case_id": case_id(case),
            "valid_features": valid,
            "feature_acceptable": feature_acceptable,
            "path_preserved": path_preserved,
            "joint_limits_satisfied": joint_safe,
            "strictly_feasible": bool(
                valid and feature_acceptable and path_preserved and joint_safe
            ),
            "rmse": finite(
                np.sqrt(np.mean(np.square(list(errors.values()))))
            ),
            "max_abs_feature_error": finite(max(errors.values())),
            "max_error_feature": max(errors, key=errors.get),
            "path_length_ratio": finite(path_ratio),
            "nearest_path_mse": finite(reward_info["nearest_path_mse"]),
            "nearest_path_max_dist": finite(
                reward_info["nearest_path_max_dist"]
            ),
            "endpoint_error": finite(reward_info["endpoint_error"]),
            "direction_error": finite(reward_info["direction_error"]),
            "smoothness_error": finite(reward_info["smoothness_error"]),
            "joint_limit_error": finite(joint_limit_error),
            "flow_abs_error": finite(errors["flow_boundness"]),
            "time_abs_error": finite(errors["time"]),
            "shape_abs_error": finite(errors["shape_arcness"]),
        }
    )
    for key in FEATURE_KEYS:
        row[f"target_{key}"] = float(target[key])
        row[f"achieved_{key}"] = finite(achieved[key])
        row[f"error_{key}"] = finite(errors[key])
    return row


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = (
        "gesture",
        "state",
        "flow_target_weight",
        "timing_basis",
        "maxiter",
        "popsize",
    )
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)

    summaries = []
    metric_names = (
        "rmse",
        "max_abs_feature_error",
        "flow_abs_error",
        "time_abs_error",
        "shape_abs_error",
        "nearest_path_mse",
        "nearest_path_max_dist",
        "endpoint_error",
        "direction_error",
        "smoothness_error",
        "joint_limit_error",
    )
    for group_key, group in groups.items():
        summary = dict(zip(keys, group_key))
        summary["seeds_completed"] = len(group)
        summary["strictly_feasible_seeds"] = sum(
            bool(row["strictly_feasible"]) for row in group
        )
        summary["all_seeds_strictly_feasible"] = (
            summary["strictly_feasible_seeds"] == len(SEEDS)
        )
        for metric in metric_names:
            values = [float(row[metric]) for row in group if row[metric] is not None]
            summary[f"mean_{metric}"] = float(np.mean(values))
            summary[f"worst_{metric}"] = float(np.max(values))
        path_ratios = [
            float(row["path_length_ratio"])
            for row in group
            if row["path_length_ratio"] is not None
        ]
        summary["mean_path_length_ratio"] = float(np.mean(path_ratios))
        summary["minimum_path_length_ratio"] = float(np.min(path_ratios))
        summary["maximum_path_length_ratio"] = float(np.max(path_ratios))
        summary["worst_path_length_ratio_deviation"] = float(
            np.max(np.abs(np.asarray(path_ratios) - 1.0))
        )
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda item: (
            item["gesture"],
            item["state"],
            -item["strictly_feasible_seeds"],
            item["worst_max_abs_feature_error"],
            item["mean_nearest_path_mse"],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("wave", "beckon-fear", "beckon-surprise", "all"),
        required=True,
    )
    parser.add_argument(
        "--out",
        default="outputs/focused_inner_screen",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    module = load_optimiser()
    rows_by_id: dict[str, dict[str, Any]] = {}
    cases = build_cases(args.phase)
    selected_ids = {case_id(case) for case in cases}
    legacy_raw_path = out_dir / "raw_results.json"
    if legacy_raw_path.exists():
        for row in json.loads(legacy_raw_path.read_text(encoding="utf-8")):
            identifier = row["case_id"]
            if identifier not in selected_ids:
                continue
            result_path = out_dir / "cases" / identifier / "result.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            if not result_path.exists():
                write_json_atomic(result_path, row)

    for identifier in sorted(selected_ids):
        result_path = out_dir / "cases" / identifier / "result.json"
        if result_path.exists():
            rows_by_id[identifier] = json.loads(
                result_path.read_text(encoding="utf-8")
            )

    phase_raw_path = out_dir / f"{args.phase}_raw_results.json"
    for index, case in enumerate(cases, start=1):
        identifier = case_id(case)
        if identifier in rows_by_id:
            print(f"SKIP {index}/{len(cases)} {identifier}", flush=True)
            continue
        print(f"RUN  {index}/{len(cases)} {identifier}", flush=True)
        row = evaluate_case(module, case, out_dir / "cases" / identifier)
        rows_by_id[identifier] = row
        write_json_atomic(
            out_dir / "cases" / identifier / "result.json",
            row,
        )
        print(
            f"  feasible={row['strictly_feasible']} "
            f"rmse={row['rmse']:.4f} max={row['max_abs_feature_error']:.4f} "
            f"flow={row['flow_abs_error']:.4f} ratio={row['path_length_ratio']:.3f}",
            flush=True,
        )

    phase_rows = []
    for identifier in sorted(selected_ids):
        result_path = out_dir / "cases" / identifier / "result.json"
        if result_path.exists():
            phase_rows.append(json.loads(result_path.read_text(encoding="utf-8")))
    write_json_atomic(phase_raw_path, phase_rows)
    summaries = aggregate(phase_rows)
    write_csv(out_dir / f"{args.phase}_raw.csv", phase_rows)
    write_csv(out_dir / f"{args.phase}_aggregate.csv", summaries)
    write_json_atomic(
        out_dir / f"{args.phase}_aggregate.json",
        summaries,
    )
    print(
        f"Completed {len(phase_rows)}/{len(cases)} runs; "
        f"{sum(row['strictly_feasible'] for row in phase_rows)} strictly feasible.",
        flush=True,
    )


if __name__ == "__main__":
    main()
