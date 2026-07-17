"""Test multiple complete Laban target profiles with the existing optimiser.

Drop this file into:
    scripts/run_multi_profile_realisation_test.py

Run from the project root, for example:

    python scripts/run_multi_profile_realisation_test.py ^
        --gesture wave ^
        --profiles-json configs/multi_profile_targets.json ^
        --maxiter 90 ^
        --popsize 8 ^
        --local-maxiter 300 ^
        --out outputs/multi_profile_realisation_wave

The JSON file may use either of these formats.

Format A:
{
  "confident": {
    "weight": 0.8,
    "time": 0.75,
    "flow_boundness": 0.65,
    "space_indirectness": 0.2,
    "shape_arcness": 0.75
  }
}

Format B:
{
  "profiles": {
    "confident": {
      "weight": 0.8,
      "time": 0.75,
      "flow_boundness": 0.65,
      "space_indirectness": 0.2,
      "shape_arcness": 0.75
    }
  }
}

Outputs:
    multi_profile_results.csv
    multi_profile_summary.json
    requested_vs_achieved_all_profiles.png
    per_profile_rmse.png
    per_feature_absolute_error.png
    cases/<profile_name>/...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import optimise_laban_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real trajectory optimiser against multiple complete "
            "five-dimensional Laban target profiles."
        )
    )
    parser.add_argument(
        "--gesture",
        choices=["wave", "reach", "point"],
        required=True,
    )
    parser.add_argument(
        "--profiles-json",
        type=str,
        required=True,
        help="JSON file containing named complete Laban target profiles.",
    )
    parser.add_argument("--maxiter", type=int, default=90)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--local-maxiter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--target-state-prefix",
        type=str,
        default="multi_profile_test",
        help=(
            "Metadata prefix passed to the optimiser API. The actual target "
            "profile always comes from the JSON file."
        ),
    )
    return parser.parse_args()


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def validate_unit_interval(value: Any, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value}.")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {value}.")
    return value


def validate_profile(
    profile_name: str,
    payload: Mapping[str, Any],
) -> dict[str, float]:
    missing = [key for key in FEATURE_KEYS if key not in payload]
    extra = [key for key in payload if key not in FEATURE_KEYS]

    if missing:
        raise ValueError(
            f"Profile {profile_name!r} is missing required keys: {missing}"
        )

    if extra:
        print(
            f"Warning: profile {profile_name!r} contains extra keys "
            f"that will be ignored: {extra}"
        )

    return {
        key: validate_unit_interval(
            payload[key],
            f"{profile_name}.{key}",
        )
        for key in FEATURE_KEYS
    }


def load_profiles(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Profiles JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if "profiles" in payload:
        payload = payload["profiles"]

    if not isinstance(payload, dict) or not payload:
        raise ValueError(
            "Profiles JSON must contain a non-empty mapping of "
            "profile name -> five-feature target profile."
        )

    profiles: dict[str, dict[str, float]] = {}
    for name, profile_payload in payload.items():
        if not isinstance(profile_payload, dict):
            raise ValueError(
                f"Profile {name!r} must be a JSON object."
            )
        profiles[str(name)] = validate_profile(
            str(name),
            profile_payload,
        )

    return profiles


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned or "profile"


def run_tests(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profiles_path = resolve_project_path(args.profiles_json)
    out_dir = resolve_project_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = load_profiles(profiles_path)

    print("\n" + "=" * 100)
    print("MULTI-PROFILE REALISATION TEST")
    print("=" * 100)
    print(f"Gesture:            {args.gesture}")
    print(f"Profiles JSON:      {profiles_path}")
    print(f"Number of profiles: {len(profiles)}")
    print(f"Output directory:   {out_dir}")
    print("=" * 100)

    rows: list[dict[str, Any]] = []

    for index, (profile_name, target_profile) in enumerate(
        profiles.items(),
        start=1,
    ):
        print("\n" + "-" * 100)
        print(
            f"[{index}/{len(profiles)}] "
            f"Profile: {profile_name}"
        )
        print("-" * 100)

        print("Requested profile:")
        for key in FEATURE_KEYS:
            print(f"  {key:22s}: {target_profile[key]:.4f}")

        case_out = (
            out_dir
            / "cases"
            / safe_name(profile_name)
        )

        result = optimise_laban_target(
            gesture=args.gesture,
            target_state=(
                f"{args.target_state_prefix}_{profile_name}"
            ),
            target_profile=target_profile,
            out_dir=case_out,
            optimiser_overrides={
                "maxiter": args.maxiter,
                "popsize": args.popsize,
                "local_maxiter": args.local_maxiter,
                "seed": args.seed,
            },
        )

        row: dict[str, Any] = {
            "gesture": args.gesture,
            "profile_name": profile_name,
            "inner_loss": float(result.inner_loss),
            "realisation_rmse": float(result.realisation_rmse),
            "case_output_dir": str(case_out),
        }

        squared_errors = []
        for key in FEATURE_KEYS:
            requested = float(result.requested_profile[key])
            achieved = float(result.achieved_profile[key])
            error = achieved - requested
            abs_error = abs(error)

            row[f"requested_{key}"] = requested
            row[f"achieved_{key}"] = achieved
            row[f"error_{key}"] = error
            row[f"abs_error_{key}"] = abs_error

            squared_errors.append(error ** 2)

        row["mean_absolute_error_all_features"] = float(
            np.mean(
                [
                    row[f"abs_error_{key}"]
                    for key in FEATURE_KEYS
                ]
            )
        )
        row["max_absolute_error_all_features"] = float(
            np.max(
                [
                    row[f"abs_error_{key}"]
                    for key in FEATURE_KEYS
                ]
            )
        )

        rows.append(row)
        save_csv(
            rows,
            out_dir / "multi_profile_results_partial.csv",
        )

        print("\nRequested -> achieved:")
        for key in FEATURE_KEYS:
            print(
                f"  {key:22s}: "
                f"{row[f'requested_{key}']:.4f} "
                f"-> {row[f'achieved_{key}']:.4f} "
                f"(abs err={row[f'abs_error_{key}']:.4f})"
            )

        print(
            f"\nOverall realisation RMSE: "
            f"{row['realisation_rmse']:.4f}"
        )
        print(
            f"Mean absolute error:      "
            f"{row['mean_absolute_error_all_features']:.4f}"
        )
        print(
            f"Max absolute error:       "
            f"{row['max_absolute_error_all_features']:.4f}"
        )

    summary = build_summary(
        gesture=args.gesture,
        profiles_path=profiles_path,
        rows=rows,
    )

    save_csv(
        rows,
        out_dir / "multi_profile_results.csv",
    )

    (out_dir / "multi_profile_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    plot_requested_vs_achieved(rows, out_dir)
    plot_profile_rmse(rows, out_dir)
    plot_feature_errors(rows, out_dir)

    return rows, summary


def save_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    *,
    gesture: str,
    profiles_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    per_feature: dict[str, Any] = {}

    for key in FEATURE_KEYS:
        requested = np.asarray(
            [row[f"requested_{key}"] for row in rows],
            dtype=float,
        )
        achieved = np.asarray(
            [row[f"achieved_{key}"] for row in rows],
            dtype=float,
        )
        abs_errors = np.abs(achieved - requested)

        if len(requested) >= 2 and np.std(requested) > 1e-12:
            correlation = float(
                np.corrcoef(
                    requested,
                    achieved,
                )[0, 1]
            )
            slope = float(
                np.polyfit(
                    requested,
                    achieved,
                    deg=1,
                )[0]
            )
        else:
            correlation = None
            slope = None

        per_feature[key] = {
            "mean_absolute_error": float(
                np.mean(abs_errors)
            ),
            "max_absolute_error": float(
                np.max(abs_errors)
            ),
            "requested_achieved_correlation": correlation,
            "requested_achieved_linear_slope": slope,
        }

    profile_ranking = sorted(
        [
            {
                "profile_name": row["profile_name"],
                "realisation_rmse": row["realisation_rmse"],
                "mean_absolute_error_all_features": (
                    row["mean_absolute_error_all_features"]
                ),
                "max_absolute_error_all_features": (
                    row["max_absolute_error_all_features"]
                ),
            }
            for row in rows
        ],
        key=lambda item: item["realisation_rmse"],
    )

    return {
        "gesture": gesture,
        "profiles_json": str(profiles_path),
        "number_of_profiles": len(rows),
        "mean_realisation_rmse": float(
            np.mean(
                [row["realisation_rmse"] for row in rows]
            )
        ),
        "max_realisation_rmse": float(
            np.max(
                [row["realisation_rmse"] for row in rows]
            )
        ),
        "per_feature": per_feature,
        "profile_ranking_best_to_worst": profile_ranking,
    }


def plot_requested_vs_achieved(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    profile_names = [
        row["profile_name"]
        for row in rows
    ]
    x = np.arange(len(profile_names))
    width = 0.36

    for key in FEATURE_KEYS:
        requested = [
            row[f"requested_{key}"]
            for row in rows
        ]
        achieved = [
            row[f"achieved_{key}"]
            for row in rows
        ]

        plt.figure(
            figsize=(
                max(8, len(profile_names) * 1.25),
                5,
            )
        )
        plt.bar(
            x - width / 2,
            requested,
            width,
            label="Requested",
        )
        plt.bar(
            x + width / 2,
            achieved,
            width,
            label="Achieved",
        )
        plt.xticks(
            x,
            profile_names,
            rotation=45,
            ha="right",
        )
        plt.ylim(0, 1)
        plt.ylabel("Normalised value")
        plt.title(
            f"Requested vs achieved: {key}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir
            / f"requested_vs_achieved_{key}.png",
            dpi=180,
        )
        plt.close()


def plot_profile_rmse(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    names = [
        row["profile_name"]
        for row in rows
    ]
    rmse = [
        row["realisation_rmse"]
        for row in rows
    ]

    positions = np.arange(len(names))

    plt.figure(
        figsize=(
            max(8, len(names) * 1.25),
            5,
        )
    )
    plt.bar(
        positions,
        rmse,
    )
    plt.xticks(
        positions,
        names,
        rotation=45,
        ha="right",
    )
    plt.ylabel("Realisation RMSE")
    plt.title(
        "Overall profile realisation error"
    )
    plt.tight_layout()
    plt.savefig(
        out_dir / "per_profile_rmse.png",
        dpi=180,
    )
    plt.close()


def plot_feature_errors(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    names = [
        row["profile_name"]
        for row in rows
    ]
    x = np.arange(len(names))

    plt.figure(
        figsize=(
            max(10, len(names) * 1.5),
            6,
        )
    )

    bottom = np.zeros(len(rows), dtype=float)

    for key in FEATURE_KEYS:
        values = np.asarray(
            [
                row[f"abs_error_{key}"]
                for row in rows
            ],
            dtype=float,
        )
        plt.bar(
            x,
            values,
            bottom=bottom,
            label=key,
        )
        bottom += values

    plt.xticks(
        x,
        names,
        rotation=45,
        ha="right",
    )
    plt.ylabel("Absolute error")
    plt.title(
        "Per-feature contribution to profile realisation error"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_dir / "per_feature_absolute_error.png",
        dpi=180,
    )
    plt.close()


def print_summary(
    summary: dict[str, Any],
) -> None:
    print("\n" + "=" * 100)
    print("MULTI-PROFILE SUMMARY")
    print("=" * 100)

    print(
        f"Mean realisation RMSE: "
        f"{summary['mean_realisation_rmse']:.4f}"
    )
    print(
        f"Max realisation RMSE:  "
        f"{summary['max_realisation_rmse']:.4f}"
    )

    print("\nFeature-level errors:")
    for key, metrics in summary["per_feature"].items():
        corr = metrics[
            "requested_achieved_correlation"
        ]
        slope = metrics[
            "requested_achieved_linear_slope"
        ]

        corr_text = (
            "undefined"
            if corr is None
            else f"{corr:.4f}"
        )
        slope_text = (
            "undefined"
            if slope is None
            else f"{slope:.4f}"
        )

        print(f"\n{key}")
        print(
            f"  Mean abs error:   "
            f"{metrics['mean_absolute_error']:.4f}"
        )
        print(
            f"  Max abs error:    "
            f"{metrics['max_absolute_error']:.4f}"
        )
        print(
            f"  Correlation:      {corr_text}"
        )
        print(
            f"  Response slope:   {slope_text}"
        )

    print("\nProfiles ranked best -> worst:")
    for rank, item in enumerate(
        summary["profile_ranking_best_to_worst"],
        start=1,
    ):
        print(
            f"  {rank:2d}. "
            f"{item['profile_name']:20s} "
            f"RMSE={item['realisation_rmse']:.4f}"
        )

    print("=" * 100)


def main() -> None:
    args = parse_args()
    _, summary = run_tests(args)
    print_summary(summary)


if __name__ == "__main__":
    main()
