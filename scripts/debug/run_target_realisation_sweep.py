"""Target-realisation sweep for the spatiotemporal Laban optimiser.

Drop this file into:
    scripts/run_target_realisation_sweep.py

Run from the project root, for example:

    python scripts/run_target_realisation_sweep.py ^
        --gesture wave ^
        --levels 0.2 0.4 0.6 0.8 ^
        --maxiter 90 ^
        --popsize 8 ^
        --local-maxiter 300 ^
        --out outputs/target_realisation_sweep_wave

The script varies ONE target Laban dimension at a time while keeping the
other dimensions fixed at the reference gesture's normalised features.

If a reference feature is NaN (currently possible for space_indirectness),
the script replaces only that baseline value with --nan-reference-default
(default: 0.5) and records the replacement in the outputs.

Outputs:
    sweep_results.csv
    sweep_summary.json
    reference_profile_used.json
    requested_vs_achieved_<feature>.png
    cross_feature_response_<feature>.png
    overall_realisation_rmse.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import optimise_laban_target


OPTIMISER_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "direct_laban_feature_optimizer_spatiotemporal_final.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep one target Laban dimension at a time and measure how "
            "accurately the existing optimiser realises each requested value."
        )
    )
    parser.add_argument(
        "--gesture",
        choices=["wave", "reach", "point"],
        required=True,
    )
    parser.add_argument(
        "--target-state",
        default="diagnostic_sweep",
        help=(
            "Metadata only. The externally supplied sweep profiles determine "
            "the actual optimiser target."
        ),
    )
    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8],
        help="Normalised target values to test for each swept dimension.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURE_KEYS,
        default=list(FEATURE_KEYS),
        help="Subset of Laban dimensions to sweep.",
    )
    parser.add_argument(
        "--nan-reference-default",
        type=float,
        default=0.5,
        help=(
            "Baseline value used only when a reference normalised feature is "
            "NaN or infinite."
        ),
    )
    parser.add_argument(
        "--baseline-profile-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing a complete five-feature baseline "
            "profile. When supplied, this replaces reference-derived baseline "
            "values."
        ),
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
    return parser.parse_args()


def validate_unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    return value


def load_optimiser_module():
    if not OPTIMISER_SCRIPT.exists():
        raise FileNotFoundError(
            f"Could not find optimiser script: {OPTIMISER_SCRIPT}"
        )

    spec = spec_from_file_location(
        "laban_spatiotemporal_optimiser_for_sweep",
        OPTIMISER_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not import optimiser script: {OPTIMISER_SCRIPT}"
        )

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_reference_profile(gesture: str) -> tuple[dict[str, float], dict[str, float]]:
    """Compute reference raw and normalised features without running optimisation."""
    module = load_optimiser_module()

    args = module.build_parser().parse_args([])
    args.gesture = gesture

    arm = module.laban.ArmConfig(
        n_points=160,
        duration=2.0,
        l1=0.30,
        l2=0.25,
    )
    filter_config = module.laban.FilterConfig(
        enabled=True,
        cutoff_hz=5.0,
        order=4,
    )

    ranges_path = args.ranges
    if not Path(ranges_path).exists():
        candidate = PROJECT_ROOT / ranges_path
        if candidate.exists():
            ranges_path = str(candidate)

    ranges = module.load_ranges_or_default(ranges_path)
    q_ref = module.make_reference_trajectory(
        gesture_type=gesture,
        arm=arm,
    )
    ref_raw, ref_norm = module.compute_raw_and_norm_features(
        q_ref,
        arm,
        filter_config,
        ranges,
    )

    raw = {key: float(ref_raw.get(key, np.nan)) for key in FEATURE_KEYS}
    norm = {key: float(ref_norm.get(key, np.nan)) for key in FEATURE_KEYS}
    return raw, norm


def load_baseline_from_json(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if "laban_profile" in payload:
        payload = payload["laban_profile"]

    missing = [key for key in FEATURE_KEYS if key not in payload]
    if missing:
        raise ValueError(
            f"Baseline profile JSON is missing keys: {missing}"
        )

    return {
        key: validate_unit_interval(payload[key], f"baseline[{key}]")
        for key in FEATURE_KEYS
    }


def make_baseline_profile(
    *,
    reference_norm: dict[str, float],
    baseline_profile_json: str | None,
    nan_reference_default: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    fallback = validate_unit_interval(
        nan_reference_default,
        "--nan-reference-default",
    )

    if baseline_profile_json is not None:
        path = Path(baseline_profile_json)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        baseline = load_baseline_from_json(path)
        metadata = {
            "source": "json",
            "path": str(path),
            "replaced_nonfinite_reference_features": [],
        }
        return baseline, metadata

    baseline: dict[str, float] = {}
    replaced: list[str] = []

    for key in FEATURE_KEYS:
        value = float(reference_norm[key])
        if math.isfinite(value):
            baseline[key] = float(np.clip(value, 0.0, 1.0))
        else:
            baseline[key] = fallback
            replaced.append(key)

    metadata = {
        "source": "reference_normalised_features",
        "nan_reference_default": fallback,
        "replaced_nonfinite_reference_features": replaced,
    }
    return baseline, metadata


def run_sweep(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    levels = [
        validate_unit_interval(level, f"--levels[{index}]")
        for index, level in enumerate(args.levels)
    ]

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_raw, reference_norm = compute_reference_profile(args.gesture)

    baseline_profile, baseline_metadata = make_baseline_profile(
        reference_norm=reference_norm,
        baseline_profile_json=args.baseline_profile_json,
        nan_reference_default=args.nan_reference_default,
    )

    reference_payload = {
        "gesture": args.gesture,
        "reference_raw_features": reference_raw,
        "reference_normalised_features": reference_norm,
        "baseline_profile_used": baseline_profile,
        "baseline_metadata": baseline_metadata,
    }
    (out_dir / "reference_profile_used.json").write_text(
        json.dumps(reference_payload, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    print("\n" + "=" * 96)
    print("TARGET-REALISATION SWEEP")
    print("=" * 96)
    print(f"Gesture: {args.gesture}")
    print("\nReference normalised features:")
    for key in FEATURE_KEYS:
        print(f"  {key:22s}: {reference_norm[key]}")

    print("\nBaseline profile used for non-swept dimensions:")
    for key in FEATURE_KEYS:
        suffix = ""
        if key in baseline_metadata["replaced_nonfinite_reference_features"]:
            suffix = "  <-- fallback because reference was non-finite"
        print(f"  {key:22s}: {baseline_profile[key]:.4f}{suffix}")

    rows: list[dict[str, Any]] = []

    total_runs = len(args.features) * len(levels)
    run_number = 0

    for swept_feature in args.features:
        for level in levels:
            run_number += 1

            target_profile = dict(baseline_profile)
            target_profile[swept_feature] = level

            case_name = (
                f"{swept_feature}_target_{level:.3f}"
                .replace(".", "p")
            )
            case_out = out_dir / "cases" / case_name

            print("\n" + "-" * 96)
            print(
                f"[{run_number}/{total_runs}] "
                f"Sweeping {swept_feature}: target={level:.3f}"
            )
            print("-" * 96)

            result = optimise_laban_target(
                gesture=args.gesture,
                target_state=args.target_state,
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
                "target_state": args.target_state,
                "swept_feature": swept_feature,
                "swept_target": level,
                "inner_loss": result.inner_loss,
                "realisation_rmse": result.realisation_rmse,
                "case_output_dir": str(case_out),
            }

            for key in FEATURE_KEYS:
                requested = float(result.requested_profile[key])
                achieved = float(result.achieved_profile[key])
                row[f"requested_{key}"] = requested
                row[f"achieved_{key}"] = achieved
                row[f"error_{key}"] = achieved - requested
                row[f"abs_error_{key}"] = abs(achieved - requested)

            rows.append(row)

            print(
                f"Requested {swept_feature}: {level:.4f} | "
                f"Achieved: {result.achieved_profile[swept_feature]:.4f} | "
                f"Abs error: "
                f"{abs(result.achieved_profile[swept_feature] - level):.4f}"
            )
            print(f"Overall realisation RMSE: {result.realisation_rmse:.4f}")

            save_csv(rows, out_dir / "sweep_results_partial.csv")

    summary = build_summary(
        rows=rows,
        gesture=args.gesture,
        levels=levels,
        features=args.features,
        baseline_profile=baseline_profile,
        baseline_metadata=baseline_metadata,
    )

    save_csv(rows, out_dir / "sweep_results.csv")
    (out_dir / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    plot_results(rows, args.features, out_dir)
    return rows, summary


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def linear_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) < 1e-12:
        return None
    slope = np.polyfit(x, y, deg=1)[0]
    return float(slope)


def build_summary(
    *,
    rows: list[dict[str, Any]],
    gesture: str,
    levels: list[float],
    features: list[str],
    baseline_profile: dict[str, float],
    baseline_metadata: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "gesture": gesture,
        "levels": levels,
        "features_swept": features,
        "baseline_profile": baseline_profile,
        "baseline_metadata": baseline_metadata,
        "per_feature": {},
    }

    for feature in features:
        feature_rows = [
            row for row in rows
            if row["swept_feature"] == feature
        ]
        requested = np.asarray(
            [row["swept_target"] for row in feature_rows],
            dtype=float,
        )
        achieved = np.asarray(
            [row[f"achieved_{feature}"] for row in feature_rows],
            dtype=float,
        )
        abs_errors = np.abs(achieved - requested)

        summary["per_feature"][feature] = {
            "mean_absolute_error": float(np.mean(abs_errors)),
            "max_absolute_error": float(np.max(abs_errors)),
            "requested_achieved_correlation": safe_correlation(
                requested,
                achieved,
            ),
            "requested_achieved_linear_slope": linear_slope(
                requested,
                achieved,
            ),
            "achieved_min": float(np.min(achieved)),
            "achieved_max": float(np.max(achieved)),
            "achieved_range": float(np.max(achieved) - np.min(achieved)),
            "fraction_at_lower_boundary": float(
                np.mean(achieved <= 1e-6)
            ),
            "fraction_at_upper_boundary": float(
                np.mean(achieved >= 1.0 - 1e-6)
            ),
        }

    return summary


def plot_results(
    rows: list[dict[str, Any]],
    features: list[str],
    out_dir: Path,
) -> None:
    for swept_feature in features:
        feature_rows = sorted(
            [
                row for row in rows
                if row["swept_feature"] == swept_feature
            ],
            key=lambda row: row["swept_target"],
        )

        requested = np.asarray(
            [row["swept_target"] for row in feature_rows],
            dtype=float,
        )
        achieved = np.asarray(
            [
                row[f"achieved_{swept_feature}"]
                for row in feature_rows
            ],
            dtype=float,
        )

        plt.figure(figsize=(7, 5))
        plt.plot(
            requested,
            achieved,
            marker="o",
            linewidth=2,
            label="Achieved",
        )
        plt.plot(
            requested,
            requested,
            linestyle="--",
            label="Ideal y = x",
        )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel(f"Requested {swept_feature}")
        plt.ylabel(f"Achieved {swept_feature}")
        plt.title(
            f"Target realisation: {swept_feature}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir
            / f"requested_vs_achieved_{swept_feature}.png",
            dpi=180,
        )
        plt.close()

        plt.figure(figsize=(8, 5))
        for observed_feature in FEATURE_KEYS:
            observed = np.asarray(
                [
                    row[f"achieved_{observed_feature}"]
                    for row in feature_rows
                ],
                dtype=float,
            )
            plt.plot(
                requested,
                observed,
                marker="o",
                label=observed_feature,
            )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel(f"Requested {swept_feature}")
        plt.ylabel("Achieved normalised feature")
        plt.title(
            f"Cross-feature response while sweeping {swept_feature}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir
            / f"cross_feature_response_{swept_feature}.png",
            dpi=180,
        )
        plt.close()

    labels = []
    rmse_values = []
    for row in rows:
        labels.append(
            f"{row['swept_feature']}\n{row['swept_target']:.2f}"
        )
        rmse_values.append(row["realisation_rmse"])

    plt.figure(figsize=(max(10, len(labels) * 0.55), 5))
    positions = np.arange(len(labels))
    plt.bar(positions, rmse_values)
    plt.xticks(positions, labels, rotation=45, ha="right")
    plt.ylabel("Overall realisation RMSE")
    plt.title("Realisation error across sweep cases")
    plt.tight_layout()
    plt.savefig(
        out_dir / "overall_realisation_rmse.png",
        dpi=180,
    )
    plt.close()


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 96)
    print("SWEEP SUMMARY")
    print("=" * 96)

    for feature, metrics in summary["per_feature"].items():
        corr = metrics["requested_achieved_correlation"]
        slope = metrics["requested_achieved_linear_slope"]

        corr_text = "undefined" if corr is None else f"{corr:.4f}"
        slope_text = "undefined" if slope is None else f"{slope:.4f}"

        print(f"\n{feature}")
        print(
            f"  Mean absolute error:      "
            f"{metrics['mean_absolute_error']:.4f}"
        )
        print(
            f"  Max absolute error:       "
            f"{metrics['max_absolute_error']:.4f}"
        )
        print(
            f"  Requested-achieved corr:  {corr_text}"
        )
        print(
            f"  Linear response slope:    {slope_text}"
        )
        print(
            f"  Achieved range:           "
            f"{metrics['achieved_range']:.4f}"
        )
        print(
            f"  Fraction at upper bound:  "
            f"{metrics['fraction_at_upper_boundary']:.2f}"
        )

    print("\nInterpretation guide:")
    print("  Good controllability: low error, correlation near +1, slope near 1.")
    print("  Weak controllability: small achieved range or slope near 0.")
    print("  Saturation: many outputs exactly at 0 or 1.")
    print("  Coupling: inspect cross_feature_response_<feature>.png.")
    print("=" * 96)


def main() -> None:
    args = parse_args()
    _, summary = run_sweep(args)
    print_summary(summary)


if __name__ == "__main__":
    main()
