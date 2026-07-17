"""Fail-fast audit for the revised local-window Space evaluation setup."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

EXPECTED = {
    "wave": (0.06269632, 0.11528429),
    "reach": (0.01198525, 0.05514839),
    "point": (0.02318369, 0.08070654),
}


def check(condition: bool, message: str, failures: list[str]) -> None:
    print(("PASS" if condition else "FAIL") + "  " + message)
    if not condition:
        failures.append(message)


def main() -> None:
    failures: list[str] = []
    config_path = ROOT / "configs" / "normalisation_ranges_balanced_3gestures_by_gesture.json"
    check(config_path.exists(), f"normalisation file exists: {config_path}", failures)
    ranges = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    for gesture, expected in EXPECTED.items():
        item = ranges.get(gesture, {}).get("space_indirectness", {})
        found = (item.get("min"), item.get("max"))
        ok = all(x is not None for x in found) and np.allclose(found, expected, rtol=0, atol=1e-8)
        check(ok, f"{gesture} revised Space range {expected}; found {found}", failures)

    try:
        import robust_laban_normalisation_balanced_3gestures as laban
        source = inspect.getsource(laban.compute_space_indirectness)
        check("window_fraction: float = 0.075" in source, "local-window Space uses fraction 0.075", failures)
        check("1.0 - efficiencies" in source, "Space computes bounded local inefficiency", failures)
        theta = np.linspace(0, 2 * np.pi, 160)
        closed = np.column_stack((0.1 * np.cos(theta), 0.1 * np.sin(theta)))
        value = float(laban.compute_space_indirectness(closed))
        check(np.isfinite(value) and 0 <= value <= 1, f"closed cyclic path is finite and bounded (value={value:.6f})", failures)
    except Exception as exc:
        check(False, f"descriptor import/runtime check: {exc}", failures)

    optimiser = ROOT / "scripts" / "direct_laban_feature_optimizer_spatiotemporal_final.py"
    if optimiser.exists():
        text = optimiser.read_text(encoding="utf-8")
        check("normalisation_ranges_balanced_3gestures_by_gesture.json" in text,
              "inner optimiser defaults to the gesture-specific range file", failures)
    else:
        check(False, f"inner optimiser exists: {optimiser}", failures)

    matrix_runner = ROOT / "scripts" / "debug" / "run_full_realisability_matrix.py"
    if matrix_runner.exists():
        text = matrix_runner.read_text(encoding="utf-8")
        check('"cases" / cli.gesture / state' in text,
              "matrix diagnostics are isolated by gesture/state/seed", failures)
    else:
        check(False, f"matrix runner exists: {matrix_runner}", failures)

    comprehensive = ROOT / "scripts" / "evaluation" / "run_comprehensive_evaluation.py"
    if comprehensive.exists():
        text = comprehensive.read_text(encoding="utf-8")
        check("preflight_revised_space_ranges" in text,
              "comprehensive runner performs descriptor/range preflight", failures)
    else:
        check(False, f"comprehensive runner exists: {comprehensive}", failures)

    if failures:
        print(f"\nSETUP INVALID: {len(failures)} check(s) failed. Do not run the evaluation.")
        raise SystemExit(1)
    print("\nSETUP VALID: revised Space descriptor, ranges, and evaluation wiring agree.")


if __name__ == "__main__":
    main()
