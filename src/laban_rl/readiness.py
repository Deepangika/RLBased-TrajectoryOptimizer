"""Automated pre-human-validation readiness audit."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from laban_rl.config import FEATURE_KEYS, GESTURE_TYPES
from laban_rl.hygiene import credential_presence_check
from laban_rl.perceptual_bandit.evaluation_cache import CACHE_FORMAT_VERSION
from laban_rl.perceptual_bandit.experiment import cases_from_matrix


READINESS_FORMAT_VERSION = 1
REQUIRED_IGNORE_MARKERS = (
    "outputs/",
    ".env",
    "*.key",
    "*.pem",
    "*cache*/",
    "*.mp4",
    "*.gif",
    "generated_calibration/",
)


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
    *,
    blocking: bool = True,
) -> None:
    checks.append(
        {
            "name": name,
            "status": "pass" if passed else ("fail" if blocking else "warning"),
            "blocking": blocking,
            "detail": detail,
        }
    )


def _read_realisability(path: Path) -> bool:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return bool(payload)
    with path.open(encoding="utf-8", newline="") as handle:
        return bool(list(csv.DictReader(handle)))


def audit_readiness(
    root: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    repository = Path(root).resolve()
    checks: list[dict[str, Any]] = []

    tests = dict(manifest.get("tests") or {})
    _check(
        checks,
        "tests",
        tests.get("status") == "passed",
        "Recorded full test status must be 'passed'.",
    )

    calibration_path = _resolve(repository, manifest.get("calibration"))
    calibration_ok = False
    if calibration_path and calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration_ok = all(
            gesture in calibration
            and all(feature in calibration[gesture] for feature in FEATURE_KEYS)
            for gesture in GESTURE_TYPES
        )
    _check(
        checks,
        "calibration",
        calibration_ok,
        "Calibration must cover every configured gesture and Laban feature.",
    )

    matrix_path = _resolve(repository, manifest.get("matrix"))
    paired_path = _resolve(repository, manifest.get("paired_results"))
    matrix_cases = []
    paired: dict[str, Any] = {}
    if matrix_path and matrix_path.exists():
        matrix_cases = cases_from_matrix(
            json.loads(matrix_path.read_text(encoding="utf-8"))
        )
    if paired_path and paired_path.exists():
        paired = json.loads(paired_path.read_text(encoding="utf-8"))
    records = list(paired.get("records") or [])
    expected_ids = {case.candidate_id for case in matrix_cases}
    observed_ids = {str(record.get("candidate_id")) for record in records}
    _check(
        checks,
        "gesture_vad_matrix_coverage",
        bool(expected_ids) and expected_ids == observed_ids,
        f"Expected {len(expected_ids)} paired cases; found {len(observed_ids)}.",
    )

    realisability_path = _resolve(
        repository, manifest.get("realisability_summary")
    )
    realisability_ok = bool(
        realisability_path
        and realisability_path.exists()
        and _read_realisability(realisability_path)
    )
    _check(
        checks,
        "realisability_summary",
        realisability_ok,
        "A non-empty realizability summary is required.",
    )

    minimum_repeats = int(manifest.get("minimum_repeats", 2))
    reliability_ok = bool(records)
    ambiguity_metrics_ok = bool(records)
    named_distribution_coverage: list[float] = []
    for record in records:
        reliability = record.get("reliability")
        if reliability is None:
            reliability_ok = False
            ambiguity_metrics_ok = False
            continue
        repeat_count = int(
            (reliability.get("repeat_reliability") or {}).get(
                "repeat_count", 0
            )
        )
        reliability_ok &= repeat_count >= minimum_repeats
        ambiguity_metrics_ok &= all(
            key in reliability
            for key in (
                "categorical_distribution_coverage",
                "categorical_unambiguous_coverage",
                "ambiguous_category_count",
                "missing_category_count",
            )
        )
        if record.get("context", {}).get("target_mode") == "named":
            named_distribution_coverage.append(
                float(reliability.get("categorical_distribution_coverage", 0.0))
            )
    reliability_status = (paired.get("test_retest_reliability") or {}).get(
        "status"
    )
    reliability_ok &= reliability_status in ("ok", "partial")
    _check(
        checks,
        "paired_repeat_reliability",
        reliability_ok,
        f"Each candidate needs >= {minimum_repeats} repeats and measured reliability.",
    )
    minimum_category_coverage = float(
        manifest.get("minimum_categorical_distribution_coverage", 1.0)
    )
    ambiguity_ok = ambiguity_metrics_ok and all(
        value >= minimum_category_coverage
        for value in named_distribution_coverage
    )
    _check(
        checks,
        "categorical_ambiguity_coverage",
        ambiguity_ok,
        "Named targets need explicit distribution/ambiguous/missing coverage metrics.",
    )

    cache_path = _resolve(repository, manifest.get("cache"))
    cache_ok = bool(cache_path and cache_path.exists())
    cache_entries = 0
    if cache_ok and cache_path:
        for path in cache_path.rglob("*.json"):
            cache_entries += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                cache_ok &= (
                    payload.get("identity", {}).get("cache_format_version")
                    == CACHE_FORMAT_VERSION
                )
            except (OSError, json.JSONDecodeError):
                cache_ok = False
    cache_ok &= cache_entries > 0
    _check(
        checks,
        "cache_compatibility",
        cache_ok,
        f"Found {cache_entries} current-schema cache entries.",
    )

    gitignore_path = repository / ".gitignore"
    gitignore = (
        gitignore_path.read_text(encoding="utf-8")
        if gitignore_path.exists()
        else ""
    )
    missing_ignores = [
        marker for marker in REQUIRED_IGNORE_MARKERS if marker not in gitignore
    ]
    _check(
        checks,
        "artifact_ignores",
        not missing_ignores,
        (
            "Generated outputs, caches, videos, credentials, and calibration "
            "artifacts must be ignored."
        ),
    )
    credentials = credential_presence_check(repository)
    _check(
        checks,
        "credential_hygiene",
        bool(credentials["safe"]),
        (
            f"{len(credentials['tracked_credential_findings'])} tracked and "
            f"{len(credentials['local_sensitive_files'])} local sensitive files found."
        ),
    )

    blocking_failures = [
        check["name"]
        for check in checks
        if check["blocking"] and check["status"] == "fail"
    ]
    return {
        "format_version": READINESS_FORMAT_VERSION,
        "ready": not blocking_failures,
        "blocking_failures": blocking_failures,
        "checks": checks,
        "credential_environment_variables_present": credentials[
            "configured_environment_variables"
        ],
    }
