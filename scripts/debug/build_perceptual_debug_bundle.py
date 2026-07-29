from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

CASE_SPECS = [
    {
        "case_id": "point_confident_seed_7",
        "role": "positive_control",
        "gesture": "point",
        "state": "confident",
        "seed": 7,
        "dest_name": "point_confident_seed_7",
    },
    {
        "case_id": "wave_friendly_seed_41",
        "role": "borderline_case",
        "gesture": "wave",
        "state": "friendly",
        "seed": 41,
        "dest_name": "wave_friendly_seed_41",
    },
    {
        "case_id": "point_confused_seed_7",
        "role": "failure_case",
        "gesture": "point",
        "state": "confused",
        "seed": 7,
        "dest_name": "point_confused_seed_7",
        "fallback": {"gesture": "reach", "state": "confused", "seed": 7},
    },
]

SOURCE_FILE_CANDIDATES = [
    "src/laban_rl/perceptual_bandit/gemini_evaluator.py",
    "src/laban_rl/perceptual_bandit/environment.py",
    "src/laban_rl/perceptual_bandit/cem.py",
    "src/laban_rl/perceptual_bandit/selection.py",
    "src/laban_rl/perceptual_bandit/variant_video.py",
    "scripts/train_cem_contextual_bandit.py",
    "scripts/debug/evaluate_fixed_profile_holdout.py",
    "scripts/evaluation/run_comprehensive_evaluation.py",
]

REPORT_FILE_PATTERNS = ["*.csv", "*.json", "*.png"]

# Never copy these paths/files into the bundle.
EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "ENV",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
EXCLUDE_FILE_EXTS = {".pyc", ".pyo", ".pyd"}
EXCLUDE_FILE_GLOBS = [
    "*.env",
    ".env",
    ".env.*",
    "*secret*",
    "*token*",
    "*credential*",
    "*apikey*",
    "*api_key*",
    "*.key",
    "id_rsa",
    "id_dsa",
]

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}
TEXT_EXTS = {
    ".txt",
    ".md",
    ".json",
    ".csv",
    ".log",
    ".yaml",
    ".yml",
    ".ini",
    ".toml",
    ".py",
}

REQUIRED_TOP_LEVEL_CASE_FILES = [
    "results_summary.json",
    "independent_validation.json",
    "selected_validated_profile.json",
    "fixed_profile_holdout.json",
]

# Include these artefacts broadly for each experiment case.
GENERAL_INCLUDE_EXTS = {
    ".json",
    ".csv",
    ".txt",
    ".log",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".mp4",
    ".mov",
    ".avi",
    ".webm",
    ".gif",
    ".pt",
}

# Restrict heavyweight optimiser subfolders to only useful final artefacts.
ALLOWED_OPTIMISER_OUTPUT_FILES = {
    "variant_only_vlm.mp4",
    "resolved_target_profile.json",
    "optimisation_history.csv",
    "generation_convergence.csv",
    "feature_comparison.png",
    "joint_comparison.png",
    "wrist_path_comparison.png",
    "best_summary.txt",
}

SECRET_REGEXES = [
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"(?i)sk-[0-9a-z]{20,}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"]?[0-9A-Za-z_\-]{16,}"),
    re.compile(r"(?i)token\s*[:=]\s*['\"]?[0-9A-Za-z_\-]{16,}"),
    re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY-----"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a perceptual-diagnostics bundle from existing CEM outputs."
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Repository root path (for example Updated_CEM_Live_Retest_V2).",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/final_clean_evaluation_v1",
        help="Evaluation output root relative to project root.",
    )
    parser.add_argument(
        "--bundle-dir",
        default="diagnostic_bundle",
        help="Destination folder name or path for the diagnostic bundle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing bundle directory and zip.",
    )
    return parser.parse_args()


def is_excluded_file(path: Path) -> bool:
    if path.suffix.lower() in EXCLUDE_FILE_EXTS:
        return True
    name = path.name
    for pat in EXCLUDE_FILE_GLOBS:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def has_excluded_parent(path: Path) -> bool:
    return any(part in EXCLUDE_DIR_NAMES for part in path.parts)


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise RuntimeError(
                f"Bundle directory already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def iter_report_files(report_dir: Path) -> Iterable[Path]:
    if not report_dir.exists():
        return []
    files: list[Path] = []
    for file_path in report_dir.iterdir():
        if not file_path.is_file():
            continue
        if is_excluded_file(file_path):
            continue
        if any(fnmatch.fnmatch(file_path.name, pat) for pat in REPORT_FILE_PATTERNS):
            files.append(file_path)
    return sorted(files)


def resolve_experiment_dir(output_root: Path, gesture: str, state: str, seed: int) -> Path:
    return output_root / "outer" / gesture / state / f"seed_{seed}"


def collect_case_files(exp_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not exp_dir.exists() or not exp_dir.is_dir():
        return files

    for path in exp_dir.rglob("*"):
        if not path.is_file():
            continue
        if has_excluded_parent(path):
            continue
        if is_excluded_file(path):
            continue

        suffix = path.suffix.lower()
        rel_parts = path.relative_to(exp_dir).parts
        if suffix not in GENERAL_INCLUDE_EXTS:
            continue

        if "optimiser_outputs" in rel_parts:
            if path.name not in ALLOWED_OPTIMISER_OUTPUT_FILES:
                continue

        files.append(path)

    return sorted(set(files))


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict | list | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def find_files_by_name(exp_dir: Path, names: set[str]) -> list[str]:
    if not exp_dir.exists():
        return []
    found: list[str] = []
    for p in exp_dir.rglob("*"):
        if p.is_file() and p.name in names:
            found.append(str(p.relative_to(exp_dir)).replace("\\", "/"))
    return sorted(found)


def find_files_by_glob(exp_dir: Path, patterns: list[str]) -> list[str]:
    if not exp_dir.exists():
        return []
    found: set[str] = set()
    for p in exp_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(exp_dir)).replace("\\", "/")
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(p.name, pat) for pat in patterns):
            found.add(rel)
    return sorted(found)


def extract_reward_equation() -> dict[str, str]:
    return {
        "per_repeat_perceptual_reward": (
            "perceptual_reward = target_probability_weight * target_probability + "
            "margin_weight * effective_margin"
        ),
        "outer_reward": (
            "outer_reward = mean_perceptual_reward - realisation_penalty_weight * realisation_rmse "
            "- stability_penalty_weight * perceptual_reward_std "
            "- max_feature_error_penalty_weight * max(0.0, max_abs_feature_error - max_feature_error_threshold)"
        ),
    }


def extract_prompt_location() -> dict[str, str]:
    return {
        "file": "src/laban_rl/perceptual_bandit/gemini_evaluator.py",
        "class": "GeminiProVideoEvaluator",
        "function": "_build_prompt",
    }


def gather_case_model_and_repeat_info(case_summary: dict | None, case_holdout: dict | None) -> dict[str, object]:
    summary = case_summary if isinstance(case_summary, dict) else {}
    holdout = case_holdout if isinstance(case_holdout, dict) else {}
    return {
        "training_model": summary.get("model"),
        "training_temperature": summary.get("temperature"),
        "training_repeats": summary.get("training_repeats") or summary.get("repeats_per_round"),
        "validation_repeats": summary.get("validation_repeats"),
        "holdout_model": holdout.get("model"),
        "holdout_temperature": holdout.get("temperature"),
        "planned_holdout_repeats": holdout.get("planned_repeats"),
        "completed_holdout_repeats": holdout.get("completed_repeats"),
    }


def write_readme(bundle_dir: Path, manifest: dict) -> None:
    prompt_info = manifest["code_evidence"].get("prompt_location", {})
    reward_eq = manifest["code_evidence"].get("reward_equation", {})
    cases = manifest.get("cases", [])

    lines: list[str] = []
    lines.append("# Perceptual Debug Diagnostic Bundle")
    lines.append("")
    lines.append("## Scope")
    lines.append("This bundle was created by copying existing files only.")
    lines.append("No CEM rerun, no external API calls, and no modifications to original experiment outputs were performed.")
    lines.append("")

    lines.append("## Case Mapping")
    for case in cases:
        role = case.get("role", "case")
        cfg = case.get("training_validation_holdout_config", {})
        lines.append(
            f"- {role}: gesture={case.get('gesture')}, state={case.get('state')}, seed={case.get('seed')}"
        )
        lines.append(
            f"  source_dir={case.get('original_experiment_directory')}"
        )
        lines.append(
            f"  exists={case.get('directory_exists')}"
        )
        if case.get("substitution_used"):
            lines.append(
                f"  substitution={case.get('substitution_used')}"
            )
        lines.append(
            "  training(model,temp,repeats)="
            f"({cfg.get('training_model')}, {cfg.get('training_temperature')}, {cfg.get('training_repeats')})"
        )
        lines.append(
            "  validation_repeats="
            f"{cfg.get('validation_repeats')}"
        )
        lines.append(
            "  holdout(model,temp,planned,completed)="
            f"({cfg.get('holdout_model')}, {cfg.get('holdout_temperature')}, "
            f"{cfg.get('planned_holdout_repeats')}, {cfg.get('completed_holdout_repeats')})"
        )
    lines.append("")

    lines.append("## Collected Artifacts")
    lines.append("- Source files for evaluator/prompt/environment/CEM/selection/holdout/report scripts.")
    lines.append("- Per-case experiment artifacts under experiments/<case_name>/ preserving relative structure.")
    lines.append("- Consolidated report CSV/JSON/PNG files under report/.")
    lines.append("- SHA-256 hashes for all copied files in SHA256SUMS.txt.")
    lines.append("")

    lines.append("## Missing Artifacts")
    any_missing = False
    for case in cases:
        missing = case.get("missing_requested_artifacts", [])
        if missing:
            any_missing = True
            lines.append(f"- {case.get('case_id')}: {', '.join(missing)}")
    if not any_missing:
        lines.append("- None detected for requested categories in the selected directories.")
    lines.append("")

    lines.append("## Candidate-Level Re-ranking Feasibility")
    lines.append(
        "Candidate-level reward re-ranking is possible when sample summaries and/or training history are present "
        "(for example rounds/*/sample_summary.json and training_history.csv)."
    )
    lines.append("See manifest.json per-case candidate-history listings for exact availability.")
    lines.append("")

    lines.append("## Prompt Auditing Feasibility")
    lines.append(
        "Prompt auditing against frozen videos is possible when fixed_profile_holdout.json and corresponding holdout/optimiser_outputs "
        "videos are present in each case directory."
    )
    lines.append("")

    lines.append("## Outer Reward Equation (Code Evidence)")
    lines.append("Per-repeat perceptual reward:")
    lines.append(f"- {reward_eq.get('per_repeat_perceptual_reward')}")
    lines.append("Final outer reward:")
    lines.append(f"- {reward_eq.get('outer_reward')}")
    lines.append("")

    lines.append("## Gemini Prompt Location")
    lines.append(f"- file: {prompt_info.get('file')}")
    lines.append(f"- class: {prompt_info.get('class')}")
    lines.append(f"- function: {prompt_info.get('function')}")
    lines.append("")

    lines.append("## Training vs Validation vs Holdout")
    lines.append("- Training evaluation: uses repeat_evaluations = training_repeats during CEM sample scoring.")
    lines.append("- Independent validation: re-evaluates informed/final/shortlisted candidates using validation_repeats.")
    lines.append("- Frozen holdout: evaluates the selected fixed profile post-selection with planned_holdout_repeats.")
    lines.append("- Per-case model/temperature/repeat values are listed in manifest.json under each case.")
    lines.append("")

    lines.append("## Security Notes")
    lines.append("- Excluded: .env, credential/token/key-like files, __pycache__, .pyc, virtualenv folders, and VCS metadata.")
    lines.append("- Copied text files were scanned for secret-like strings; findings are listed in manifest.json.")
    lines.append("")

    (bundle_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sha256_sums(bundle_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    lines: list[str] = []
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "SHA256SUMS.txt":
            continue
        rel = str(path.relative_to(bundle_dir)).replace("\\", "/")
        digest = sha256_of_file(path)
        rows.append({"path": rel, "sha256": digest})
        lines.append(f"{digest}  {rel}")
    (bundle_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def scan_for_secrets(bundle_dir: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_EXTS:
            continue
        rel = str(path.relative_to(bundle_dir)).replace("\\", "/")
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for regex in SECRET_REGEXES:
            for match in regex.finditer(text):
                findings.append(
                    {
                        "path": rel,
                        "pattern": regex.pattern,
                        "match_preview": match.group(0)[:80],
                    }
                )
    return findings


def make_zip(bundle_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                arcname = str(path.relative_to(bundle_dir.parent)).replace("\\", "/")
                zf.write(path, arcname=arcname)


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve()

    bundle_dir_arg = Path(args.bundle_dir)
    bundle_dir = (
        bundle_dir_arg
        if bundle_dir_arg.is_absolute()
        else project_root / bundle_dir_arg
    ).resolve()
    zip_path = bundle_dir.with_suffix(".zip")

    if not output_root.exists():
        raise RuntimeError(f"Output root does not exist: {output_root}")

    if bundle_dir.exists() and not args.overwrite:
        raise RuntimeError(
            f"Bundle directory already exists: {bundle_dir}. Use --overwrite to replace it."
        )
    if zip_path.exists() and not args.overwrite:
        raise RuntimeError(
            f"Bundle zip already exists: {zip_path}. Use --overwrite to replace it."
        )

    ensure_clean_dir(bundle_dir, overwrite=args.overwrite)

    source_dst = bundle_dir / "source"
    experiments_dst = bundle_dir / "experiments"
    report_dst = bundle_dir / "report"
    source_dst.mkdir(parents=True, exist_ok=True)
    experiments_dst.mkdir(parents=True, exist_ok=True)
    report_dst.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "bundle_dir": str(bundle_dir),
        "bundle_zip": str(zip_path),
        "cases": [],
        "source_files_copied": [],
        "source_files_missing": [],
        "report_files_copied": [],
        "report_files_missing": [],
        "excluded_rules": {
            "exclude_dir_names": sorted(EXCLUDE_DIR_NAMES),
            "exclude_file_exts": sorted(EXCLUDE_FILE_EXTS),
            "exclude_file_globs": sorted(EXCLUDE_FILE_GLOBS),
            "excluded_examples": [".env", "credentials", "tokens", "keys", "__pycache__", ".pyc", "virtual environments", ".git metadata"],
        },
        "substitutions": [],
        "assumptions": [
            "Experiment directories follow outputs/<run>/outer/<gesture>/<state>/seed_<seed>.",
            "Independent validation and selection metadata are in independent_validation.json and selected_validated_profile.json.",
            "Holdout completion is read from fixed_profile_holdout.json completed_repeats.",
        ],
        "code_evidence": {
            "prompt_location": extract_prompt_location(),
            "reward_equation": extract_reward_equation(),
        },
        "security_scan": {
            "secret_like_findings": [],
        },
    }

    # Source files
    for rel in SOURCE_FILE_CANDIDATES:
        src = project_root / rel
        if src.exists() and src.is_file() and src.suffix == ".py":
            dst = source_dst / rel
            copy_file(src, dst)
            manifest["source_files_copied"].append(str(dst.relative_to(bundle_dir)).replace("\\", "/"))
        else:
            manifest["source_files_missing"].append(rel)

    # Experiment cases
    for spec in CASE_SPECS:
        gesture = spec["gesture"]
        state = spec["state"]
        seed = spec["seed"]
        primary_dir = resolve_experiment_dir(output_root, gesture, state, seed)
        selected_dir = primary_dir
        substitution_desc = None

        if not primary_dir.exists() and spec.get("fallback"):
            fb = spec["fallback"]
            fallback_dir = resolve_experiment_dir(
                output_root, fb["gesture"], fb["state"], fb["seed"]
            )
            if fallback_dir.exists():
                selected_dir = fallback_dir
                substitution_desc = {
                    "requested": {
                        "gesture": gesture,
                        "state": state,
                        "seed": seed,
                    },
                    "used": {
                        "gesture": fb["gesture"],
                        "state": fb["state"],
                        "seed": fb["seed"],
                    },
                    "reason": "requested case directory missing",
                }
                manifest["substitutions"].append(substitution_desc)

        case_dest = experiments_dst / spec["dest_name"]
        copied_dest_paths: list[str] = []

        if selected_dir.exists() and selected_dir.is_dir():
            for src_file in collect_case_files(selected_dir):
                rel_from_case = src_file.relative_to(selected_dir)
                dst_file = case_dest / rel_from_case
                copy_file(src_file, dst_file)
                copied_dest_paths.append(str(dst_file.relative_to(bundle_dir)).replace("\\", "/"))

        summary = read_json(selected_dir / "results_summary.json") if selected_dir.exists() else None
        holdout = read_json(selected_dir / "fixed_profile_holdout.json") if selected_dir.exists() else None

        holdout_completed = None
        if isinstance(holdout, dict):
            value = holdout.get("completed_repeats")
            holdout_completed = int(value) if isinstance(value, (int, float)) else value

        video_files = []
        if selected_dir.exists():
            video_files = sorted(
                str(p.relative_to(selected_dir)).replace("\\", "/")
                for p in selected_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS
            )

        candidate_history = find_files_by_glob(
            selected_dir,
            [
                "training_history.csv",
                "rounds/*/sample_summary.json",
                "latest_checkpoint.pt",
                "best_profile.json",
                "selected_validated_profile.json",
                "independent_validation.json",
                "results_summary.json",
            ],
        )

        per_repeat_vlm_records = find_files_by_glob(
            selected_dir,
            [
                "fixed_profile_holdout.json",
                "independent_validation.json",
                "rounds/*/sample_summary.json",
            ],
        )

        requested_category_checks = {
            "results_summary.json": bool((selected_dir / "results_summary.json").exists()),
            "independent_validation.json": bool((selected_dir / "independent_validation.json").exists()),
            "selected_validated_profile.json": bool((selected_dir / "selected_validated_profile.json").exists()),
            "fixed_profile_holdout.json": bool((selected_dir / "fixed_profile_holdout.json").exists()),
            "metadata_json_files": len(find_files_by_glob(selected_dir, ["*.json"])) > 0,
            "cem_training_history": len(find_files_by_glob(selected_dir, ["training_history.csv", "rounds/*/sample_summary.json"])) > 0,
            "candidate_archive": len(find_files_by_glob(selected_dir, ["rounds/*/sample_summary.json"])) > 0,
            "validated_candidate_pool": len(find_files_by_glob(selected_dir, ["independent_validation.json", "selected_validated_profile.json"])) > 0,
            "per_repeat_gemini_records": len(per_repeat_vlm_records) > 0,
            "requested_and_achieved_profiles": len(find_files_by_glob(selected_dir, ["*profile*.json", "*resolved_target_profile.json"])) > 0,
            "feature_errors_and_realisation_metrics": len(find_files_by_glob(selected_dir, ["*feature*.png", "*feature*.json", "*realisation*.csv", "*realisation*.json"])) > 0,
            "reference_initial_selected_candidate_videos": len(video_files) > 0,
            "plots": len(find_files_by_glob(selected_dir, ["*.png", "*.svg"])) > 0,
            "warning_or_evaluator_failure_logs": len(find_files_by_glob(selected_dir, ["*.log", "*warning*.txt", "*error*.txt"])) > 0,
        }

        missing_requested = [key for key, ok in requested_category_checks.items() if not ok]

        model_repeat_info = gather_case_model_and_repeat_info(
            summary if isinstance(summary, dict) else None,
            holdout if isinstance(holdout, dict) else None,
        )

        case_entry = {
            "case_id": spec["case_id"],
            "role": spec["role"],
            "gesture": gesture,
            "state": state,
            "seed": seed,
            "original_experiment_directory": str(primary_dir),
            "resolved_experiment_directory": str(selected_dir),
            "directory_exists": bool(selected_dir.exists() and selected_dir.is_dir()),
            "substitution_used": substitution_desc,
            "selected_profile_file": "selected_validated_profile.json" if (selected_dir / "selected_validated_profile.json").exists() else None,
            "holdout_file": "fixed_profile_holdout.json" if (selected_dir / "fixed_profile_holdout.json").exists() else None,
            "completed_holdout_repeats": holdout_completed,
            "available_video_files": video_files,
            "available_candidate_history_files": candidate_history,
            "available_per_repeat_vlm_records": per_repeat_vlm_records,
            "missing_requested_artifacts": missing_requested,
            "copied_destination_paths": copied_dest_paths,
            "training_validation_holdout_config": model_repeat_info,
        }

        manifest["cases"].append(case_entry)

    # Report files
    report_dir = output_root / "report"
    report_files = list(iter_report_files(report_dir))
    if not report_dir.exists():
        manifest["report_files_missing"].append(str(report_dir))
    else:
        for src in report_files:
            dst = report_dst / src.name
            copy_file(src, dst)
            manifest["report_files_copied"].append(str(dst.relative_to(bundle_dir)).replace("\\", "/"))

        for required_name in (
            "overall_metrics.json",
            "holdout_seed_results.csv",
            "holdout_context_summary.csv",
        ):
            if not (report_dir / required_name).exists():
                manifest["report_files_missing"].append(required_name)

    # Initial manifest write, then README, then security scan + hashes + final manifest.
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    write_readme(bundle_dir, manifest)

    secret_findings = scan_for_secrets(bundle_dir)
    manifest["security_scan"]["secret_like_findings"] = secret_findings

    sha_rows = write_sha256_sums(bundle_dir)
    manifest["sha256_files"] = sha_rows

    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    # Hash file changes after manifest rewrite; regenerate hashes once to include final manifest hash.
    sha_rows = write_sha256_sums(bundle_dir)
    manifest["sha256_files"] = sha_rows
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    # Final hash pass so SHA256SUMS reflects final manifest.
    write_sha256_sums(bundle_dir)

    make_zip(bundle_dir, zip_path)

    missing_summary: list[str] = []
    missing_summary.extend(manifest.get("source_files_missing", []))
    missing_summary.extend(manifest.get("report_files_missing", []))
    for case in manifest.get("cases", []):
        cid = case.get("case_id")
        for missing in case.get("missing_requested_artifacts", []):
            missing_summary.append(f"{cid}:{missing}")

    print("Perceptual debug bundle build complete.")
    print(f"Bundle directory: {bundle_dir}")
    print(f"Bundle zip: {zip_path}")
    print(f"Cases processed: {len(manifest.get('cases', []))}")
    print(f"Source files copied: {len(manifest.get('source_files_copied', []))}")
    print(f"Report files copied: {len(manifest.get('report_files_copied', []))}")
    print(f"Secret-like findings: {len(secret_findings)}")
    print(f"Missing artifact entries: {len(missing_summary)}")
    if missing_summary:
        preview = missing_summary[:20]
        print("Missing summary (first up to 20):")
        for item in preview:
            print(f"- {item}")


if __name__ == "__main__":
    main()
