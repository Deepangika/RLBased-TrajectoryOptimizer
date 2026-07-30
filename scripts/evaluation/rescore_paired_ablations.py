"""Rescore paired cached observations across reward/feasibility settings."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.perceptual_bandit.ablation import write_ablation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-results", required=True)
    parser.add_argument("--configurations", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    payload = write_ablation(
        args.paired_results, args.configurations, args.out
    )
    for configuration in payload["configurations"]:
        suffix = (
            " (inner optimizer rerun required)"
            if configuration["requires_inner_optimizer_rerun"]
            else ""
        )
        print(f"{configuration['name']}{suffix}")
        for metric in ("vad_score", "categorical_score"):
            ranked = sorted(
                (
                    row
                    for row in configuration["records"]
                    if row.get(f"{metric}_rank") is not None
                ),
                key=lambda row: row[f"{metric}_rank"],
            )
            print(
                f"  {metric}: "
                + (
                    ", ".join(row["candidate_id"] for row in ranked)
                    if ranked
                    else "unavailable"
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
