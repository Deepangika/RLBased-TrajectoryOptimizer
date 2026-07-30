"""Create a metadata-versioned copy of an eligible CEM checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.perceptual_bandit.compatibility import (
    migrate_metadata_only_checkpoint,
    validate_checkpoint,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    source = Path(args.checkpoint)
    destination = Path(args.out)
    if source.resolve() == destination.resolve():
        parser.error("--out must differ from --checkpoint.")
    with source.open("rb") as handle:
        checkpoint = pickle.load(handle)
    migrated, changed = migrate_metadata_only_checkpoint(checkpoint)
    validate_checkpoint(
        migrated,
        expected_kind="cem",
        context=migrated.get("context"),
    )
    if not changed:
        parser.error("Checkpoint already contains current metadata.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(migrated, handle)
    print(f"Wrote migrated checkpoint copy: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
