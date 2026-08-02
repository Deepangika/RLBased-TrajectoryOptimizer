"""Explicit compatibility metadata for checkpoints and cached experiments."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from laban_rl.config import FEATURE_KEYS, GESTURE_TYPES
from laban_rl.envs import OBSERVATION_SIZE


CHECKPOINT_FORMAT_VERSION = 2
POLICY_OBSERVATION_SCHEMA = "laban-observation-v2"
CEM_STATE_SCHEMA = "beta-cem-v1"
OUTER_LOOP_SCHEMA = "strict-feasible-cem-v1"


class CheckpointCompatibilityError(RuntimeError):
    """A checkpoint cannot be safely interpreted by the current code."""


def target_fingerprint(context: Mapping[str, Any]) -> str:
    """Fingerprint the complete named/direct-VAD target semantics."""
    canonical = json.dumps(
        dict(context),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def checkpoint_metadata(
    kind: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if kind not in ("cem", "learned_policy"):
        raise ValueError(f"Unknown checkpoint kind {kind!r}.")
    metadata: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": kind,
        "feature_keys": list(FEATURE_KEYS),
        "gesture_types": list(GESTURE_TYPES),
        "state_schema": (
            CEM_STATE_SCHEMA if kind == "cem" else POLICY_OBSERVATION_SCHEMA
        ),
    }
    if kind == "learned_policy":
        metadata["observation_size"] = OBSERVATION_SIZE
    else:
        metadata["outer_loop_schema"] = OUTER_LOOP_SCHEMA
    if context is not None:
        metadata["target_fingerprint"] = target_fingerprint(context)
    return metadata


def migrate_metadata_only_checkpoint(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Add metadata only when payload semantics are already unambiguous."""
    migrated = copy.deepcopy(dict(checkpoint))
    if "metadata" in migrated:
        return migrated, False
    if "policy_state_dict" in migrated:
        raise CheckpointCompatibilityError(
            "Learned-policy checkpoint has no observation-schema metadata. "
            "Its weights may use an older observation shape and must not be "
            "silently reinterpreted; retrain or use the original code version."
        )
    if "cem_state" in migrated:
        raise CheckpointCompatibilityError(
            "Metadata-less CEM checkpoints predate strict feasible-only outer-loop "
            "updates and cannot be migrated safely. Start a new run."
        )
    if not all(key in migrated for key in ("cem_state", "context", "resume_config")):
        raise CheckpointCompatibilityError(
            "Pre-VAD or legacy CEM checkpoint lacks complete context/config "
            "metadata. Start a new run; learned search state cannot be inferred."
        )
    migrated["metadata"] = checkpoint_metadata(
        "cem", context=migrated["context"]
    )
    return migrated, True


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_kind: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise CheckpointCompatibilityError(
            "Checkpoint metadata is missing. Run the metadata migration utility "
            "only for eligible CEM checkpoints."
        )
    if metadata.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            "Checkpoint format version is incompatible with this release."
        )
    if metadata.get("kind") != expected_kind:
        raise CheckpointCompatibilityError(
            f"Expected a {expected_kind!r} checkpoint, found "
            f"{metadata.get('kind')!r}."
        )
    if list(metadata.get("feature_keys", [])) != list(FEATURE_KEYS):
        raise CheckpointCompatibilityError(
            "Checkpoint feature order differs from the current feature schema."
        )
    if expected_kind == "learned_policy":
        if metadata.get("state_schema") != POLICY_OBSERVATION_SCHEMA:
            raise CheckpointCompatibilityError(
                "Learned-policy observation schema is incompatible; weights "
                "cannot be migrated safely."
            )
        if metadata.get("observation_size") != OBSERVATION_SIZE:
            raise CheckpointCompatibilityError(
                "Learned-policy observation shape is incompatible; retraining is required."
            )
    else:
        if metadata.get("state_schema") != CEM_STATE_SCHEMA:
            raise CheckpointCompatibilityError("CEM state schema is incompatible.")
        if metadata.get("outer_loop_schema") != OUTER_LOOP_SCHEMA:
            raise CheckpointCompatibilityError(
                "CEM outer-loop semantics are incompatible; start a new run."
            )
    if context is not None and metadata.get("target_fingerprint") != target_fingerprint(
        context
    ):
        raise CheckpointCompatibilityError(
            "Checkpoint target fingerprint does not match the named/direct-VAD target."
        )
