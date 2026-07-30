"""Versioned content-addressed cache for raw perceptual observations."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from laban_rl.affect import validate_vad
from laban_rl.optimiser_api import LabanOptimisationResult
from laban_rl.perceptual_bandit.scoring import observation_to_dict


CACHE_FORMAT_VERSION = 2
_SECRET_MARKERS = ("api_key", "apikey", "secret", "token", "credential", "password")
_SECRET_ENV_NAMES = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


class CacheCompatibilityError(RuntimeError):
    """A cache file exists but does not match its content-addressed identity."""


@dataclass(frozen=True)
class EvaluatorCacheIdentity:
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    settings: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "settings": dict(self.settings),
        }
        _reject_secrets(payload)
        json.dumps(payload, sort_keys=True, allow_nan=False)
        return payload

    @classmethod
    def from_evaluator(cls, evaluator: Any) -> "EvaluatorCacheIdentity":
        method = getattr(evaluator, "cache_identity", None)
        if method is None:
            raise TypeError(
                "Evaluator must provide cache_identity() or an explicit identity."
            )
        payload = dict(method())
        return cls(
            provider=str(payload["provider"]),
            model=str(payload["model"]),
            prompt_version=str(payload["prompt_version"]),
            schema_version=str(payload["schema_version"]),
            settings=dict(payload.get("settings", {})),
        )


def _reject_secrets(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            child_path = f"{path}.{key}" if path else str(key)
            if any(marker in key_text for marker in _SECRET_MARKERS):
                raise ValueError(
                    f"Evaluator cache identity must not contain secrets: {child_path}"
                )
            _reject_secrets(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_secrets(child, f"{path}[{index}]")
    elif isinstance(value, str):
        import os

        configured_secrets = {
            os.environ[name]
            for name in _SECRET_ENV_NAMES
            if os.environ.get(name)
        }
        if value in configured_secrets:
            raise ValueError(
                f"Evaluator cache identity contains a credential value at {path}."
            )


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def clip_content_sha256(result: LabanOptimisationResult) -> str:
    """Hash rendered motion plus evaluator-visible achieved-profile semantics."""
    digest = hashlib.sha256()
    for name, array in (("q_ref", result.q_ref), ("q_var", result.q_var)):
        values = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(name.encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    digest.update(
        _canonical_bytes(
            {
                "achieved_profile": {
                    key: float(value)
                    for key, value in sorted(result.achieved_profile.items())
                }
            }
        )
    )
    return digest.hexdigest()


class PerceptualObservationCache:
    """Persist every successful repeat immediately for interruption-safe reuse."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def identity_payload(
        self,
        *,
        context: Any,
        result: LabanOptimisationResult,
        evaluator_identity: EvaluatorCacheIdentity,
    ) -> dict[str, Any]:
        return {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "clip_content_sha256": clip_content_sha256(result),
            "prompt_context": {"gesture": str(context.gesture)},
            "evaluator": evaluator_identity.to_dict(),
        }

    def cache_key(self, **kwargs: Any) -> str:
        payload = self.identity_payload(**kwargs)
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def _load(
        self,
        path: Path,
        *,
        key: str,
        identity: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheCompatibilityError(
                f"Unreadable evaluator cache entry: {path}"
            ) from exc
        if payload.get("cache_key") != key or payload.get("identity") != identity:
            raise CacheCompatibilityError(
                f"Stale or incompatible evaluator cache entry: {path}"
            )
        observations = payload.get("observations")
        if not isinstance(observations, list):
            raise CacheCompatibilityError(
                f"Evaluator cache observations are invalid: {path}"
            )
        for observation in observations:
            if not isinstance(observation, dict):
                raise CacheCompatibilityError(
                    f"Evaluator cache observation is invalid: {path}"
                )
            try:
                validate_vad(
                    observation.get("affect_ratings", {}),
                    name="Cached evaluator VAD",
                )
                probabilities = observation.get("probabilities", {})
                if not isinstance(probabilities, dict):
                    raise ValueError("probabilities must be an object")
                if probabilities:
                    values = np.asarray(list(probabilities.values()), dtype=float)
                    if (
                        not np.all(np.isfinite(values))
                        or np.any(values < 0.0)
                        or not np.isclose(float(np.sum(values)), 1.0, atol=1e-5)
                    ):
                        raise ValueError("invalid probability vector")
                status = observation.get(
                    "category_status",
                    "complete" if probabilities else "missing",
                )
                if status not in ("complete", "ambiguous", "missing"):
                    raise ValueError("invalid category status")
                if status == "missing" and probabilities:
                    raise ValueError(
                        "missing categorical result contains probabilities"
                    )
                intensities = observation.get("category_intensities", {})
                if not isinstance(intensities, dict):
                    raise ValueError("category_intensities must be an object")
                intensity_values = np.asarray(
                    list(intensities.values()), dtype=float
                )
                if intensities and (
                    not np.all(np.isfinite(intensity_values))
                    or np.any(intensity_values < 0.0)
                    or np.any(intensity_values > 1.0)
                ):
                    raise ValueError("invalid category intensities")
                confidence = observation.get("confidence")
                if confidence is not None and (
                    not np.isfinite(float(confidence))
                    or not 0.0 <= float(confidence) <= 1.0
                ):
                    raise ValueError("invalid confidence")
            except (TypeError, ValueError) as exc:
                raise CacheCompatibilityError(
                    f"Evaluator cache observation is incompatible: {path}"
                ) from exc
        return observations

    def _write(
        self,
        path: Path,
        *,
        key: str,
        identity: Mapping[str, Any],
        observations: list[dict[str, Any]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_key": key,
            "identity": identity,
            "observations": observations,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def collect(
        self,
        *,
        context: Any,
        result: LabanOptimisationResult,
        evaluator: Any,
        repeats: int,
        evaluator_identity: EvaluatorCacheIdentity | None = None,
    ) -> list[dict[str, Any]]:
        if repeats < 1:
            raise ValueError("repeats must be at least 1.")
        resolved_identity = (
            evaluator_identity
            if evaluator_identity is not None
            else EvaluatorCacheIdentity.from_evaluator(evaluator)
        )
        identity = self.identity_payload(
            context=context,
            result=result,
            evaluator_identity=resolved_identity,
        )
        key = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        path = self._path(key)
        observations = (
            self._load(path, key=key, identity=identity) if path.exists() else []
        )
        while len(observations) < repeats:
            repeat_index = len(observations)
            evaluate_repeat = getattr(evaluator, "evaluate_repeat", None)
            if evaluate_repeat is None:
                observation = evaluator.evaluate(context, result)
            else:
                observation = evaluate_repeat(
                    context,
                    result,
                    repeat_index=repeat_index,
                    clip_hash=identity["clip_content_sha256"],
                )
            if hasattr(observation, "validate"):
                observation.validate(getattr(context, "target_state", None))
            observations.append(observation_to_dict(observation))
            self._write(
                path,
                key=key,
                identity=identity,
                observations=observations,
            )
        return observations[:repeats]
