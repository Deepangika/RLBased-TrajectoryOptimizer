"""Resumable blinded reference-versus-styled perceptual comparisons."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, Field

from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import LabanOptimisationResult
from laban_rl.perceptual_bandit.environment import Context
from laban_rl.perceptual_bandit.evaluation_cache import (
    EvaluatorCacheIdentity,
    clip_content_sha256,
)
from laban_rl.perceptual_bandit.gemini_evaluator import (
    GeminiProVideoEvaluator,
    _retry_message,
)
from laban_rl.perceptual_bandit.variant_video import shared_camera_limits
from laban_rl.targets import TARGET_PROFILES


PAIR_CACHE_FORMAT_VERSION = 1
PAIR_PROMPT_VERSION = "blinded-target-affect-ab-v1"
PAIR_SCHEMA_VERSION = "paired-affect-preference-v1"


class PairedPreferenceAssessment(BaseModel):
    choice: Literal["A", "B", "neither"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str = Field(min_length=1, max_length=1000)


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reference_first(pair_hash: str, repeat_index: int) -> bool:
    offset = int(pair_hash[:2], 16) % 2
    return (repeat_index + offset) % 2 == 0


class PairedPreferenceCache:
    """Persist each blinded pair judgment immediately."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def identity_payload(
        self,
        *,
        context: Context,
        reference_result: LabanOptimisationResult,
        styled_result: LabanOptimisationResult,
        evaluator_identity: EvaluatorCacheIdentity,
    ) -> dict[str, Any]:
        payload = {
            "cache_format_version": PAIR_CACHE_FORMAT_VERSION,
            "reference_clip_sha256": clip_content_sha256(reference_result),
            "styled_clip_sha256": clip_content_sha256(styled_result),
            "target": context.to_dict(),
            "evaluator": evaluator_identity.to_dict(),
        }
        reference_context = reference_result.raw_result.get(
            "evaluator_render_context"
        )
        styled_context = styled_result.raw_result.get(
            "evaluator_render_context"
        )
        if reference_context != styled_context:
            raise ValueError(
                "Paired clips must use identical evaluator render context."
            )
        if reference_context is not None:
            payload["evaluator_render_context"] = json.loads(
                json.dumps(
                    reference_context,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        return payload

    def cache_key(self, **kwargs: Any) -> str:
        return hashlib.sha256(
            _canonical_bytes(self.identity_payload(**kwargs))
        ).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    @staticmethod
    def _validate_observation(
        observation: Mapping[str, Any],
        *,
        repeat_index: int,
        pair_hash: str,
    ) -> None:
        reference_first = _reference_first(pair_hash, repeat_index)
        expected = (
            ("reference", "styled")
            if reference_first
            else ("styled", "reference")
        )
        if int(observation.get("repeat_index", -1)) != repeat_index:
            raise ValueError("Invalid paired preference repeat index.")
        if (
            observation.get("displayed_a"),
            observation.get("displayed_b"),
        ) != expected:
            raise ValueError("Invalid paired preference display order.")
        raw_choice = observation.get("raw_choice")
        if raw_choice not in {"A", "B", "neither"}:
            raise ValueError("Invalid raw paired preference choice.")
        translated = {
            "A": expected[0],
            "B": expected[1],
            "neither": "neither",
        }[raw_choice]
        if observation.get("choice") != translated:
            raise ValueError("Inconsistent translated paired preference choice.")
        confidence = float(observation.get("confidence", np.nan))
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Invalid paired preference confidence.")

    def collect(
        self,
        *,
        context: Context,
        reference_result: LabanOptimisationResult,
        styled_result: LabanOptimisationResult,
        evaluator: Any,
        repeats: int,
    ) -> list[dict[str, Any]]:
        if repeats < 1:
            raise ValueError("paired preference repeats must be at least 1.")
        evaluator_identity = EvaluatorCacheIdentity.from_evaluator(evaluator)
        identity = self.identity_payload(
            context=context,
            reference_result=reference_result,
            styled_result=styled_result,
            evaluator_identity=evaluator_identity,
        )
        key = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        observations: list[dict[str, Any]] = []
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("cache_key") != key or payload.get("identity") != identity:
                raise ValueError(f"Incompatible paired preference cache: {path}")
            observations = list(payload.get("observations", []))
            for index, observation in enumerate(observations):
                self._validate_observation(
                    observation,
                    repeat_index=index,
                    pair_hash=key,
                )
        while len(observations) < repeats:
            repeat_index = len(observations)
            observation = dict(
                evaluator.evaluate_pair(
                    context,
                    reference_result,
                    styled_result,
                    repeat_index=repeat_index,
                    pair_hash=key,
                )
            )
            self._validate_observation(
                observation,
                repeat_index=repeat_index,
                pair_hash=key,
            )
            observations.append(observation)
            _atomic_json(
                path,
                {
                    "cache_key": key,
                    "identity": identity,
                    "observations": observations,
                },
            )
        return observations[:repeats]


class GeminiPairedPreferenceEvaluator:
    """Ask Gemini which blinded clip better expresses the supplied target."""

    def __init__(self, **kwargs: Any) -> None:
        self.video_evaluator = GeminiProVideoEvaluator(**kwargs)
        self.model = self.video_evaluator.model
        self.temperature = self.video_evaluator.temperature

    def cache_identity(self) -> dict[str, Any]:
        video_identity = self.video_evaluator.cache_identity()
        return {
            "provider": "gemini-paired-preference",
            "model": self.model,
            "prompt_version": PAIR_PROMPT_VERSION,
            "schema_version": PAIR_SCHEMA_VERSION,
            "settings": dict(video_identity["settings"]),
        }

    @staticmethod
    def _prompt(context: Context) -> str:
        target_vad = context.target_vad or {}
        return f"""
You are comparing two robot-arm animation clips, A and B.

Gesture category: {context.gesture.upper()}
Target affect: {context.target_label}
Target VAD:
- valence: {float(target_vad['valence']):.3f}
- arousal: {float(target_vad['arousal']):.3f}
- dominance: {float(target_vad['dominance']):.3f}

The clips are blinded. Do not infer their source from filenames or metadata.
Judge visible movement only. Which animation expresses the target affect more
strongly: A, B, or neither? Choose neither when the difference is not
perceptually meaningful or neither clip expresses the target.
""".strip()

    def evaluate_pair(
        self,
        context: Context,
        reference_result: LabanOptimisationResult,
        styled_result: LabanOptimisationResult,
        *,
        repeat_index: int,
        pair_hash: str,
        max_retries: int = 5,
        initial_backoff: float = 2.0,
    ) -> dict[str, Any]:
        reference_first = _reference_first(pair_hash, repeat_index)
        ordered = (
            (reference_result, styled_result)
            if reference_first
            else (styled_result, reference_result)
        )
        displayed = (
            ("reference", "styled")
            if reference_first
            else ("styled", "reference")
        )
        render_context = reference_result.raw_result.get(
            "evaluator_render_context",
            {},
        )
        saved_limits = render_context.get("camera_limits")
        camera_limits = (
            (
                tuple(float(value) for value in saved_limits[0]),
                tuple(float(value) for value in saved_limits[1]),
            )
            if saved_limits is not None
            else shared_camera_limits(
                [reference_result.q_var, styled_result.q_var],
                duration_seconds=(
                    self.video_evaluator.video_duration_seconds
                ),
                lead_in_seconds=(
                    self.video_evaluator.video_lead_in_seconds
                ),
                repetitions=self.video_evaluator.video_repetitions,
                inter_repeat_transition_seconds=(
                    self.video_evaluator.video_inter_repeat_transition_seconds
                ),
                final_hold_seconds=(
                    self.video_evaluator.video_final_hold_seconds
                ),
            )
        )
        uploads = [
            self.video_evaluator._upload_video(
                self.video_evaluator._prepare_video(
                    result,
                    camera_limits=camera_limits,
                )
            )
            for result in ordered
        ]
        from google.genai import types

        backoff = initial_backoff
        for attempt in range(max_retries + 1):
            try:
                response = self.video_evaluator.client.models.generate_content(
                    model=self.model,
                    contents=[uploads[0], uploads[1], self._prompt(context)],
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        response_mime_type="application/json",
                        response_schema=PairedPreferenceAssessment,
                    ),
                )
                if not response.text:
                    raise RuntimeError("Gemini returned an empty paired response.")
                assessment = PairedPreferenceAssessment.model_validate_json(
                    response.text
                )
                translated = {
                    "A": displayed[0],
                    "B": displayed[1],
                    "neither": "neither",
                }[assessment.choice]
                return {
                    "repeat_index": repeat_index,
                    "displayed_a": displayed[0],
                    "displayed_b": displayed[1],
                    "raw_choice": assessment.choice,
                    "choice": translated,
                    "confidence": float(assessment.confidence),
                    "reasoning_summary": assessment.reasoning_summary,
                }
            except Exception as exc:
                transient = (
                    "503" in str(exc)
                    or "UNAVAILABLE" in str(exc)
                    or "validation error" in str(exc).lower()
                    or "empty paired response" in str(exc).lower()
                )
                if transient and attempt < max_retries:
                    print(_retry_message(backoff, attempt, max_retries))
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise
        raise RuntimeError("Gemini paired preference retry loop exhausted.")

    def close(self) -> None:
        self.video_evaluator.close()


class MockPairedPreferenceEvaluator:
    """Deterministic offline evaluator for cache and orchestration tests."""

    def cache_identity(self) -> dict[str, Any]:
        return {
            "provider": "mock-paired-preference",
            "model": "laban-target-distance",
            "prompt_version": PAIR_PROMPT_VERSION,
            "schema_version": PAIR_SCHEMA_VERSION,
            "settings": {"neither_distance_delta": 0.01},
        }

    def evaluate_pair(
        self,
        context: Context,
        reference_result: LabanOptimisationResult,
        styled_result: LabanOptimisationResult,
        *,
        repeat_index: int,
        pair_hash: str,
    ) -> dict[str, Any]:
        if context.target_state is None:
            raise ValueError(
                "Mock paired preference requires a named target state."
            )
        target = TARGET_PROFILES[context.target_state]

        def distance(result: LabanOptimisationResult) -> float:
            return float(
                np.sqrt(
                    np.mean(
                        [
                            (
                                float(result.achieved_profile[key])
                                - float(target[key])
                            )
                            ** 2
                            for key in FEATURE_KEYS
                        ]
                    )
                )
            )

        reference_distance = distance(reference_result)
        styled_distance = distance(styled_result)
        if abs(reference_distance - styled_distance) <= 0.01:
            choice = "neither"
        else:
            choice = (
                "reference"
                if reference_distance < styled_distance
                else "styled"
            )
        reference_first = _reference_first(pair_hash, repeat_index)
        displayed = (
            ("reference", "styled")
            if reference_first
            else ("styled", "reference")
        )
        raw_choice = (
            "neither"
            if choice == "neither"
            else ("A" if choice == displayed[0] else "B")
        )
        return {
            "repeat_index": repeat_index,
            "displayed_a": displayed[0],
            "displayed_b": displayed[1],
            "raw_choice": raw_choice,
            "choice": choice,
            "confidence": 0.8,
            "reasoning_summary": "Deterministic target-distance comparison.",
        }


@dataclass(frozen=True)
class PreferencePair:
    pair_id: str
    context: Context
    reference_result: LabanOptimisationResult
    styled_result: LabanOptimisationResult
    target_layers: Mapping[str, Any]


def run_paired_preference_experiment(
    pairs: Sequence[PreferencePair],
    *,
    evaluator: Any,
    cache: PairedPreferenceCache,
    repeats: int,
    out_path: str | Path,
) -> dict[str, Any]:
    records = []
    for pair in pairs:
        observations = cache.collect(
            context=pair.context,
            reference_result=pair.reference_result,
            styled_result=pair.styled_result,
            evaluator=evaluator,
            repeats=repeats,
        )
        counts = {
            choice: sum(row["choice"] == choice for row in observations)
            for choice in ("reference", "styled", "neither")
        }
        records.append(
            {
                "pair_id": pair.pair_id,
                "context": pair.context.to_dict(),
                "target_layers": dict(pair.target_layers),
                "repeat_count": len(observations),
                "choice_counts": counts,
                "styled_preference_rate": counts["styled"] / len(observations),
                "reference_preference_rate": (
                    counts["reference"] / len(observations)
                ),
                "neither_rate": counts["neither"] / len(observations),
                "mean_confidence": float(
                    np.mean([row["confidence"] for row in observations])
                ),
                "observations": observations,
            }
        )
        _atomic_json(
            Path(out_path),
            {
                "format_version": 1,
                "status": "in_progress",
                "repeat_count": repeats,
                "records": records,
            },
        )
    payload = {
        "format_version": 1,
        "status": "complete",
        "repeat_count": repeats,
        "records": records,
    }
    _atomic_json(Path(out_path), payload)
    return payload
