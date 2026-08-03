"""Gemini evaluator using a same-style, variant-only rendered video.

Drop into:
    src/laban_rl/perceptual_bandit/gemini_evaluator.py
"""
from __future__ import annotations

import os
import time
import hashlib
import json
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from laban_rl.config import EMOTION_STATES
from laban_rl.affect import VAD_KEYS
from laban_rl.optimiser_api import LabanOptimisationResult
from laban_rl.perceptual_bandit.environment import (
    Context,
    PerceptualEvaluation,
)
from laban_rl.perceptual_bandit.evaluation_cache import clip_content_sha256
from laban_rl.perceptual_bandit.variant_video import (
    render_variant_only_mp4,
)

try:
    from google import genai
    from google.genai import types
except ImportError as exc:
    raise ImportError(
        "Gemini integration requires:\n"
        "python -m pip install -U google-genai pydantic"
    ) from exc


STATE_LABELS = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
)
PROMPT_VERSION = "lma-vad-category-ambiguity-v2"
SCHEMA_VERSION = "gemini-gesture-assessment-v2"
RENDERER_VERSION = "variant-only-mp4-v1"


class GeminiGestureAssessment(BaseModel):
    valence: float = Field(
        ge=0.0,
        le=1.0,
        description="0 is very unpleasant/negative; 1 is very pleasant/positive.",
    )
    arousal: float = Field(
        ge=0.0,
        le=1.0,
        description="0 is calm/inactive; 1 is highly energized/activated.",
    )
    dominance: float = Field(
        ge=0.0,
        le=1.0,
        description="0 is weak/submissive; 1 is powerful/dominant.",
    )
    perceived_state: Literal[
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
        "ambiguous",
        "none",
    ]
    category_status: Literal["complete", "ambiguous", "missing"]
    confidence: float = Field(ge=0.0, le=1.0)
    anger: float = Field(ge=0.0, le=1.0)
    disgust: float = Field(ge=0.0, le=1.0)
    fear: float = Field(ge=0.0, le=1.0)
    happiness: float = Field(ge=0.0, le=1.0)
    sadness: float = Field(ge=0.0, le=1.0)
    surprise: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str

    @model_validator(mode="after")
    def validate_probabilities(self):
        values = np.asarray(
            [
                self.anger,
                self.disgust,
                self.fear,
                self.happiness,
                self.sadness,
                self.surprise,
            ],
            dtype=float,
        )

        total = float(np.sum(values))
        if not np.isfinite(total) or total < 0.0:
            raise ValueError("State intensities must have a finite non-negative sum.")

        # Gemini commonly emits rounded probabilities (for example a total of
        # 0.95). Treat them as non-negative scores and normalise them rather
        # than throwing away an otherwise usable evaluation.
        if self.category_status == "complete":
            if total <= 0.0:
                raise ValueError(
                    "Complete category output requires positive intensities."
                )
            expected = STATE_LABELS[int(np.argmax(values))]
            if self.perceived_state != expected:
                raise ValueError(
                    "Complete perceived_state must match the strongest intensity."
                )
        elif self.category_status == "ambiguous":
            if self.perceived_state != "ambiguous":
                raise ValueError(
                    "Ambiguous category output requires perceived_state='ambiguous'."
                )
        elif self.perceived_state != "none":
            raise ValueError(
                "Missing category output requires perceived_state='none'."
            )

        return self

    def probability_dict(self) -> dict[str, float]:
        values = np.asarray(
            [
                self.anger,
                self.disgust,
                self.fear,
                self.happiness,
                self.sadness,
                self.surprise,
            ],
            dtype=float,
        )
        total = float(np.sum(values))
        if total <= 0.0 or self.category_status == "missing":
            return {}
        values = values / total

        return {
            label: float(value)
            for label, value in zip(STATE_LABELS, values)
        }

    def affect_dict(self) -> dict[str, float]:
        return {key: float(getattr(self, key)) for key in VAD_KEYS}

    def intensity_dict(self) -> dict[str, float]:
        return {
            label: float(getattr(self, label))
            for label in STATE_LABELS
        }


def _retry_message(backoff: float, attempt: int, max_retries: int) -> str:
    return (
        "WARNING: Gemini evaluation failed transiently. "
        f"Retrying in {backoff:.1f}s "
        f"(attempt {attempt + 1}/{max_retries})..."
    )


class GeminiProVideoEvaluator:
    def __init__(
        self,
        *,
        model: str = "gemini-2.5-pro",
        api_key: str | None = None,
        temperature: float = 0.7,
        upload_poll_seconds: float = 2.0,
        upload_timeout_seconds: float = 180.0,
        video_duration_seconds: float | None = None,
        video_fps: float | None = None,
        keep_uploaded_files: bool = True,
    ) -> None:
        if tuple(EMOTION_STATES) != STATE_LABELS:
            raise ValueError(
                "Gemini evaluator label schema does not match EMOTION_STATES. "
                f"Evaluator={STATE_LABELS}, config={tuple(EMOTION_STATES)}"
            )
        self.model = model
        self.temperature = float(temperature)
        self.upload_poll_seconds = float(upload_poll_seconds)
        self.upload_timeout_seconds = float(upload_timeout_seconds)

        self.video_duration_seconds = (
            2.0
            if video_duration_seconds is None
            else float(video_duration_seconds)
        )
        self.video_fps = (
            None if video_fps is None else float(video_fps)
        )

        self.keep_uploaded_files = bool(keep_uploaded_files)

        resolved_key = (
            api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not resolved_key:
            raise RuntimeError(
                "No Gemini API key found. In PowerShell run:\n"
                '$env:GOOGLE_API_KEY="YOUR_API_KEY"'
            )

        self.client = genai.Client(api_key=resolved_key)
        self._upload_cache: dict[str, object] = {}
        self._prepared_video_hashes: dict[str, str] = {}

        self.last_assessment: GeminiGestureAssessment | None = None
        self.last_video_path: str | None = None

    def cache_identity(self) -> dict[str, object]:
        """Return only non-secret settings that affect evaluator observations."""
        return {
            "provider": "gemini",
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "settings": {
                "temperature": self.temperature,
                "video_duration_seconds": self.video_duration_seconds,
                "video_fps": self.video_fps,
                "renderer_version": RENDERER_VERSION,
                "response_mime_type": "application/json",
            },
        }

    def _build_prompt(self, gesture: str) -> str:
        return f"""
You are an expert movement evaluator specializing in Laban Movement Analysis (LMA) and Affective Computing. Your task is to look at a trajectory video/GIF of an agent and evaluate its emotional/affective state based on its kinematic properties.

Gesture category: {gesture.upper()}

The video shows only the generated styled robot-arm motion. The intended state is NOT provided.

Your primary task is to rate the visible movement continuously on three
affective dimensions. The category judgment is secondary.

First, internally assess the visible movement cues:
- speed and urgency
- forcefulness or lightness
- directness versus wavering/indirectness
- smoothness versus interrupted or start-stop motion
- spatial extent and openness
- endpoint commitment: whether the motion appears decisive or uncertain
- hesitation cues: pauses, delays, wavering, retreat, undershoot, or correction

Rate each primary dimension from 0 to 1:
- valence: 0 = very unpleasant/negative, 1 = very pleasant/positive
- arousal: 0 = calm/inactive, 1 = highly energized/activated
- dominance: 0 = weak/submissive, 1 = powerful/dominant

Use the following movement-based rubric for the secondary category judgment.
Judge only cues visible in the arm
motion; do not infer facial expression, speech, narrative, or task outcome.

ANGER:
- forceful, heavy, tense, and urgent
- fast or sudden acceleration
- direct, focused, sharp, or striking quality
- sustained pressure or bound control

DISGUST:
- visibly avoidant, rejecting, recoiling, or withdrawing
- restrained or tense movement
- may turn away, pull back, interrupt approach, or create distance
- not merely slow: there should be a visible rejection/avoidance quality

FEAR:
- guarded, defensive, apprehensive, or retreating
- may combine quick reactions with hesitation or withdrawal
- tense or bound quality, reduced commitment, or protective contraction
- not merely energetic: there should be visible threat avoidance

HAPPINESS:
- buoyant, lively, open, expansive, or celebratory
- energetic but not aggressive
- smooth, rhythmic, playful, or upward/outward quality
- clear positive engagement rather than withdrawal

SADNESS:
- low-energy, slow, sustained, contracted, or drooping
- reduced amplitude or weak commitment
- subdued, heavy, downward, or withdrawn quality
- not merely calm: it should appear diminished or de-energised

SURPRISE:
- abrupt, sudden, reactive, or startle-like
- rapid expansion, interruption, or change in movement
- high temporal contrast with an immediate response
- not merely fast: there should be an unexpected reactive quality

If one category is clearly strongest, classify it using:
- anger
- disgust
- fear
- happiness
- sadness
- surprise

If multiple categories are similarly plausible, use perceived_state "ambiguous"
and category_status "ambiguous". If no category is supported by visible evidence,
use perceived_state "none" and category_status "missing". Do not force a winner.

Do not evaluate task correctness.
No intended state is provided.
Do not use filenames, folder names, or hidden metadata as evidence.

Return:
- valence, arousal, and dominance as continuous values from 0 to 1
- perceived_state
- category_status
- confidence from 0 to 1
- independent 0-to-1 intensity ratings for all 6 states; they need not sum to 1
- a brief reasoning summary that mentions the key visible cues used
""".strip()

    def _prepare_video(
        self,
        optimisation_result: LabanOptimisationResult,
    ):
        video_path = (
            optimisation_result.output_dir
            / "variant_only_vlm.mp4"
        )
        content_hash = clip_content_sha256(optimisation_result)
        render_identity = hashlib.sha256(
            json.dumps(
                {
                    "content_hash": content_hash,
                    "duration": self.video_duration_seconds,
                    "fps": self.video_fps,
                    "renderer_version": RENDERER_VERSION,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        path_key = str(video_path.resolve())
        marker_path = video_path.with_suffix(".content.sha256")
        persisted_hash = (
            marker_path.read_text(encoding="ascii").strip()
            if marker_path.exists()
            else None
        )

        if (
            not video_path.exists()
            or (
                self._prepared_video_hashes.get(path_key) != render_identity
                and persisted_hash != render_identity
            )
        ):
            video_path = render_variant_only_mp4(
                optimisation_result.q_ref,
                optimisation_result.q_var,
                video_path,
                duration_seconds=self.video_duration_seconds,
                fps=self.video_fps,
                overwrite=True,
            )
            temporary_marker = marker_path.with_suffix(".tmp")
            temporary_marker.write_text(render_identity + "\n", encoding="ascii")
            temporary_marker.replace(marker_path)
        self._prepared_video_hashes[path_key] = render_identity

        self.last_video_path = str(video_path)
        return video_path

    def _upload_video(self, video_path):
        cache_key = str(video_path.resolve())
        current_mtime = video_path.stat().st_mtime_ns

        if cache_key in self._upload_cache:
            cached_mtime, cached_upload = self._upload_cache[cache_key]
            if cached_mtime == current_mtime:
                return cached_upload
            # File has been overwritten since the last upload; discard stale entry.

        uploaded = self.client.files.upload(file=video_path)
        deadline = time.monotonic() + self.upload_timeout_seconds

        while True:
            state_name = (
                uploaded.state.name
                if uploaded.state is not None
                else None
            )

            if state_name == "ACTIVE":
                break

            if state_name == "FAILED":
                raise RuntimeError(
                    f"Gemini failed to process video: {video_path}"
                )

            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out processing Gemini video: {video_path}"
                )

            time.sleep(self.upload_poll_seconds)
            uploaded = self.client.files.get(name=uploaded.name)

        self._upload_cache[cache_key] = (current_mtime, uploaded)
        return uploaded

    def evaluate(
        self,
        context: Context,
        optimisation_result: LabanOptimisationResult,
        max_retries: int = 5,
        initial_backoff: float = 2.0,
    ) -> PerceptualEvaluation:
        """Evaluate video with Gemini, with exponential backoff retry on 503 errors.
        
        max_retries: Maximum number of retry attempts on transient failures (503).
        initial_backoff: Initial backoff in seconds, doubles on each retry.
        """
        video_path = self._prepare_video(optimisation_result)
        uploaded_video = self._upload_video(video_path)

        backoff = initial_backoff
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[
                        uploaded_video,
                        self._build_prompt(context.gesture),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        response_mime_type="application/json",
                        response_schema=GeminiGestureAssessment,
                    ),
                )

                if not response.text:
                    raise RuntimeError("Gemini returned an empty response.")

                assessment = GeminiGestureAssessment.model_validate_json(
                    response.text
                )
                self.last_assessment = assessment
                return PerceptualEvaluation(
                    affect_ratings=assessment.affect_dict(),
                    probabilities=assessment.probability_dict(),
                    confidence=float(assessment.confidence),
                    perceived_state=assessment.perceived_state,
                    category_status=assessment.category_status,
                    category_intensities=assessment.intensity_dict(),
                    reasoning_summary=assessment.reasoning_summary,
                )
                
            except Exception as e:
                last_error = e
                # Retry transient service failures and malformed structured
                # responses. A formatting error should not invalidate the
                # other repeats for an otherwise valid gesture.
                is_503 = "503" in str(e) or "UNAVAILABLE" in str(e)
                is_structured_output_error = any(
                    marker in str(e).lower()
                    for marker in (
                        "validation error",
                        "invalid json",
                        "empty response",
                    )
                )
                
                if (is_503 or is_structured_output_error) and attempt < max_retries:
                    print(_retry_message(backoff, attempt, max_retries))
                    time.sleep(backoff)
                    backoff *= 2  # Exponential backoff
                    continue
                else:
                    # Not a 503, or out of retries
                    raise
        
        # Should never reach here, but fail explicitly if we do
        raise RuntimeError(
            f"Gemini evaluation failed after {max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    def close(self) -> None:
        if not self.keep_uploaded_files:
            for _mtime, uploaded in self._upload_cache.values():
                try:
                    self.client.files.delete(name=uploaded.name)
                except Exception:
                    pass

        self._upload_cache.clear()
        self._prepared_video_hashes.clear()
        self.client.close()