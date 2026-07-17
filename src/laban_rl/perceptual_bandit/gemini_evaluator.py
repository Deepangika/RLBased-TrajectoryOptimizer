
"""Gemini evaluator using a same-style, variant-only rendered video.

Drop into:
    src/laban_rl/perceptual_bandit/gemini_evaluator.py
"""
from __future__ import annotations

import os
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from laban_rl.optimiser_api import LabanOptimisationResult
from laban_rl.perceptual_bandit.environment import (
    Context,
    PerceptualEvaluation,
)
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
    "confident",
    "calm",
    "hesitant",
    "friendly",
    "confused",
    "angry",
)


class GeminiGestureAssessment(BaseModel):
    perceived_state: Literal[
        "confident",
        "calm",
        "hesitant",
        "friendly",
        "confused",
        "angry",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    confident: float = Field(ge=0.0, le=1.0)
    calm: float = Field(ge=0.0, le=1.0)
    hesitant: float = Field(ge=0.0, le=1.0)
    friendly: float = Field(ge=0.0, le=1.0)
    confused: float = Field(ge=0.0, le=1.0)
    angry: float = Field(ge=0.0, le=1.0)
    reasoning_summary: str

    @model_validator(mode="after")
    def validate_probabilities(self):
        values = np.asarray(
            [
                self.confident,
                self.calm,
                self.hesitant,
                self.friendly,
                self.confused,
                self.angry,
            ],
            dtype=float,
        )

        total = float(np.sum(values))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("State probabilities must have a finite positive sum.")

        # Gemini commonly emits rounded probabilities (for example a total of
        # 0.95). Treat them as non-negative scores and normalise them rather
        # than throwing away an otherwise usable evaluation.
        values = values / total

        expected = STATE_LABELS[int(np.argmax(values))]
        if self.perceived_state != expected:
            # Instead of failing, auto-correct to match the highest probability
            # This handles cases where Gemini's reasoning and probabilities disagree
            print(
                f"⚠️  Correcting perceived_state: {self.perceived_state!r} → {expected!r} "
                f"(highest probability label)"
            )
            object.__setattr__(self, "perceived_state", expected)

        return self

    def probability_dict(self) -> dict[str, float]:
        values = np.asarray(
            [
                self.confident,
                self.calm,
                self.hesitant,
                self.friendly,
                self.confused,
                self.angry,
            ],
            dtype=float,
        )
        values = values / np.sum(values)

        return {
            label: float(value)
            for label, value in zip(STATE_LABELS, values)
        }


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

        self.last_assessment: GeminiGestureAssessment | None = None
        self.last_video_path: str | None = None

    def _build_prompt(self, gesture: str) -> str:
        return f"""
You are an expert movement evaluator specializing in Laban Movement Analysis (LMA) and Affective Computing. Your task is to look at a trajectory video/GIF of an agent and evaluate its emotional/affective state based on its kinematic properties.

Gesture category: {gesture.upper()}

The video shows only the generated styled robot-arm motion. The intended state is NOT provided.

Your task is to judge which expressive state is most clearly conveyed by the visible movement style.

First, internally assess the visible movement cues:
- speed and urgency
- forcefulness or lightness
- directness versus wavering/indirectness
- smoothness versus interrupted or start-stop motion
- spatial extent and openness
- endpoint commitment: whether the motion appears decisive or uncertain
- hesitation cues: pauses, delays, wavering, retreat, undershoot, or correction

Use the following rubric:

CONFIDENT:
- decisive and committed movement
- relatively direct path
- clear endpoint intention
- stronger or more assertive motion
- may be faster or more forceful
- not wavering or uncertain

CALM:
- smooth, controlled, relaxed movement
- low urgency
- steady timing
- not forceful
- not interrupted
- important: slow and smooth movement should usually be calm, not hesitant

HESITANT:
- visibly uncertain or uncommitted movement
- may include delay, pause, interruption, wavering, retreat, correction, or undershoot
- may appear cautious or unsure
- important: do not classify as hesitant merely because it is slow; there must be visible uncertainty

FRIENDLY:
- open, approachable, inviting movement
- smooth and warm quality
- moderate energy
- rounded or welcoming motion
- not overly forceful or uncertain

CONFUSED:
- uncertain, wandering, directionless movement
- indirect, meandering path with no clear endpoint intention
- free-flowing but disorganised quality
- low force and energy
- may appear lost or searching
- important: not just slow — must appear spatially disoriented

ANGRY:
- forceful, heavy, urgent movement
- fast and sudden with high urgency
- bound (tense, controlled) quality
- direct and focused path
- sharp, angular motion
- strong and assertive, not light or hesitant

Classify the expressive state using exactly one of:
- confident
- calm
- hesitant
- friendly
- confused
- angry

Do not evaluate task correctness.
No intended state is provided.
Do not use filenames, folder names, or hidden metadata as evidence.

Return:
- perceived_state
- confidence from 0 to 1
- probabilities for all 6 states that sum to 1
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

        video_path = render_variant_only_mp4(
            optimisation_result.q_ref,
            optimisation_result.q_var,
            video_path,
            duration_seconds=self.video_duration_seconds,
            fps=self.video_fps,
        )

        self.last_video_path = str(video_path)
        return video_path

    def _upload_video(self, video_path):
        cache_key = str(video_path.resolve())

        if cache_key in self._upload_cache:
            return self._upload_cache[cache_key]

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

        self._upload_cache[cache_key] = uploaded
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
                    probabilities=assessment.probability_dict()
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
                    print(
                        f"⚠ Gemini evaluation failed transiently. Retrying in {backoff:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
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
            for uploaded in self._upload_cache.values():
                try:
                    self.client.files.delete(name=uploaded.name)
                except Exception:
                    pass

        self._upload_cache.clear()
        self.client.close()
