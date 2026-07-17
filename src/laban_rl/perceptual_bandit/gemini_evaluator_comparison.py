"""Gemini Pro video evaluator for expressive robot gestures.

Fix in this version:
- `video_duration_seconds=None` is now valid.
- None means: preserve the original GIF timing during MP4 conversion.
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
from laban_rl.perceptual_bandit.video_utils import (
    ensure_mp4_for_vlm,
    find_optimiser_animation,
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
                self.friendly,
                self.confused,
                self.angry,
            ],
            dtype=float,
        )
        total = float(np.sum(values))

        if not np.isclose(total, 1.0, atol=0.03):
            raise ValueError(
                "State probabilities must sum approximately to 1.0; "
                f"received {total:.6f}."
            )

        expected = STATE_LABELS[int(np.argmax(values))]
        if self.perceived_state != expected:
            raise ValueError(
                "perceived_state must match the highest-probability label. "
                f"Got {self.perceived_state!r}, expected {expected!r}."
            )

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
    """Evaluate generated gesture videos with Gemini Pro or Flash."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-pro",
        api_key: str | None = None,
        temperature: float = 0.7,
        upload_poll_seconds: float = 2.0,
        upload_timeout_seconds: float = 180.0,
        video_duration_seconds: float | None = None,
        keep_uploaded_files: bool = True,
    ) -> None:
        self.model = model
        self.temperature = float(temperature)
        self.upload_poll_seconds = float(upload_poll_seconds)
        self.upload_timeout_seconds = float(upload_timeout_seconds)

        # IMPORTANT:
        # None means preserve the original GIF duration.
        self.video_duration_seconds = (
            None
            if video_duration_seconds is None
            else float(video_duration_seconds)
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
You are evaluating an expressive robot arm gesture from video.

Reference gesture category: {gesture.upper()}

The video contains a blue motion labelled "Reference" and an orange motion
labelled "Variant". Evaluate ONLY the orange Variant motion. Use the blue
Reference only to understand the base gesture category; do not classify it.

Classify the expressive state conveyed by the orange Variant using exactly:
- confident
- calm
- hesitant
- friendly

Base the judgement only on visible movement dynamics and geometry. Consider:
- strength and movement intensity,
- speed, urgency, and decisiveness,
- directness versus indirectness,
- smoothness, control, and boundedness,
- spatial expansion, contraction, and curvature.

Do not evaluate task correctness.
No intended state is provided, so do not infer one from the prompt.
Do not use filenames, folder names, or hidden metadata as evidence.

Return one perceived state, confidence from 0 to 1, a probability for every
candidate state summing to 1, and one brief reasoning summary.
""".strip()

    def _prepare_video(
        self,
        optimisation_result: LabanOptimisationResult,
    ):
        animation_path = find_optimiser_animation(
            optimisation_result.output_dir
        )
        video_path = ensure_mp4_for_vlm(
            animation_path,
            target_duration_seconds=self.video_duration_seconds,
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
    ) -> PerceptualEvaluation:
        video_path = self._prepare_video(optimisation_result)
        uploaded_video = self._upload_video(video_path)

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

    def close(self) -> None:
        if not self.keep_uploaded_files:
            for uploaded in self._upload_cache.values():
                try:
                    self.client.files.delete(name=uploaded.name)
                except Exception:
                    pass

        self._upload_cache.clear()
        self.client.close()
