"""
LLM-assisted Laban profile generation.

This module keeps LLM use outside the optimisation/RL loop. The LLM is used once
at the start of an experiment to propose a normalised target profile. After that,
the optimiser sees the same plain dictionary used by the existing predefined
TARGET_PROFILES setup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from .config import FEATURE_KEYS, GESTURE_TYPES
from .targets import TARGET_PROFILES


LabanProfileDict = Dict[str, float]


def _profile_schema() -> dict:
    feature_properties = {
        key: {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": f"Normalised target value for {key} on a 0 to 1 scale.",
        }
        for key in FEATURE_KEYS
    }

    return {
        "type": "object",
        "properties": {
            "target_name": {"type": "string"},
            "gesture_type": {"type": "string", "enum": GESTURE_TYPES},
            "context": {"type": "string"},
            "laban_profile": {
                "type": "object",
                "properties": feature_properties,
                "required": FEATURE_KEYS,
                "additionalProperties": False,
            },
            "rationale": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": [
            "target_name",
            "gesture_type",
            "context",
            "laban_profile",
            "rationale",
            "confidence",
        ],
        "additionalProperties": False,
    }


def validate_laban_profile(profile: Mapping[str, Any]) -> LabanProfileDict:
    """Validate that a generated profile exactly matches FEATURE_KEYS and [0, 1]."""
    missing = [key for key in FEATURE_KEYS if key not in profile]
    extra = [key for key in profile.keys() if key not in FEATURE_KEYS]
    if missing or extra:
        raise ValueError(
            f"Invalid Laban profile keys. Missing={missing}; extra={extra}; "
            f"expected={FEATURE_KEYS}"
        )

    validated: LabanProfileDict = {}
    for key in FEATURE_KEYS:
        try:
            value = float(profile[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Laban profile value for {key!r} is not numeric: {profile[key]!r}") from exc

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Laban profile value for {key!r} must be in [0, 1], got {value}")

        validated[key] = value

    return validated


def load_laban_profile_json(path: str | Path) -> Tuple[LabanProfileDict, dict]:
    """Load either a raw profile dict or a full LLM response JSON file."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if "laban_profile" in data:
        profile = validate_laban_profile(data["laban_profile"])
        metadata = {key: value for key, value in data.items() if key != "laban_profile"}
    else:
        profile = validate_laban_profile(data)
        metadata = {"source": "json", "path": str(path)}

    return profile, metadata


def save_laban_profile_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Save a generated profile and metadata so runs are reproducible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_laban_profile_prompt(
    *,
    target_name: str,
    gesture_type: str,
    context: str,
    reference_features: Mapping[str, float] | None,
) -> str:
    reference_text = "not provided"
    if reference_features:
        reference_text = json.dumps(
            {key: reference_features.get(key) for key in FEATURE_KEYS},
            indent=2,
        )

    return f"""
Generate a computational Laban-inspired target profile for this robot gesture experiment.

Target affect / communicative state: {target_name}
Gesture type: {gesture_type}
Interaction context: {context or "general socially assistive robot gesture"}
Reference trajectory normalised features: {reference_text}

Return normalised values in [0, 1] for exactly these features:
- weight: energy/force proxy from motion dynamics. Higher = stronger/more forceful.
- time: urgency/suddenness proxy from acceleration. Higher = quicker/more sudden.
- flow_boundness: jerk-based boundness. Higher = more controlled, restrained, or interrupted; lower = freer/smoother.
- space_indirectness: path-length-to-displacement proxy. Higher = more indirect/detoured; lower = more direct.
- shape_arcness: curvature/arc quality. Higher = more arcing/rounded; lower = straighter/more linear.

Important constraints:
- These are not full formal LMA labels. They are numerical reward targets for this codebase.
- Do not output values purely from emotion stereotypes. Consider the gesture type and whether preserving recognisability matters.
- Avoid extreme 0.0 or 1.0 values unless the affect strongly requires it.
- For wave gestures, space_indirectness may be unstable in this implementation, but still provide a value for schema consistency.
- Prefer targets that are plausible for a 2D shoulder/elbow arm with limited motion range.
""".strip()


def generate_laban_profile_openai(
    *,
    target_name: str,
    gesture_type: str,
    context: str = "",
    reference_features: Mapping[str, float] | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> dict:
    """
    Call OpenAI once to generate a structured target profile.

    Requires:
        pip install openai
        set OPENAI_API_KEY=...
    """
    if gesture_type not in GESTURE_TYPES:
        raise ValueError(f"gesture_type must be one of {GESTURE_TYPES}, got {gesture_type!r}")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Set it before using --profile-source llm.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("The openai package is required for LLM profiles. Install with: pip install openai") from exc

    client = OpenAI()

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You map robot communicative states to validated, normalised "
                    "Laban-inspired reward targets for a 2D robot-arm trajectory styler. "
                    "Return only the requested structured object."
                ),
            },
            {
                "role": "user",
                "content": build_laban_profile_prompt(
                    target_name=target_name,
                    gesture_type=gesture_type,
                    context=context,
                    reference_features=reference_features,
                ),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "laban_target_profile",
                "strict": True,
                "schema": _profile_schema(),
            }
        },
        temperature=temperature,
    )

    content = getattr(response, "output_text", None)
    if not content:
        # Fallback for SDK response shapes that expose nested content items.
        try:
            content = response.output[0].content[0].text
        except Exception as exc:  # pragma: no cover - defensive SDK compatibility path
            raise RuntimeError(f"Could not extract text from OpenAI response: {response!r}") from exc

    data = json.loads(content)
    data["laban_profile"] = validate_laban_profile(data["laban_profile"])
    data["source"] = "openai"
    data["model"] = model
    return data


def resolve_target_profile(
    *,
    target_name: str,
    gesture_type: str,
    profile_source: str = "predefined",
    profile_json: str | Path | None = None,
    context: str = "",
    reference_features: Mapping[str, float] | None = None,
    llm_model: str = "gpt-4o-mini",
    llm_temperature: float = 0.2,
    refresh_llm_profile: bool = False,
) -> Tuple[LabanProfileDict, dict]:
    """
    Resolve a target profile from predefined values, JSON, or an LLM.

    Returns:
        (profile, metadata)
    """
    profile_source = profile_source.strip().lower()

    if profile_source == "predefined":
        if target_name not in TARGET_PROFILES:
            raise ValueError(
                f"Unknown predefined target {target_name!r}. Available: {list(TARGET_PROFILES.keys())}. "
                "Use --profile-source llm for a new affective state."
            )
        return validate_laban_profile(TARGET_PROFILES[target_name]), {
            "source": "predefined",
            "target_name": target_name,
        }

    if profile_source == "json":
        if profile_json is None:
            raise ValueError("--profile-json is required when --profile-source json")
        profile, metadata = load_laban_profile_json(profile_json)
        metadata.setdefault("source", "json")
        metadata.setdefault("target_name", target_name)
        return profile, metadata

    if profile_source == "llm":
        if profile_json is not None and Path(profile_json).exists() and not refresh_llm_profile:
            profile, metadata = load_laban_profile_json(profile_json)
            metadata.setdefault("source", "llm-cache")
            metadata.setdefault("target_name", target_name)
            return profile, metadata

        payload = generate_laban_profile_openai(
            target_name=target_name,
            gesture_type=gesture_type,
            context=context,
            reference_features=reference_features,
            model=llm_model,
            temperature=llm_temperature,
        )
        profile = validate_laban_profile(payload["laban_profile"])
        if profile_json is not None:
            save_laban_profile_json(profile_json, payload)
        return profile, payload

    raise ValueError("profile_source must be one of: predefined, json, llm")
