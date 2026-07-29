"""Target Laban profiles for affective states."""

from __future__ import annotations

TARGET_PROFILES = {
    "anger": {
        "weight": 0.90,
        "time": 0.85,
        "flow_boundness": 0.85,
        "space_indirectness": 0.10,
        "shape_arcness": 0.15,
    },
    "disgust": {
        "weight": 0.45,
        "time": 0.40,
        "flow_boundness": 0.70,
        "space_indirectness": 0.15,
        "shape_arcness": 0.20,
    },
    "fear": {
        "weight": 0.35,
        "time": 0.80,
        "flow_boundness": 0.75,
        "space_indirectness": 0.65,
        "shape_arcness": 0.25,
    },
    "happiness": {
        "weight": 0.50,
        "time": 0.55,
        "flow_boundness": 0.20,
        "space_indirectness": 0.30,
        "shape_arcness": 0.70,
    },
    "sadness": {
        "weight": 0.12,
        "time": 0.15,
        "flow_boundness": 0.35,
        "space_indirectness": 0.25,
        "shape_arcness": 0.25,
    },
    "surprise": {
        "weight": 0.62,
        "time": 0.90,
        "flow_boundness": 0.25,
        "space_indirectness": 0.45,
        "shape_arcness": 0.65,
    },
}
