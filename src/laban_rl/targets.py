"""Target Laban profiles for affective states."""

from __future__ import annotations

TARGET_PROFILES = {
    "confident": {
        "weight": 0.80,
        "time": 0.75,
        "flow_boundness": 0.25,
        "space_indirectness": 0.20,
        "shape_arcness": 0.25,
    },
    "calm": {
        "weight": 0.15,
        "time": 0.15,
        "flow_boundness": 0.05,
        "space_indirectness": 0.10,
        "shape_arcness": 0.20,
    },
    "hesitant": {
        "weight": 0.20,
        "time": 0.25,
        "flow_boundness": 0.45,
        "space_indirectness": 0.35,
        "shape_arcness": 0.30,
    },
    "friendly": {
        "weight": 0.40,
        "time": 0.35,
        "flow_boundness": 0.15,
        "space_indirectness": 0.25,
        "shape_arcness": 0.45,
    },
    "confused": {
        "weight": 0.20,
        "time": 0.30,
        "flow_boundness": 0.15,
        "space_indirectness": 0.80,
        "shape_arcness": 0.50,
    },
    "angry": {
        "weight": 0.90,
        "time": 0.85,
        "flow_boundness": 0.85,
        "space_indirectness": 0.10,
        "shape_arcness": 0.15,
    },
}
