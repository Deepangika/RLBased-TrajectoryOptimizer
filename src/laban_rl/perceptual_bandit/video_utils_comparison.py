"""GIF-to-MP4 utilities that preserve original motion timing by default."""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageSequence


def find_optimiser_animation(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    candidates = (
        output_dir / "variant_only.mp4",
        output_dir / "arm_comparison.mp4",
        output_dir / "arm_comparison.gif",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find optimiser animation. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _read_gif_frames_and_duration(
    gif_path: Path,
) -> tuple[list[np.ndarray], float]:
    frames: list[np.ndarray] = []
    durations_ms: list[float] = []

    with Image.open(gif_path) as image:
        default_duration_ms = float(image.info.get("duration", 0.0))

        for frame in ImageSequence.Iterator(image):
            array = np.asarray(frame.convert("RGB"), dtype=np.uint8)

            height, width = array.shape[:2]
            if height % 2:
                array = array[:-1, :, :]
            if width % 2:
                array = array[:, :-1, :]

            frames.append(array)
            durations_ms.append(
                float(frame.info.get("duration", default_duration_ms))
            )

    if not frames:
        raise ValueError(f"No frames found in GIF: {gif_path}")

    positive = [d for d in durations_ms if d > 0.0]
    if positive:
        total_duration_seconds = sum(positive) / 1000.0
    else:
        total_duration_seconds = len(frames) / 20.0

    return frames, total_duration_seconds


def ensure_mp4_for_vlm(
    source_path: str | Path,
    *,
    target_duration_seconds: float | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Convert GIF to MP4 while preserving source duration by default."""
    source_path = Path(source_path)

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if source_path.suffix.lower() == ".mp4":
        return source_path

    if source_path.suffix.lower() != ".gif":
        raise ValueError("Expected .gif or .mp4 input.")

    if target_duration_seconds is not None and target_duration_seconds <= 0:
        raise ValueError(
            "target_duration_seconds must be positive when provided."
        )

    if output_path is None:
        output_path = source_path.with_name(
            source_path.stem + "_vlm.mp4"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames, detected_duration = _read_gif_frames_and_duration(source_path)

    output_duration = (
        float(target_duration_seconds)
        if target_duration_seconds is not None
        else detected_duration
    )

    fps = len(frames) / output_duration
    fps = float(np.clip(fps, 1.0, 60.0))

    imageio.mimwrite(
        output_path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    print(
        f"VLM video timing: source ~= {detected_duration:.3f}s | "
        f"output target = {output_duration:.3f}s | fps = {fps:.3f}"
    )

    return output_path
