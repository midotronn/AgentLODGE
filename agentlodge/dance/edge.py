"""EDGE long-form dance generation wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentlodge.config import FPS, Settings
from agentlodge.subprocess_runner import run_edge_inference_subprocess


@dataclass
class EdgeResult:
    motion: np.ndarray
    summary: str


def generate_edge_dance(
    wav_path: Path,
    edge_slices: list[np.ndarray],
    settings: Settings,
    work_dir: Path,
    *,
    seed: int | None = None,
) -> EdgeResult:
    """Run EDGE long-form generation in an isolated EDGE subprocess.

    ``seed`` seeds the EDGE diffusion sampler for reproducible / seed-diverse output (best-of-K).
    """
    edge_root = settings.edge_code_path
    if not edge_root.exists():
        raise FileNotFoundError(f"EDGE codebase not found at {edge_root}")

    work_dir.mkdir(parents=True, exist_ok=True)
    features_path = work_dir / "edge_features.npy"
    output_path = work_dir / "edge_motion.npy"
    np.save(features_path, np.array(edge_slices))

    num_clips = len(edge_slices)
    overlap_seconds = 2.5
    clip_seconds = 5.0
    expected_frames = int(
        clip_seconds * FPS + (num_clips - 1) * overlap_seconds * FPS
    )

    run_edge_inference_subprocess(
        edge_root,
        settings.edge_weights_path,
        work_dir,
        features_path,
        output_path,
        seed=seed,
    )

    motion = np.load(output_path).astype(np.float32)
    if motion.shape[0] > expected_frames:
        motion = motion[:expected_frames]

    summary = (
        f"EDGE long-form pipeline with {num_clips} chained 5s clips "
        f"and 2.5s overlap; output length {motion.shape[0]} frames."
    )
    return EdgeResult(motion=motion, summary=summary)
