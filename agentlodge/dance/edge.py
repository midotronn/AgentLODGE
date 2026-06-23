"""EDGE long-form dance generation wrapper."""

from __future__ import annotations

import glob
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentlodge.config import FPS, Settings


@dataclass
class EdgeResult:
    motion: np.ndarray
    summary: str


def _pkl_to_edge151(pkl_path: Path) -> np.ndarray:
    import torch
    from dataset.quaternion import ax_from_6v

    data = pickle.load(open(pkl_path, "rb"))
    trans = data["smpl_trans"].astype(np.float32)
    poses_aa = data["smpl_poses"].reshape(-1, 24, 3)
    rot_6d = ax_from_6v(torch.from_numpy(poses_aa)).numpy().reshape(len(trans), 144)
    contact = np.zeros((len(trans), 4), dtype=np.float32)
    return np.concatenate([trans, rot_6d, contact], axis=1)


def generate_edge_dance(
    wav_path: Path,
    edge_slices: list[np.ndarray],
    settings: Settings,
    work_dir: Path,
) -> EdgeResult:
    """Run EDGE long-form generation with 5s clips and 2.5s overlap."""
    edge_root = settings.edge_code_path
    if not edge_root.exists():
        raise FileNotFoundError(f"EDGE codebase not found at {edge_root}")

    os.chdir(edge_root)
    sys.path.insert(0, str(edge_root))

    import torch
    from EDGE import EDGE

    cond = torch.from_numpy(np.array(edge_slices))
    num_clips = len(edge_slices)
    overlap_seconds = 2.5
    clip_seconds = 5.0
    expected_frames = int(
        clip_seconds * FPS + (num_clips - 1) * overlap_seconds * FPS
    )

    slice_dir = work_dir / "edge_slices"
    wav_slices = sorted(slice_dir.glob("*.wav"), key=lambda p: int(p.stem.split("slice")[-1]))
    filenames = [str(p) for p in wav_slices[:num_clips]]

    render_dir = work_dir / "edge_renders"
    motion_dir = work_dir / "edge_motions"
    render_dir.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)

    model = EDGE("jukebox", str(settings.edge_weights_path))
    model.eval()

    data_tuple = None, cond, filenames
    model.render_sample(
        data_tuple,
        "test",
        str(render_dir),
        render_count=-1,
        fk_out=str(motion_dir),
        render=False,
    )

    pkls = sorted(glob.glob(str(motion_dir / "test_*.pkl")))
    if not pkls:
        raise RuntimeError(f"EDGE generation produced no motion files in {motion_dir}")

    motion = _pkl_to_edge151(Path(pkls[0]))
    if motion.shape[0] > expected_frames:
        motion = motion[:expected_frames]

    summary = (
        f"EDGE long-form pipeline with {num_clips} chained 5s clips "
        f"and 2.5s overlap; output length {motion.shape[0]} frames."
    )
    return EdgeResult(motion=motion, summary=summary)
