#!/usr/bin/env python3
"""Run EDGE long-form inference in an isolated process."""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def pkl_to_edge151(pkl_path: Path) -> np.ndarray:
    import torch
    from dataset.quaternion import ax_from_6v

    data = pickle.load(open(pkl_path, "rb"))
    trans = data["smpl_trans"].astype(np.float32)
    poses_aa = data["smpl_poses"].reshape(-1, 24, 3)
    rot_6d = ax_from_6v(torch.from_numpy(poses_aa)).numpy().reshape(len(trans), 144)
    contact = np.zeros((len(trans), 4), dtype=np.float32)
    return np.concatenate([trans, rot_6d, contact], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--features-npy", required=True)
    parser.add_argument("--output-npy", required=True)
    args = parser.parse_args()

    edge_root = Path(args.edge_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    features = np.load(args.features_npy)
    output_npy = Path(args.output_npy).resolve()
    num_clips = len(features)

    os.chdir(edge_root)
    sys.path.insert(0, str(edge_root))

    import torch
    from EDGE import EDGE

    fps = 30
    overlap_seconds = 2.5
    clip_seconds = 5.0
    expected_frames = int(clip_seconds * fps + (num_clips - 1) * overlap_seconds * fps)

    slice_dir = work_dir / "edge_slices"
    wav_slices = sorted(
        slice_dir.glob("*.wav"), key=lambda p: int(p.stem.split("slice")[-1])
    )
    filenames = [str(p) for p in wav_slices[:num_clips]]

    render_dir = work_dir / "edge_renders"
    motion_dir = work_dir / "edge_motions"
    render_dir.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)

    cond = torch.from_numpy(features)
    model = EDGE("jukebox", args.checkpoint)
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
        raise SystemExit(f"No EDGE motion output in {motion_dir}")

    motion = pkl_to_edge151(Path(pkls[0]))
    if motion.shape[0] > expected_frames:
        motion = motion[:expected_frames]

    np.save(str(output_npy), motion.astype(np.float32))
    print(f"saved {motion.shape} -> {output_npy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
