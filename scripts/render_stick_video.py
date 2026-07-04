#!/usr/bin/env python3
"""Render a (L, 139) motion .npy to a stick-figure mp4 in an isolated process."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# LODGE modules pulled in for FK may import pyrender/OpenGL; force headless OSMesa.
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentlodge-root", required=True)
    parser.add_argument("--motion-npy", required=True)
    parser.add_argument("--output-mp4", required=True)
    parser.add_argument("--lodge-code-path", required=True)
    parser.add_argument("--audio", default="")
    parser.add_argument("--smpl-joint-path", default="")
    args = parser.parse_args()

    root = Path(args.agentlodge_root).resolve()
    sys.path.insert(0, str(root))

    from agentlodge.video.stick_figure import render_motion_npy_to_video

    audio = Path(args.audio).resolve() if args.audio else None
    smpl_joint_path = (
        Path(args.smpl_joint_path).resolve() if args.smpl_joint_path else None
    )
    output = render_motion_npy_to_video(
        Path(args.motion_npy).resolve(),
        Path(args.output_mp4).resolve(),
        lodge_code_path=Path(args.lodge_code_path).resolve(),
        audio_path=audio,
        smpl_joint_path=smpl_joint_path,
    )
    print(f"Saved stick figure video to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
