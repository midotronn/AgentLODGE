#!/usr/bin/env python3
"""Render a (L, 139) motion .npy to a shaded 3D mp4 in an isolated process."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# pyrender renders through EGL (GPU) by default; override with PYRENDER_GL=osmesa for
# software rendering on CPU-only hosts (needs a compatible OSMesa/PyOpenGL build).
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYRENDER_GL", "egl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentlodge-root", required=True)
    parser.add_argument("--motion-npy", required=True)
    parser.add_argument("--output-mp4", required=True)
    parser.add_argument("--lodge-code-path", required=True)
    parser.add_argument("--audio", default="")
    parser.add_argument("--smplx-model-path", default="")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.agentlodge_root).resolve()))
    from agentlodge.video.mesh_render import render_dance_video

    audio = Path(args.audio).resolve() if args.audio else None
    smplx = Path(args.smplx_model_path).resolve() if args.smplx_model_path else None
    out = render_dance_video(
        Path(args.motion_npy).resolve(),
        Path(args.output_mp4).resolve(),
        lodge_code_path=Path(args.lodge_code_path).resolve(),
        audio_path=audio,
        smplx_model_path=smplx,
        img_size=(args.width, args.height),
    )
    print(f"Saved 3D dance video to {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
