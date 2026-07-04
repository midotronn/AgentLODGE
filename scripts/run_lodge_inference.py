#!/usr/bin/env python3
"""Run LODGE inference in an isolated process."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# LODGE imports pyrender (via render.py) at import time; force the headless OSMesa
# platform so PyOpenGL loads without a display. Respect an explicit override.
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentlodge-root", required=True)
    parser.add_argument("--features-npy", required=True)
    parser.add_argument("--output-npy", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--lodge-code-path", required=True)
    parser.add_argument("--lodge-weights-path", required=True)
    parser.add_argument("--lodge-global-weights-path", required=True)
    parser.add_argument("--lodge-genre", default="Hiphop")
    args = parser.parse_args()

    root = Path(args.agentlodge_root).resolve()
    sys.path.insert(0, str(root))

    import numpy as np

    from agentlodge.config import Settings
    from agentlodge.dance.lodge import generate_lodge_dance

    settings = Settings(
        anthropic_api_key=None,
        openai_api_key=None,
        gemini_api_key=None,
        image_backend="openai",
        output_dir=Path(args.work_dir),
        lodge_code_path=Path(args.lodge_code_path),
        edge_code_path=Path(args.lodge_code_path),
        lodge_weights_path=Path(args.lodge_weights_path),
        lodge_global_weights_path=Path(args.lodge_global_weights_path),
        edge_weights_path=Path(args.lodge_weights_path),
        lodge_genre=args.lodge_genre,
        min_audio_seconds=20,
        max_edge_slices=7,
    )

    features = np.load(args.features_npy)
    work_dir = Path(args.work_dir).resolve()
    output_npy = Path(args.output_npy).resolve()
    result = generate_lodge_dance(features, settings, work_dir)
    np.save(output_npy, result.motion.astype("float32"))
    summary_path = work_dir / "lodge_summary.txt"
    summary_path.write_text(result.summary)
    print(f"saved {result.motion.shape} -> {output_npy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
