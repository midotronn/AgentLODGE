#!/usr/bin/env python3
"""Extract Jukebox embeddings for EDGE slices in an isolated process."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-root", required=True)
    parser.add_argument("--slice-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()

    edge_root = Path(args.edge_root).resolve()
    slice_dir = Path(args.slice_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(edge_root)
    sys.path.insert(0, str(edge_root))

    from data.audio_extraction.jukebox_features import extract as juke_extract

    wav_slices = sorted(slice_dir.glob("*.wav"), key=lambda p: int(p.stem.split("slice")[-1]))
    if not wav_slices:
        raise SystemExit(f"No wav slices found in {slice_dir}")

    for index, wav_slice in enumerate(wav_slices):
        out_path = cache_dir / f"{wav_slice.stem}.npy"
        if out_path.exists():
            print(f"[{index + 1}/{len(wav_slices)}] cached {out_path.name}")
            continue
        print(f"[{index + 1}/{len(wav_slices)}] extracting {wav_slice.name}")
        reps, _ = juke_extract(str(wav_slice))
        np.save(out_path, np.asarray(reps, dtype=np.float32))
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"done {len(wav_slices)} slices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
